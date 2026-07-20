// Package agent implements the server-side agentic tool-call loop.
//
// When confidential-ai is configured with one or more upstream MCP servers
// (per-fleet `private-rag`, `enclave-cloud`, `container-app-lightpanda`,
// etc.), incoming chat completion requests are intercepted: the proxy
// fetches the available tools from each server, advertises them to vLLM,
// detects `tool_calls` in the model's reply, dispatches each call to the
// owning MCP server, appends the result back into the messages list, and
// re-invokes vLLM. The loop is bounded so a misbehaving model cannot
// burn tokens forever.
//
// The HTTP-level glue (request/response shape, SSE event emission) lives
// in internal/handler. This package is the headless engine and is fully
// unit-testable without spinning up vLLM.
package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/privasys/confidential-ai/internal/agent/mcpsse"
)

// TransportPrivasysHTTP is the legacy in-house MCP transport: GET
// <base>/api/v1/mcp/tools and POST <base>/api/v1/mcp/tools/<name>.
// TransportMCPSSE is the standard MCP-over-SSE transport (long-lived).
const (
	TransportPrivasysHTTP = "privasys_http"
	TransportMCPSSE       = "mcp_sse"
)

// AuthMode controls how the proxy mints / forwards a bearer per call.
const (
	AuthModeForward  = "forward"  // forward end-user JWT verbatim
	AuthModeExchange = "exchange" // mint via /api/v1/internal/token-exchange
	AuthModeStatic   = "static"   // inject from StaticBearer
	AuthModeNone     = "none"     // send no Authorization header
	// AuthModeAssistant is the §8.7 RAG-in-enclave path to Privasys Drive:
	// the call carries `Authorization: Assistant <AssistantToken>` plus an
	// `X-Privasys-On-Behalf-Of: <sub>` header naming the end user, so Drive
	// runs its read-only, AI-scoped RAG surface for that user.
	AuthModeAssistant = "assistant"
)

// Server describes a single upstream MCP server the proxy talks to.
type Server struct {
	// Name is the per-server prefix the orchestrator prepends to the
	// underlying tool name when advertising to the model. Two MCP servers
	// can each expose a `search` tool without collision: the model sees
	// `private_rag__search` and `lightpanda__browse`. The double
	// underscore is preserved by every tokenizer we care about.
	Name string

	// BaseURL is the root of the MCP server (no trailing slash). For
	// in-fleet servers this is typically the per-app DNS name on the
	// fleet's private network (e.g. http://rag-db:8443 in compose,
	// or the privasys.org subdomain for cross-fleet calls).
	BaseURL string

	// Transport is one of TransportPrivasysHTTP (default for backward
	// compat) or TransportMCPSSE.
	Transport string

	// AuthMode is one of forward|exchange|static|none. Empty defaults to
	// forward to keep dev wiring simple, but every production tool
	// configured via management-service uses 'exchange'.
	AuthMode string

	// AuthAudience is the target audience for AuthModeExchange. Required
	// when AuthMode == "exchange".
	AuthAudience string

	// AuthScopes is the requested scope set for AuthModeExchange.
	AuthScopes []string

	// StaticBearer is the literal token sent when AuthMode == "static".
	StaticBearer string

	// AssistantToken is the shared secret sent as `Authorization: Assistant
	// <token>` when AuthMode == "assistant" (§8.7 RAG-in-enclave, interim
	// gate). Paired with the acting user's sub in X-Privasys-On-Behalf-Of.
	AssistantToken string

	// BearerForward is the legacy flag that maps to AuthMode == forward.
	// Kept so existing MCP_SERVERS env strings still work; new code
	// should set AuthMode directly.
	BearerForward bool

	// RequiresUserConfirmation, if true, every tool call this server
	// produces will be flagged so the front-end displays a
	// <ConfirmToolCall> modal before the call is dispatched. Used for
	// write tools (`add_to_data_room`, `upload`, etc.). The actual
	// per-tool flag comes from the catalogue; this is the per-server
	// default when the tool descriptor omits it.
	RequiresUserConfirmation bool

	// ExpectedDigest is the attested workload digest (OID 3.2 value, bare
	// hex) of the tool's enclave, carried by the tool-grant for enclave
	// tools. Empty for EXTERNAL (off-platform) tools — its absence is what
	// routes the server onto the WebPKI+SSRF-guarded transport instead of
	// attested RA-TLS (see KindRouter).
	ExpectedDigest string
}

// effectiveAuthMode normalises AuthMode + the legacy BearerForward flag.
func (s Server) effectiveAuthMode() string {
	if s.AuthMode != "" {
		return s.AuthMode
	}
	if s.BearerForward {
		return AuthModeForward
	}
	return AuthModeNone
}

// Tool is the catalogue entry for a single MCP tool, augmented with the
// server it belongs to. Mirrors the JSON-Schema descriptor `private-rag`
// publishes at GET /api/v1/mcp/tools.
type Tool struct {
	Server                   string          `json:"-"`
	Name                     string          `json:"name"`
	Description              string          `json:"description"`
	InputSchema              json.RawMessage `json:"input_schema"`
	RequiresUserConfirmation bool            `json:"requires_user_confirmation,omitempty"`
}

// QualifiedName returns the "<server>__<tool>" name advertised to vLLM.
func (t Tool) QualifiedName() string { return t.Server + "__" + t.Name }

// SplitQualifiedName parses a "<server>__<tool>" name back to its two
// halves. Returns ok=false if the input does not contain the separator.
func SplitQualifiedName(qualified string) (server, tool string, ok bool) {
	const sep = "__"
	i := strings.Index(qualified, sep)
	if i < 0 {
		return "", "", false
	}
	return qualified[:i], qualified[i+len(sep):], true
}

// Catalog fetches and caches tool descriptors from each configured MCP
// server. Concurrent reads are safe; refreshes happen lazily.
type Catalog struct {
	servers []Server
	client  *http.Client
	ttl     time.Duration
	logf    func(string, ...any)

	mu        sync.Mutex
	lastFetch time.Time
	cached    []Tool
	cachedErr error

	sseMu      sync.Mutex
	sseClients map[string]*mcpsse.Client
}

// NewCatalog returns a catalogue that refreshes every ttl. A zero ttl is
// treated as 60s. The client is reused for every fetch.
func NewCatalog(servers []Server, client *http.Client, ttl time.Duration) *Catalog {
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Second}
	}
	if ttl <= 0 {
		ttl = 60 * time.Second
	}
	return &Catalog{
		servers:    servers,
		client:     client,
		ttl:        ttl,
		sseClients: map[string]*mcpsse.Client{},
		logf:       func(string, ...any) {},
	}
}

// SetLogger installs a logger for SSE adapter diagnostics. Optional.
func (c *Catalog) SetLogger(logf func(string, ...any)) {
	if logf != nil {
		c.logf = logf
	}
}

// sseClient returns the persistent MCP-SSE client for s, creating it on
// first use. Safe for concurrent callers.
func (c *Catalog) sseClient(s Server) *mcpsse.Client {
	c.sseMu.Lock()
	defer c.sseMu.Unlock()
	if cli, ok := c.sseClients[s.Name]; ok {
		return cli
	}
	cli := mcpsse.New(s.BaseURL, nil, c.logf)
	c.sseClients[s.Name] = cli
	return cli
}

// Close tears down all persistent SSE sessions.
func (c *Catalog) Close() {
	c.sseMu.Lock()
	defer c.sseMu.Unlock()
	for _, cli := range c.sseClients {
		cli.Close()
	}
	c.sseClients = map[string]*mcpsse.Client{}
}

// Replace atomically swaps the configured server list. The tool cache
// is invalidated so the next Tools() call repopulates from the new set,
// and any persistent SSE sessions are torn down (they'll be re-opened
// lazily against the new servers on next use). Callers may pass nil or
// an empty slice to disable tools entirely.
//
// Safe to call concurrently with Tools(); the next reader observes the
// new set atomically.
func (c *Catalog) Replace(servers []Server) {
	dup := make([]Server, len(servers))
	copy(dup, servers)
	c.mu.Lock()
	c.servers = dup
	c.lastFetch = time.Time{}
	c.cached = nil
	c.cachedErr = nil
	c.mu.Unlock()
	c.Close()
}

// Tools returns the merged tool list. If the cache is fresh, it is
// returned as-is; otherwise every server is queried in serial (small N).
// On per-server failure, that server's tools are dropped from the result
// and the error is logged via the returned error (best-effort: a
// successful response is still returned if at least one server replied).
func (c *Catalog) Tools(ctx context.Context) ([]Tool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.lastFetch.IsZero() && time.Since(c.lastFetch) < c.ttl {
		return c.cached, c.cachedErr
	}

	var (
		all   []Tool
		errs  []string
		anyOK bool
	)
	for _, s := range c.servers {
		tools, err := c.fetchTools(ctx, s)
		if err != nil {
			// Loud per-server: a failing server otherwise just vanishes
			// from the model's tool list ("I don't have that tool") with
			// nothing in the logs when the other servers are healthy.
			log.Printf("[agent] tool fetch failed for %q (%s): %v", s.Name, s.BaseURL, err)
			errs = append(errs, fmt.Sprintf("%s: %v", s.Name, err))
			continue
		}
		anyOK = true
		all = append(all, tools...)
	}

	c.lastFetch = time.Now()
	c.cached = all
	if !anyOK && len(errs) > 0 {
		c.cachedErr = errors.New(strings.Join(errs, "; "))
	} else {
		c.cachedErr = nil
	}
	return c.cached, c.cachedErr
}

// Server looks up a configured server by name.
func (c *Catalog) Server(name string) (Server, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.servers {
		if s.Name == name {
			return s, true
		}
	}
	return Server{}, false
}

// Servers returns a copy of the configured server list. Used by callers
// that need to compute attestation digests over the catalogue.
func (c *Catalog) Servers() []Server {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Server, len(c.servers))
	copy(out, c.servers)
	return out
}

// ServersDigest is the canonical sha256 over the configured server set
// (sorted by name; only the security-relevant fields contribute). The
// hex-encoded result is bound into RA-TLS attestation extension OID
// 1.3.6.1.4.1.65230.3.5.7 so a verifier can prove which tool servers the
// confidential-ai container was configured to talk to.
func (c *Catalog) ServersDigest() string {
	servers := c.Servers()
	type canon struct {
		Name      string `json:"name"`
		BaseURL   string `json:"base_url"`
		Transport string `json:"transport"`
		AuthMode  string `json:"auth_mode"`
		Audience  string `json:"audience,omitempty"`
		Confirm   bool   `json:"confirm,omitempty"`
	}
	cs := make([]canon, 0, len(servers))
	for _, s := range servers {
		t := s.Transport
		if t == "" {
			t = TransportPrivasysHTTP
		}
		cs = append(cs, canon{
			Name:      s.Name,
			BaseURL:   s.BaseURL,
			Transport: t,
			AuthMode:  s.effectiveAuthMode(),
			Audience:  s.AuthAudience,
			Confirm:   s.RequiresUserConfirmation,
		})
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].Name < cs[j].Name })
	body, _ := json.Marshal(cs)
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

// Tool looks up a single cached tool by qualified name ("<server>__<tool>").
// The cache is NOT refreshed on miss: the caller should already have
// driven Tools(ctx) at least once before invoking this.
func (c *Catalog) Tool(qualifiedName string) (Tool, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, t := range c.cached {
		if t.QualifiedName() == qualifiedName {
			return t, true
		}
	}
	return Tool{}, false
}

func (c *Catalog) fetchTools(ctx context.Context, s Server) ([]Tool, error) {
	if s.Transport == TransportMCPSSE {
		return c.fetchToolsSSE(ctx, s)
	}
	return fetchToolsPrivasysHTTP(ctx, c.client, s)
}

func (c *Catalog) fetchToolsSSE(ctx context.Context, s Server) ([]Tool, error) {
	cli := c.sseClient(s)
	mtools, err := cli.ListTools(ctx, c.ttl)
	if err != nil {
		return nil, err
	}
	out := make([]Tool, 0, len(mtools))
	for _, t := range mtools {
		out = append(out, Tool{
			Server:                   s.Name,
			Name:                     t.Name,
			Description:              t.Description,
			InputSchema:              t.InputSchema,
			RequiresUserConfirmation: s.RequiresUserConfirmation,
		})
	}
	return out, nil
}

func fetchToolsPrivasysHTTP(ctx context.Context, client *http.Client, s Server) ([]Tool, error) {
	url := strings.TrimRight(s.BaseURL, "/") + "/api/v1/mcp/tools"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var payload struct {
		Tools []Tool `json:"tools"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	for i := range payload.Tools {
		payload.Tools[i].Server = s.Name
		if s.RequiresUserConfirmation {
			payload.Tools[i].RequiresUserConfirmation = true
		}
	}
	return payload.Tools, nil
}
