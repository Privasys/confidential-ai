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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Server describes a single upstream MCP server the proxy talks to.
//
// Convention: each server exposes
//   GET  <BaseURL>/api/v1/mcp/tools             -> {"tools": [Tool, ...]}
//   POST <BaseURL>/api/v1/mcp/tools/<tool_name> -> arbitrary JSON result
//
// Same shape served by Privasys/private-rag.
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

	// BearerForward, if true, forwards the end-user's Authorization
	// header from the original chat request to the MCP server. Required
	// for `private-rag` (per-user conversations + ownership). For
	// stateless tools like `lightpanda.browse` it can stay false.
	BearerForward bool

	// RequiresUserConfirmation, if true, every tool call this server
	// produces will be flagged so the front-end displays a
	// <ConfirmToolCall> modal before the call is dispatched. Used for
	// write tools (`add_to_data_room`, `upload`, etc.). The actual
	// per-tool flag comes from the catalogue; this is the per-server
	// default when the tool descriptor omits it.
	RequiresUserConfirmation bool
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

	mu        sync.Mutex
	lastFetch time.Time
	cached    []Tool
	cachedErr error
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
	return &Catalog{servers: servers, client: client, ttl: ttl}
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
		all  []Tool
		errs []string
		anyOK bool
	)
	for _, s := range c.servers {
		tools, err := fetchTools(ctx, c.client, s)
		if err != nil {
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
	for _, s := range c.servers {
		if s.Name == name {
			return s, true
		}
	}
	return Server{}, false
}

func fetchTools(ctx context.Context, client *http.Client, s Server) ([]Tool, error) {
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
