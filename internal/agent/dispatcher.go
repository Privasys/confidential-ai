package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Dispatcher executes a single tool call against the owning MCP server.
type Dispatcher struct {
	catalog *Catalog
	client  *http.Client
}

// NewDispatcher wraps a Catalog. The client is reused for every call.
func NewDispatcher(catalog *Catalog, client *http.Client) *Dispatcher {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	return &Dispatcher{catalog: catalog, client: client}
}

// Call invokes the named tool. `qualifiedName` is the "<server>__<tool>"
// shape advertised to vLLM. `args` is the JSON object the model produced.
// `bearer`, if non-empty, is forwarded to MCP servers configured with
// BearerForward=true (the user's session JWT).
//
// The returned ToolResult is suitable for both:
//   - inclusion in the next vLLM call (as the `content` of a `tool` role
//     message), and
//   - emission as an SSE `tool_result` event to the front-end.
func (d *Dispatcher) Call(ctx context.Context, qualifiedName string, args json.RawMessage, bearer string) ToolResult {
	started := time.Now()

	server, tool, ok := SplitQualifiedName(qualifiedName)
	if !ok {
		// The model frequently calls a tool by its FRIENDLY name — the
		// grant's server name ("kv_store") or the tool's bare name,
		// often with underscores where the tool uses dashes — especially
		// on the text-rescue path. Refusing outright surfaced as opaque
		// tool failures in the chat; resolve unambiguous bare names
		// instead and only error when the reference is genuinely
		// ambiguous or unknown.
		if q, rok := d.resolveBareName(ctx, qualifiedName); rok {
			server, tool, _ = SplitQualifiedName(q)
			qualifiedName = q
		} else {
			return errResult(qualifiedName, "unknown tool name (expected <server>__<tool>)", started)
		}
	}
	srv, ok := d.catalog.Server(server)
	if !ok {
		return errResult(qualifiedName, fmt.Sprintf("unknown MCP server %q", server), started)
	}

	if srv.Transport == TransportMCPSSE {
		return d.callSSE(ctx, qualifiedName, srv, tool, args, bearer, started)
	}

	url := strings.TrimRight(srv.BaseURL, "/") + "/api/v1/mcp/tools/" + tool
	body := args
	if len(body) == 0 {
		body = []byte("{}")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return errResult(qualifiedName, err.Error(), started)
	}
	req.Header.Set("Content-Type", "application/json")
	if d.authHeader(srv, bearer) != "" {
		req.Header.Set("Authorization", d.authHeader(srv, bearer))
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return errResult(qualifiedName, err.Error(), started)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20)) // 4 MiB cap

	if resp.StatusCode >= 400 {
		return errResult(qualifiedName, fmt.Sprintf("status %d: %s", resp.StatusCode, summarize(respBody)), started)
	}
	// Validate it's JSON; if not, wrap as a string field.
	if !json.Valid(respBody) {
		respBody, _ = json.Marshal(map[string]string{"text": string(respBody)})
	}

	return ToolResult{
		Name:       qualifiedName,
		Status:     "ok",
		Result:     respBody,
		DurationMs: time.Since(started).Milliseconds(),
	}
}

// resolveBareName maps an unqualified tool reference to a unique
// "<server>__<tool>" name. Resolution ladder:
//  1. a unique exact tool-name match across the catalogue;
//  2. a unique match after underscore/dash normalisation (models write
//     kv_store for a tool named kv-store);
//  3. a server whose catalogue has exactly one tool.
//
// Anything ambiguous resolves to false — silently picking one of several
// candidates would run a tool the model did not intend.
func (d *Dispatcher) resolveBareName(ctx context.Context, name string) (string, bool) {
	tools, _ := d.catalog.Tools(ctx)
	norm := func(s string) string { return strings.ReplaceAll(s, "_", "-") }

	var exact, fuzzy, srvTools []Tool
	for _, t := range tools {
		if t.Name == name {
			exact = append(exact, t)
		}
		if norm(t.Name) == norm(name) {
			fuzzy = append(fuzzy, t)
		}
		if t.Server == name {
			srvTools = append(srvTools, t)
		}
	}
	switch {
	case len(exact) == 1:
		return exact[0].QualifiedName(), true
	case len(fuzzy) == 1:
		return fuzzy[0].QualifiedName(), true
	case len(srvTools) == 1:
		return srvTools[0].QualifiedName(), true
	}
	return "", false
}

// authHeader resolves the Authorization header value for an outbound
// tool call based on the server's auth_mode. The `exchange` mode is
// meant to mint a tool-scoped JWT via the Privasys IdP at privasys.id
// (jwt-bearer grant + audience:<auth_audience> scope). Until that
// IdP-side flow lands `exchange` falls back to `forward` semantics so
// existing in-fleet calls continue to work end-to-end.
// TODO(ai-tools): mint via Privasys IdP - NOT via management-service.
func (d *Dispatcher) authHeader(srv Server, bearer string) string {
	switch srv.effectiveAuthMode() {
	case AuthModeForward, AuthModeExchange:
		if bearer == "" {
			return ""
		}
		return "Bearer " + bearer
	case AuthModeStatic:
		if srv.StaticBearer == "" {
			return ""
		}
		return "Bearer " + srv.StaticBearer
	default:
		return ""
	}
}

// callSSE dispatches via the persistent MCP-over-SSE client.
func (d *Dispatcher) callSSE(ctx context.Context, qualifiedName string, srv Server, tool string, args json.RawMessage, bearer string, started time.Time) ToolResult {
	cli := d.catalog.sseClient(srv)
	hp := func() http.Header {
		auth := d.authHeader(srv, bearer)
		if auth == "" {
			return nil
		}
		h := http.Header{}
		h.Set("Authorization", auth)
		return h
	}
	res, err := cli.CallTool(ctx, tool, args, hp)
	if err != nil {
		return errResult(qualifiedName, err.Error(), started)
	}
	// Surface IsError from the MCP result as our error status.
	if res.IsError {
		text := ""
		for _, b := range res.Content {
			if b.Type == "text" {
				text = b.Text
				break
			}
		}
		if text == "" {
			text = "tool reported error"
		}
		return errResult(qualifiedName, text, started)
	}
	// Use the raw JSON-RPC result so the model sees the full MCP shape
	// (content blocks etc).
	out := res.Raw
	if len(out) == 0 || !json.Valid(out) {
		out, _ = json.Marshal(map[string]any{"content": res.Content})
	}
	return ToolResult{
		Name:       qualifiedName,
		Status:     "ok",
		Result:     out,
		DurationMs: time.Since(started).Milliseconds(),
	}
}

// ToolResult is what the dispatcher returns for one call. JSON-tagged so
// it can be emitted directly to the SSE channel.
type ToolResult struct {
	Name       string          `json:"name"`
	Status     string          `json:"status"` // "ok" | "error"
	Result     json.RawMessage `json:"result,omitempty"`
	Error      string          `json:"error,omitempty"`
	DurationMs int64           `json:"duration_ms"`
}

// MaxToolResultBytes caps the size of a single tool-result payload
// re-fed into the model's context. A large LightPanda page fetch or
// RAG response would otherwise blow past the model's `max_model_len`,
// leaving no room for the assistant's reply — the user-visible symptom
// is "thought process shown, no final answer" because the entire token
// budget gets consumed in the reasoning channel before any content can
// be emitted. The SSE `tool_result` event still carries the full
// payload to the front-end, so this cap is invisible to the user.
const MaxToolResultBytes = 32 * 1024

// AsToolMessageContent renders the result as a string suitable for the
// `content` of an OpenAI-compatible `tool` role message.
func (r ToolResult) AsToolMessageContent() string {
	if r.Status == "error" {
		// Surface the error to the model so it can recover.
		b, _ := json.Marshal(map[string]string{"error": r.Error})
		return string(b)
	}
	s := string(r.Result)
	if len(s) > MaxToolResultBytes {
		s = s[:MaxToolResultBytes] + "\n\n[truncated: tool result exceeded 32KB; ask the user to narrow the query if more detail is needed]"
	}
	return s
}

func errResult(name, msg string, started time.Time) ToolResult {
	return ToolResult{
		Name:       name,
		Status:     "error",
		Error:      msg,
		DurationMs: time.Since(started).Milliseconds(),
	}
}

func summarize(body []byte) string {
	const max = 256
	s := strings.TrimSpace(string(body))
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// ErrToolNotAllowed is returned (via ToolResult.Error) when the catalogue
// rejects a call (e.g. a write tool without explicit user confirmation).
var ErrToolNotAllowed = errors.New("tool requires user confirmation")
