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
		return errResult(qualifiedName, "malformed tool name (expected <server>__<tool>)", started)
	}
	srv, ok := d.catalog.Server(server)
	if !ok {
		return errResult(qualifiedName, fmt.Sprintf("unknown MCP server %q", server), started)
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
	if srv.BearerForward && bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
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

// ToolResult is what the dispatcher returns for one call. JSON-tagged so
// it can be emitted directly to the SSE channel.
type ToolResult struct {
	Name       string          `json:"name"`
	Status     string          `json:"status"` // "ok" | "error"
	Result     json.RawMessage `json:"result,omitempty"`
	Error      string          `json:"error,omitempty"`
	DurationMs int64           `json:"duration_ms"`
}

// AsToolMessageContent renders the result as a string suitable for the
// `content` of an OpenAI-compatible `tool` role message.
func (r ToolResult) AsToolMessageContent() string {
	if r.Status == "error" {
		// Surface the error to the model so it can recover.
		b, _ := json.Marshal(map[string]string{"error": r.Error})
		return string(b)
	}
	return string(r.Result)
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
