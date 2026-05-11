package mcpsse

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeMCP is a tiny in-memory MCP-over-SSE server for tests.
type fakeMCP struct {
	t *testing.T

	mu       sync.Mutex
	stream   chan string // SSE events to push to the connected client
	tools    []Tool
	gotCalls atomic.Int32
	closeAll func()
}

func newFakeMCP(t *testing.T, tools []Tool) (*fakeMCP, *httptest.Server) {
	f := &fakeMCP{t: t, stream: make(chan string, 32), tools: tools}

	mux := http.NewServeMux()
	mux.HandleFunc("/sse", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		// Send endpoint event.
		fmt.Fprintf(w, "event: endpoint\ndata: /messages?session=test\n\n")
		flusher.Flush()
		ctx := r.Context()
		for {
			select {
			case ev := <-f.stream:
				fmt.Fprint(w, ev)
				flusher.Flush()
			case <-ctx.Done():
				return
			}
		}
	})

	mux.HandleFunc("/messages", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      json.RawMessage `json:"id,omitempty"`
			Method  string          `json:"method"`
			Params  json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		if req.ID == nil {
			return // notification
		}
		// Build response.
		var result any
		switch req.Method {
		case "initialize":
			result = map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{}}
		case "tools/list":
			f.mu.Lock()
			result = map[string]any{"tools": f.tools}
			f.mu.Unlock()
		case "tools/call":
			f.gotCalls.Add(1)
			var p struct {
				Name string `json:"name"`
			}
			_ = json.Unmarshal(req.Params, &p)
			result = map[string]any{
				"isError": false,
				"content": []map[string]any{{"type": "text", "text": "called " + p.Name}},
			}
		default:
			http.Error(w, "unknown method", http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"jsonrpc": "2.0",
			"id":      json.RawMessage(req.ID),
			"result":  result,
		}
		body, _ := json.Marshal(resp)
		f.stream <- fmt.Sprintf("event: message\ndata: %s\n\n", string(body))
	})

	srv := httptest.NewServer(mux)
	f.closeAll = srv.Close
	return f, srv
}

// pushNotification queues an unsolicited notification for the next read.
func (f *fakeMCP) pushNotification(method string, params any) {
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
		"params":  params,
	})
	f.stream <- fmt.Sprintf("event: message\ndata: %s\n\n", string(body))
}

func TestClient_ListAndCall(t *testing.T) {
	f, srv := newFakeMCP(t, []Tool{
		{Name: "browse", Description: "browse a URL", InputSchema: json.RawMessage(`{"type":"object"}`)},
	})
	defer srv.Close()
	defer f.closeAll()

	c := New(srv.URL, nil, t.Logf)
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	tools, err := c.ListTools(ctx, time.Minute)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "browse" {
		t.Fatalf("unexpected tools: %+v", tools)
	}

	res, err := c.CallTool(ctx, "browse", json.RawMessage(`{"url":"https://example.com"}`), nil)
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if res.IsError || len(res.Content) == 0 || res.Content[0].Text != "called browse" {
		t.Fatalf("unexpected call result: %+v", res)
	}
	if got := f.gotCalls.Load(); got != 1 {
		t.Fatalf("expected 1 call, got %d", got)
	}
}

func TestClient_ToolsListChanged_InvalidatesCache(t *testing.T) {
	f, srv := newFakeMCP(t, []Tool{{Name: "v1"}})
	defer srv.Close()
	defer f.closeAll()

	c := New(srv.URL, nil, t.Logf)
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if _, err := c.ListTools(ctx, time.Minute); err != nil {
		t.Fatalf("first list: %v", err)
	}

	// Mutate server tool list and push notification.
	f.mu.Lock()
	f.tools = []Tool{{Name: "v1"}, {Name: "v2"}}
	f.mu.Unlock()
	f.pushNotification("notifications/tools/list_changed", map[string]any{})

	// Allow the SSE event to be processed.
	time.Sleep(150 * time.Millisecond)

	tools, err := c.ListTools(ctx, time.Minute)
	if err != nil {
		t.Fatalf("second list: %v", err)
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools after invalidation, got %d", len(tools))
	}
}
