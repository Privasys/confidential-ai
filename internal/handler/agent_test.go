package handler

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/privasys/confidential-ai/internal/config"
)

// TestAgenticChatCompletions exercises the full agentic streaming path:
// catalogue fetch -> InjectTools -> tool_call event -> dispatcher -> tool
// message in next vllm call -> final delta -> tool_call summary in the
// reproducibility block -> [DONE].
func TestAgenticChatCompletions(t *testing.T) {
	// Fake MCP server with one tool.
	mcp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/mcp/tools":
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"tools":[{"name":"search","description":"S","input_schema":{"type":"object"}}]}`))
		case "/api/v1/mcp/tools/search":
			if got := r.Header.Get("Authorization"); got != "Bearer user-jwt" {
				t.Errorf("expected forwarded bearer, got %q", got)
			}
			w.Write([]byte(`{"hits":[{"id":"c1","text":"42"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer mcp.Close()

	// Fake vLLM: first call -> SSE chunk with a tool_call; second ->
	// SSE chunk with the final assistant content. Both end with the
	// standard `data: [DONE]` terminator vLLM uses in stream mode.
	vllmCalls := 0
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			return
		}
		vllmCalls++
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["stream"] != true {
			t.Errorf("vllm call %d: expected stream=true, got %v", vllmCalls, req["stream"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		writeChunk := func(payload string) {
			w.Write([]byte("data: " + payload + "\n\n"))
			if flusher != nil {
				flusher.Flush()
			}
		}
		if vllmCalls == 1 {
			// First call must include `tools` array we injected.
			if _, ok := req["tools"]; !ok {
				t.Errorf("first vllm call missing tools array")
			}
			// Tool-call as a single delta with the full args (vLLM
			// emits these in fragments in real life — the accumulator
			// handles either case).
			writeChunk(`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"rag__search","arguments":"{\"query\":\"q\"}"}}]},"finish_reason":"tool_calls"}]}`)
			writeChunk("[DONE]")
			return
		}
		// Second call must include the tool message at position N-1.
		msgs, _ := req["messages"].([]any)
		last := msgs[len(msgs)-1].(map[string]any)
		if last["role"] != "tool" || last["tool_call_id"] != "t1" {
			t.Errorf("second call: expected last message to be tool, got %+v", last)
		}
		writeChunk(`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"the answer is 42"},"finish_reason":"stop"}]}`)
		writeChunk("[DONE]")
	}))
	defer vllm.Close()

	h := New(&config.Config{
		ModelName:    "m",
		VLLMUpstream: vllm.URL,
		MCPServers:   "rag=" + mcp.URL + "?bearer=1",
		TeeType:      "tdx",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := `{"model":"m","stream":true,"messages":[{"role":"user","content":"what is the answer"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer user-jwt")
	req.Header.Set("X-Privasys-Reproducibility", "1")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status %d body %s", rec.Code, rec.Body.String())
	}
	if vllmCalls != 2 {
		t.Fatalf("expected 2 vllm calls, got %d", vllmCalls)
	}

	// Parse SSE: collect event names and the `data` payloads.
	type ev struct{ name, data string }
	var events []ev
	sc := bufio.NewScanner(strings.NewReader(rec.Body.String()))
	var curName string
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			curName = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			events = append(events, ev{name: curName, data: strings.TrimPrefix(line, "data: ")})
			curName = "" // event: applies to next data:
		}
	}

	// Expectations:
	//  1. event: tool_call with name=rag__search
	//  2. event: tool_result with status=ok
	//  3. data: final delta containing "the answer is 42"
	//  4. data: reproducibility block including tool_calls
	//  5. data: [DONE]
	if len(events) < 5 {
		t.Fatalf("expected >=5 events, got %d: %+v", len(events), events)
	}
	if events[0].name != "tool_call" || !strings.Contains(events[0].data, "rag__search") {
		t.Fatalf("event[0]: %+v", events[0])
	}
	if events[1].name != "tool_result" || !strings.Contains(events[1].data, `"status":"ok"`) {
		t.Fatalf("event[1]: %+v", events[1])
	}
	// Find the final-content delta and the reproducibility block.
	var sawFinal, sawTools, sawDone bool
	for _, e := range events[2:] {
		if strings.Contains(e.data, "the answer is 42") {
			sawFinal = true
		}
		if strings.Contains(e.data, `"tool_calls"`) && strings.Contains(e.data, "rag__search") {
			sawTools = true
		}
		if e.data == "[DONE]" {
			sawDone = true
		}
	}
	if !sawFinal {
		t.Fatalf("no final assistant delta in stream: %+v", events)
	}
	if !sawTools {
		t.Fatalf("reproducibility block missing tool_calls: %+v", events)
	}
	if !sawDone {
		t.Fatalf("missing [DONE] sentinel")
	}
}
