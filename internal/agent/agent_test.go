package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakeMCP serves /api/v1/mcp/tools and /api/v1/mcp/tools/<name>.
type fakeMCP struct {
	tools  []Tool
	calls  atomic.Int32
	answer func(tool string, body []byte, bearer string) (int, []byte)
}

func (f *fakeMCP) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/mcp/tools", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"tools": f.tools})
	})
	mux.HandleFunc("/api/v1/mcp/tools/", func(w http.ResponseWriter, r *http.Request) {
		f.calls.Add(1)
		name := strings.TrimPrefix(r.URL.Path, "/api/v1/mcp/tools/")
		body := make([]byte, r.ContentLength)
		r.Body.Read(body)
		bearer := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		status, resp := f.answer(name, body, bearer)
		w.WriteHeader(status)
		w.Write(resp)
	})
	return mux
}

func TestCatalog_FetchAndCache(t *testing.T) {
	f := &fakeMCP{tools: []Tool{
		{Name: "search", Description: "search", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}}
	srv := httptest.NewServer(f.handler())
	defer srv.Close()

	cat := NewCatalog([]Server{{Name: "rag", BaseURL: srv.URL}}, nil, time.Hour)
	tools, err := cat.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Server != "rag" || tools[0].Name != "search" {
		t.Fatalf("unexpected tools: %+v", tools)
	}
	if got := tools[0].QualifiedName(); got != "rag__search" {
		t.Fatalf("QualifiedName = %q", got)
	}
	// Cache hit on second call: shut down server, second fetch should still work.
	srv.Close()
	tools2, err := cat.Tools(context.Background())
	if err != nil || len(tools2) != 1 {
		t.Fatalf("cache miss: %v %+v", err, tools2)
	}
}

func TestSplitQualifiedName(t *testing.T) {
	cases := map[string]struct {
		in        string
		srv, tool string
		ok        bool
	}{
		"plain":   {"rag__search", "rag", "search", true},
		"missing": {"search", "", "", false},
		"empty_t": {"rag__", "rag", "", true},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			s, tool, ok := SplitQualifiedName(tc.in)
			if s != tc.srv || tool != tc.tool || ok != tc.ok {
				t.Fatalf("got (%q,%q,%v)", s, tool, ok)
			}
		})
	}
}

func TestDispatcher_OK(t *testing.T) {
	f := &fakeMCP{
		tools: []Tool{{Name: "search"}},
		answer: func(tool string, body []byte, bearer string) (int, []byte) {
			if tool != "search" {
				return 404, []byte(`{"error":"unknown"}`)
			}
			if bearer != "user-jwt" {
				return 401, []byte(`{"error":"no bearer"}`)
			}
			return 200, []byte(`{"hits":[{"id":"c1"}]}`)
		},
	}
	srv := httptest.NewServer(f.handler())
	defer srv.Close()

	cat := NewCatalog([]Server{{Name: "rag", BaseURL: srv.URL, BearerForward: true}}, nil, time.Hour)
	if _, err := cat.Tools(context.Background()); err != nil {
		t.Fatal(err)
	}
	d := NewDispatcher(cat, nil)

	res := d.Call(context.Background(), "rag__search", json.RawMessage(`{"query":"x"}`), "user-jwt")
	if res.Status != "ok" {
		t.Fatalf("status %q error %q", res.Status, res.Error)
	}
	if !strings.Contains(string(res.Result), "c1") {
		t.Fatalf("result: %s", res.Result)
	}
}

func TestDispatcher_ErrorPaths(t *testing.T) {
	f := &fakeMCP{
		tools:  []Tool{{Name: "search"}},
		answer: func(string, []byte, string) (int, []byte) { return 500, []byte(`{"error":"boom"}`) },
	}
	srv := httptest.NewServer(f.handler())
	defer srv.Close()
	cat := NewCatalog([]Server{{Name: "rag", BaseURL: srv.URL}}, nil, time.Hour)
	cat.Tools(context.Background())
	d := NewDispatcher(cat, nil)

	cases := map[string]struct {
		name   string
		expect string
	}{
		"unknown_server": {"other__search", "unknown MCP server"},
		"malformed":      {"plain", "malformed tool name"},
		"upstream_500":   {"rag__search", "status 500"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			r := d.Call(context.Background(), tc.name, json.RawMessage(`{}`), "")
			if r.Status != "error" || !strings.Contains(r.Error, tc.expect) {
				t.Fatalf("got status=%q err=%q", r.Status, r.Error)
			}
		})
	}
}

func TestInjectTools(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)
	tools := []Tool{
		{Server: "rag", Name: "search", Description: "Search", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)},
	}
	out, err := InjectTools(body, tools)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	tarr, _ := got["tools"].([]any)
	if len(tarr) != 1 {
		t.Fatalf("tools: %+v", tarr)
	}
	tc, _ := got["tool_choice"].(string)
	if tc != "auto" {
		t.Fatalf("tool_choice: %q", tc)
	}
	fn, _ := tarr[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "rag__search" {
		t.Fatalf("function.name: %v", fn["name"])
	}
}

func TestRun_NoToolCalls_Passthrough(t *testing.T) {
	cat := NewCatalog(nil, nil, time.Hour)
	d := NewDispatcher(cat, nil)
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`)
	final, results, err := Run(context.Background(), d, body, LoopOptions{
		Invoke: func(ctx context.Context, b []byte) ([]byte, error) {
			// The loop is mode-agnostic: it must NOT mutate stream /
			// stream_options. Verify the body reaches Invoke verbatim.
			var req map[string]any
			json.Unmarshal(b, &req)
			if req["stream"] != true {
				t.Fatalf("stream should be passed through as true, got %v", req["stream"])
			}
			if _, ok := req["stream_options"]; !ok {
				t.Fatalf("stream_options should be passed through, got nothing")
			}
			return []byte(`{"choices":[{"message":{"role":"assistant","content":"hello"}}]}`), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Fatalf("results: %+v", results)
	}
	if !strings.Contains(string(final), "hello") {
		t.Fatalf("final: %s", final)
	}
}

func TestRun_ToolCallLoop(t *testing.T) {
	f := &fakeMCP{
		tools: []Tool{{Name: "search"}},
		answer: func(_ string, body []byte, _ string) (int, []byte) {
			// vLLM emits arguments as a JSON-string-wrapped JSON object.
			// The dispatcher MUST unwrap it and post the raw object.
			var obj map[string]any
			if err := json.Unmarshal(body, &obj); err != nil {
				t.Fatalf("dispatcher posted non-object body %q: %v", body, err)
			}
			if obj["query"] != "q" {
				t.Fatalf("dispatcher dropped args, got %v", obj)
			}
			return 200, []byte(`{"hits":[{"id":"c1","text":"the answer is 42"}]}`)
		},
	}
	srv := httptest.NewServer(f.handler())
	defer srv.Close()
	cat := NewCatalog([]Server{{Name: "rag", BaseURL: srv.URL}}, nil, time.Hour)
	cat.Tools(context.Background())
	d := NewDispatcher(cat, nil)

	calls := 0
	events := []string{}
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"q"}]}`)
	final, results, err := Run(context.Background(), d, body, LoopOptions{
		EmitEvent: func(name string, _ any) { events = append(events, name) },
		Invoke: func(ctx context.Context, b []byte) ([]byte, error) {
			calls++
			if calls == 1 {
				// First call: model emits a tool call.
				return []byte(`{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"rag__search","arguments":"{\"query\":\"q\"}"}}]}}]}`), nil
			}
			// Second call must include the tool result message.
			var req map[string]any
			json.Unmarshal(b, &req)
			msgs := req["messages"].([]any)
			if len(msgs) < 3 {
				t.Fatalf("expected >=3 messages on second call, got %d", len(msgs))
			}
			last := msgs[len(msgs)-1].(map[string]any)
			if last["role"] != "tool" || last["tool_call_id"] != "t1" {
				t.Fatalf("expected tool message, got %+v", last)
			}
			return []byte(`{"choices":[{"message":{"role":"assistant","content":"42"}}]}`), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 vllm calls, got %d", calls)
	}
	if len(results) != 1 || results[0].Status != "ok" {
		t.Fatalf("results: %+v", results)
	}
	if !strings.Contains(string(final), "42") {
		t.Fatalf("final: %s", final)
	}
	if len(events) != 2 || events[0] != "tool_call" || events[1] != "tool_result" {
		t.Fatalf("events: %v", events)
	}
}

func TestRun_BoundedIterations(t *testing.T) {
	f := &fakeMCP{
		tools:  []Tool{{Name: "loop"}},
		answer: func(string, []byte, string) (int, []byte) { return 200, []byte(`{"x":1}`) },
	}
	srv := httptest.NewServer(f.handler())
	defer srv.Close()
	cat := NewCatalog([]Server{{Name: "x", BaseURL: srv.URL}}, nil, time.Hour)
	cat.Tools(context.Background())
	d := NewDispatcher(cat, nil)

	calls := 0
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"q"}]}`)
	_, results, err := Run(context.Background(), d, body, LoopOptions{
		MaxIterations: 3,
		Invoke: func(ctx context.Context, b []byte) ([]byte, error) {
			calls++
			// Always emit a tool call so we'd loop forever without the cap.
			// Last call (after delete tools) must NOT have tools.
			if calls == 4 {
				var req map[string]any
				json.Unmarshal(b, &req)
				if _, has := req["tools"]; has {
					t.Fatalf("final call must strip tools")
				}
				return []byte(`{"choices":[{"message":{"role":"assistant","content":"giving up"}}]}`), nil
			}
			return []byte(`{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"t","type":"function","function":{"name":"x__loop","arguments":"{}"}}]}}]}`), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 4 { // 3 iterations of tool calls + 1 final no-tool
		t.Fatalf("expected 4 calls, got %d", calls)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 tool results, got %d", len(results))
	}
}

func TestRun_WaitConsentDenied(t *testing.T) {
f := &fakeMCP{
tools: []Tool{{Name: "send_email", RequiresUserConfirmation: true}},
answer: func(string, []byte, string) (int, []byte) {
t.Fatalf("dispatcher should NOT be called when consent is denied")
return 200, nil
},
}
srv := httptest.NewServer(f.handler())
defer srv.Close()
cat := NewCatalog([]Server{{Name: "mail", BaseURL: srv.URL}}, nil, time.Hour)
cat.Tools(context.Background())
d := NewDispatcher(cat, nil)

events := []string{}
calls := 0
body := []byte(`{"model":"m","messages":[{"role":"user","content":"send mail"}]}`)
_, results, err := Run(context.Background(), d, body, LoopOptions{
EmitEvent: func(name string, _ any) { events = append(events, name) },
WaitConsent: func(ctx context.Context, callID, name string, args []byte) (bool, error) {
if callID != "t1" || name != "mail__send_email" {
t.Errorf("unexpected consent: id=%s name=%s", callID, name)
}
return false, nil
},
Invoke: func(ctx context.Context, b []byte) ([]byte, error) {
calls++
if calls == 1 {
return []byte(`{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"mail__send_email","arguments":"{\"to\":\"x@y\"}"}}]}}]}`), nil
}
// Second call MUST contain a tool message with the denial error.
var req map[string]any
json.Unmarshal(b, &req)
msgs := req["messages"].([]any)
last := msgs[len(msgs)-1].(map[string]any)
if last["role"] != "tool" {
t.Fatalf("expected tool message, got %+v", last)
}
content, _ := last["content"].(string)
if !strings.Contains(content, "user_denied") {
t.Fatalf("expected user_denied in tool content, got %q", content)
}
return []byte(`{"choices":[{"message":{"role":"assistant","content":"OK, I won't."}}]}`), nil
},
})
if err != nil {
t.Fatal(err)
}
if calls != 2 {
t.Fatalf("expected 2 vllm calls, got %d", calls)
}
if len(results) != 1 || results[0].Status != "error" || results[0].Error != "user_denied" {
t.Fatalf("results: %+v", results)
}
// Expect tool_call -> tool_confirm_request -> tool_result
want := []string{"tool_call", "tool_confirm_request", "tool_result"}
if len(events) != len(want) {
t.Fatalf("events: %v", events)
}
for i, w := range want {
if events[i] != w {
t.Fatalf("event[%d] = %s, want %s", i, events[i], w)
}
}
}
