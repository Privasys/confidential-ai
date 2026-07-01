package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInjectDynamicContext_InsertsAfterLeadingSystem(t *testing.T) {
	// Multi-turn conversation with a leading system prompt: the injected
	// context must land right after it (contiguous system run), NOT mid-
	// conversation — vLLM rejects a system message after a non-system turn.
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"system","content":"STATIC PROMPT"},` +
		`{"role":"user","content":"hello"},` +
		`{"role":"assistant","content":"hi"},` +
		`{"role":"user","content":"what time is it?"}` +
		`]}`)
	out, err := injectDynamicContext(body, "The current date and time is 2026-06-30T20:48:00Z (UTC).")
	if err != nil {
		t.Fatalf("injectDynamicContext: %v", err)
	}
	var m struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(m.Messages) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(m.Messages))
	}
	// Static system prompt stays message 0 (cacheable prefix).
	if m.Messages[0].Role != "system" || m.Messages[0].Content != "STATIC PROMPT" {
		t.Errorf("system prompt mutated: %+v", m.Messages[0])
	}
	// The injected context is a system message at index 1 (end of the leading
	// system run), so ALL system messages precede the first user turn.
	if m.Messages[1].Role != "system" || !strings.Contains(m.Messages[1].Content, "2026-06-30T20:48:00Z") {
		t.Errorf("expected injected system at index 1, got %+v", m.Messages[1])
	}
	// No system message may appear after a non-system message (vLLM constraint).
	seenNonSystem := false
	for i, mm := range m.Messages {
		if mm.Role != "system" {
			seenNonSystem = true
		} else if seenNonSystem {
			t.Errorf("system message at index %d follows a non-system message", i)
		}
	}
	// History + last user turn preserved and in order.
	if m.Messages[2].Content != "hello" || m.Messages[3].Content != "hi" || m.Messages[4].Content != "what time is it?" {
		t.Errorf("conversation reordered: %+v", m.Messages)
	}
}

func TestInjectDynamicContext_NoLeadingSystem(t *testing.T) {
	// No system prompt: the context becomes message 0.
	body := []byte(`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},{"role":"user","content":"time?"}]}`)
	out, err := injectDynamicContext(body, "TIME-CTX")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var m struct {
		Messages []struct{ Role, Content string } `json:"messages"`
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(m.Messages) != 4 || m.Messages[0].Role != "system" || m.Messages[0].Content != "TIME-CTX" {
		t.Fatalf("expected injected system at index 0, got %+v", m.Messages)
	}
	if m.Messages[1].Role != "user" || m.Messages[1].Content != "hi" {
		t.Errorf("first user turn changed: %+v", m.Messages[1])
	}
}

func TestInjectDynamicContext_NoUserMessage(t *testing.T) {
	body := []byte(`{"messages":[{"role":"system","content":"x"}]}`)
	out, err := injectDynamicContext(body, "T")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(out) != string(body) {
		t.Errorf("body changed when no user message present")
	}
}

func TestInjectDynamicContext_EmptyContextNoop(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	out, _ := injectDynamicContext(body, "")
	if string(out) != string(body) {
		t.Errorf("empty context should be a no-op")
	}
}

func TestDynamicContext_HeaderOverrideForReplay(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-Privasys-Dynamic-Context", "FIXED REPLAY VALUE")
	if got := dynamicContext(r); got != "FIXED REPLAY VALUE" {
		t.Errorf("replay override not honoured: %q", got)
	}
}

func TestDynamicContext_DefaultsToClock(t *testing.T) {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := dynamicContext(r)
	if !strings.HasPrefix(got, "The current date and time is ") {
		t.Errorf("expected a clock context, got %q", got)
	}
}
