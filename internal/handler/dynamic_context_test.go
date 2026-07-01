package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInjectDynamicContext_MergesIntoLeadingSystem(t *testing.T) {
	// Multi-turn conversation with a leading system prompt: the context must be
	// APPENDED to it (one system message, first) — vLLM/Qwen reject both a
	// mid-conversation system message and a second system message.
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
	// No message inserted — merged into the existing system prompt.
	if len(m.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(m.Messages))
	}
	// Exactly one system message, first, carrying both the static prompt and
	// the injected time.
	if m.Messages[0].Role != "system" {
		t.Errorf("message 0 not system: %+v", m.Messages[0])
	}
	if !strings.HasPrefix(m.Messages[0].Content, "STATIC PROMPT") || !strings.Contains(m.Messages[0].Content, "2026-06-30T20:48:00Z") {
		t.Errorf("system prompt not merged: %q", m.Messages[0].Content)
	}
	for i, mm := range m.Messages[1:] {
		if mm.Role == "system" {
			t.Errorf("unexpected extra system message at index %d", i+1)
		}
	}
	// Conversation preserved.
	if m.Messages[1].Content != "hello" || m.Messages[2].Content != "hi" || m.Messages[3].Content != "what time is it?" {
		t.Errorf("conversation changed: %+v", m.Messages)
	}
}

func TestInjectDynamicContext_NoLeadingSystem(t *testing.T) {
	// No system prompt: the context becomes a new leading system message.
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

func TestInjectDynamicContext_SystemOnlyMerges(t *testing.T) {
	// A system-only body still merges into the one system message (harmless,
	// and keeps a single leading system message).
	body := []byte(`{"messages":[{"role":"system","content":"x"}]}`)
	out, err := injectDynamicContext(body, "T")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var m struct {
		Messages []struct{ Role, Content string } `json:"messages"`
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(m.Messages) != 1 || m.Messages[0].Role != "system" || !strings.Contains(m.Messages[0].Content, "T") {
		t.Errorf("expected merged single system message, got %+v", m.Messages)
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
