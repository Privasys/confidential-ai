package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInjectDynamicContext_PrependsLastUserMessage(t *testing.T) {
	body := []byte(`{"model":"m","messages":[` +
		`{"role":"system","content":"STATIC PROMPT"},` +
		`{"role":"user","content":"hello"},` +
		`{"role":"assistant","content":"hi"},` +
		`{"role":"user","content":"what time is it?"}` +
		`]}`)
	out, err := injectDynamicContext(body, "Current date and time: 2026-06-30T20:48:00Z (UTC).")
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
	// Static system prompt must be untouched (cacheable prefix).
	if m.Messages[0].Content != "STATIC PROMPT" {
		t.Errorf("system prompt mutated: %q", m.Messages[0].Content)
	}
	// First user turn must be untouched — only the LAST user message gets it.
	if m.Messages[1].Content != "hello" {
		t.Errorf("earlier user turn mutated: %q", m.Messages[1].Content)
	}
	last := m.Messages[3].Content
	if !strings.Contains(last, "2026-06-30T20:48:00Z") {
		t.Errorf("context not injected into last user turn: %q", last)
	}
	if !strings.HasSuffix(last, "what time is it?") {
		t.Errorf("original user text not preserved: %q", last)
	}
	if !strings.HasPrefix(last, "<context>") {
		t.Errorf("context block not at the start: %q", last)
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
	if !strings.HasPrefix(got, "Current date and time:") {
		t.Errorf("expected a clock context, got %q", got)
	}
}
