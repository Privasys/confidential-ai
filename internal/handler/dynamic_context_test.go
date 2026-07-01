package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInjectDynamicContext_InsertsSystemBeforeLastUser(t *testing.T) {
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
	// One message inserted; original four preserved.
	if len(m.Messages) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(m.Messages))
	}
	// Static system prompt stays first (cacheable prefix), history untouched.
	if m.Messages[0].Content != "STATIC PROMPT" {
		t.Errorf("system prompt mutated: %q", m.Messages[0].Content)
	}
	if m.Messages[1].Content != "hello" || m.Messages[1].Role != "user" {
		t.Errorf("earlier user turn mutated: %+v", m.Messages[1])
	}
	// The injected context is a dedicated system message right before the
	// last user turn.
	if m.Messages[3].Role != "system" {
		t.Errorf("expected injected system message at index 3, got role %q", m.Messages[3].Role)
	}
	if !strings.Contains(m.Messages[3].Content, "2026-06-30T20:48:00Z") {
		t.Errorf("time not in injected system message: %q", m.Messages[3].Content)
	}
	// The last user message is unchanged and stays last.
	if m.Messages[4].Role != "user" || m.Messages[4].Content != "what time is it?" {
		t.Errorf("last user turn changed: %+v", m.Messages[4])
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
