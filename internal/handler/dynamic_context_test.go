package handler

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestInjectDynamicContext_TailOfLastUserMessage(t *testing.T) {
	// Multi-turn conversation: the context is appended as a delimited block
	// at the END of the LAST user message, so the system prompt and the
	// whole history stay byte-stable (the cacheable prefix). Verified
	// empirically on Qwen3.6-35B (2026-07-23): tail-injected context is
	// answered correctly in thinking and non-thinking modes.
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
	if len(m.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(m.Messages))
	}
	// System prompt and history untouched — byte-stable prefix.
	if m.Messages[0].Role != "system" || m.Messages[0].Content != "STATIC PROMPT" {
		t.Errorf("system prompt must stay untouched: %+v", m.Messages[0])
	}
	if m.Messages[1].Content != "hello" || m.Messages[2].Content != "hi" {
		t.Errorf("history changed: %+v", m.Messages)
	}
	// Context appended to the final user turn, delimited.
	last := m.Messages[3]
	if last.Role != "user" ||
		!strings.HasPrefix(last.Content, "what time is it?") ||
		!strings.Contains(last.Content, "---\n(Context: The current date and time is 2026-06-30T20:48:00Z (UTC).)") {
		t.Errorf("context not tail-injected: %q", last.Content)
	}
}

func TestInjectDynamicContext_LastUserNotTrailing(t *testing.T) {
	// The last USER message wins even when the conversation ends on an
	// assistant/tool turn (agentic bodies mid-loop).
	body := []byte(`{"messages":[` +
		`{"role":"user","content":"question"},` +
		`{"role":"assistant","content":"calling tool"}` +
		`]}`)
	out, err := injectDynamicContext(body, "T-CTX")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var m struct {
		Messages []struct{ Role, Content string } `json:"messages"`
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !strings.Contains(m.Messages[0].Content, "T-CTX") {
		t.Errorf("context not injected into last user turn: %+v", m.Messages)
	}
	if strings.Contains(m.Messages[1].Content, "T-CTX") {
		t.Errorf("assistant turn must not carry the context: %+v", m.Messages[1])
	}
}

func TestInjectDynamicContext_NoUserFallsBackToSystem(t *testing.T) {
	// A system-only body merges into the one system message (there must be
	// exactly one system message, first).
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

func TestInjectDynamicContext_NoUserNoSystemPrepends(t *testing.T) {
	body := []byte(`{"messages":[{"role":"assistant","content":"yo"}]}`)
	out, err := injectDynamicContext(body, "T-CTX")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var m struct {
		Messages []struct{ Role, Content string } `json:"messages"`
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(m.Messages) != 2 || m.Messages[0].Role != "system" || m.Messages[0].Content != "T-CTX" {
		t.Fatalf("expected prepended system message, got %+v", m.Messages)
	}
}

func TestInjectDynamicContext_MultimodalUserContent(t *testing.T) {
	// Array-of-parts user content gets a text part appended.
	body := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"describe"}]}]}`)
	out, err := injectDynamicContext(body, "T-CTX")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !strings.Contains(string(out), "T-CTX") {
		t.Errorf("context missing from multimodal body: %s", out)
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
