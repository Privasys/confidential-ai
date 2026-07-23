package handler

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/privasys/confidential-ai/internal/config"
)

func saltTestHandler() *Handler {
	return &Handler{cfg: &config.Config{}, saltKey: newSaltKey()}
}

func TestCacheSaltSessionStablePerCaller(t *testing.T) {
	h := saltTestHandler()
	r1 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r1 = r1.WithContext(context.WithValue(r1.Context(), callerCtxKey{}, "alice"))
	r2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r2 = r2.WithContext(context.WithValue(r2.Context(), callerCtxKey{}, "alice"))
	r3 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r3 = r3.WithContext(context.WithValue(r3.Context(), callerCtxKey{}, "bob"))

	s1 := h.cacheSalt(r1, kvCacheModeSession)
	s2 := h.cacheSalt(r2, kvCacheModeSession)
	s3 := h.cacheSalt(r3, kvCacheModeSession)
	if s1 == "" || s1 != s2 {
		t.Fatalf("same caller must get a stable session salt: %q vs %q", s1, s2)
	}
	if s1 == s3 {
		t.Fatal("different callers must get different session salts")
	}
}

func TestCacheSaltSessionDiffersAcrossProcesses(t *testing.T) {
	// Two handlers = two processes: same caller must land in different
	// partitions (per-process HMAC key).
	h1, h2 := saltTestHandler(), saltTestHandler()
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r = r.WithContext(context.WithValue(r.Context(), callerCtxKey{}, "alice"))
	if h1.cacheSalt(r, kvCacheModeSession) == h2.cacheSalt(r, kvCacheModeSession) {
		t.Fatal("session salts must be keyed per process")
	}
}

func TestCacheSaltStrictSingleUse(t *testing.T) {
	h := saltTestHandler()
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r = r.WithContext(context.WithValue(r.Context(), callerCtxKey{}, "alice"))
	s1 := h.cacheSalt(r, kvCacheModeStrict)
	s2 := h.cacheSalt(r, kvCacheModeStrict)
	if s1 == s2 {
		t.Fatal("strict salts must be single-use")
	}
	if !strings.HasPrefix(s1, "strict-") {
		t.Fatalf("strict salt should be marked: %q", s1)
	}
}

func TestWantsStrictReproducibility(t *testing.T) {
	cases := map[string]bool{
		"strict": true, "STRICT": true, " strict ": true,
		"1": false, "true": false, "": false, "0": false,
	}
	for v, want := range cases {
		r := httptest.NewRequest("POST", "/", nil)
		if v != "" {
			r.Header.Set("X-Privasys-Reproducibility", v)
		}
		if got := wantsStrictReproducibility(r); got != want {
			t.Errorf("header %q: got %v want %v", v, got, want)
		}
	}
	// strict also opts in to the reproducibility block
	r := httptest.NewRequest("POST", "/", nil)
	r.Header.Set("X-Privasys-Reproducibility", "strict")
	if !wantsReproducibility(r) {
		t.Fatal("strict must imply the reproducibility opt-in")
	}
}

func TestInjectCacheSaltOverridesClientValue(t *testing.T) {
	body := []byte(`{"model":"m","cache_salt":"attacker-chosen","messages":[]}`)
	out, err := injectCacheSalt(body, "server-derived")
	if err != nil {
		t.Fatalf("injectCacheSalt: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if m["cache_salt"] != "server-derived" {
		t.Fatalf("client-supplied cache_salt must be overridden, got %v", m["cache_salt"])
	}
}

func TestExtractCachedTokens(t *testing.T) {
	body := []byte(`{"id":"c1","usage":{"prompt_tokens":100,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":96}}}`)
	c := extractCachedTokens(body)
	if c == nil || *c != 96 {
		t.Fatalf("cached_tokens: got %v want 96", c)
	}
	if extractCachedTokens([]byte(`{"usage":{"prompt_tokens":1,"completion_tokens":1}}`)) != nil {
		t.Fatal("absent details must yield nil")
	}
	if extractCachedTokens([]byte(`{"usage":{"prompt_tokens":1,"prompt_tokens_details":null}}`)) != nil {
		t.Fatal("null details must yield nil")
	}
	ev := []byte("data: {\"id\":\"c2\",\"choices\":[],\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":32}}}\n\n")
	c = extractStreamCachedTokens(ev)
	if c == nil || *c != 32 {
		t.Fatalf("stream cached_tokens: got %v want 32", c)
	}
}
