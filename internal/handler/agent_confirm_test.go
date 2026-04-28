package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/privasys/confidential-ai/internal/agent"
	"github.com/privasys/confidential-ai/internal/config"
)

// TestAgentConfirmEndpoint dials the bare /v1/agent/confirm/{id}
// route and checks that an in-flight Wait() unblocks with the
// decision, that unknown ids are 404, and that malformed bodies
// are 400.
func TestAgentConfirmEndpoint(t *testing.T) {
	h := New(&config.Config{}, nil)
	h.agentConsent = agent.NewConsentRegistry()
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// 404 when no waiter is registered.
	resp, err := http.Post(srv.URL+"/v1/agent/confirm/missing", "application/json", bytes.NewReader([]byte(`{"allowed":true}`)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 for unknown id, got %d", resp.StatusCode)
	}

	// Register a waiter and confirm via HTTP from a goroutine.
	type result struct {
		dec agent.ConsentDecision
		err error
	}
	done := make(chan result, 1)
	go func() {
		dec, err := h.agentConsent.Wait(context.Background(), "abc", time.Second)
		done <- result{dec, err}
	}()
	// Give the waiter a moment to register.
	time.Sleep(20 * time.Millisecond)

	resp, err = http.Post(srv.URL+"/v1/agent/confirm/abc", "application/json", bytes.NewReader([]byte(`{"allowed":true,"reason":"ok"}`)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("expected 204, got %d", resp.StatusCode)
	}

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("Wait err: %v", r.err)
		}
		if !r.dec.Allowed || r.dec.Reason != "ok" {
			t.Fatalf("decision wrong: %+v", r.dec)
		}
	case <-time.After(time.Second):
		t.Fatalf("Wait never returned")
	}

	// 400 on malformed JSON.
	resp, err = http.Post(srv.URL+"/v1/agent/confirm/abc", "application/json", bytes.NewReader([]byte(`{notjson`)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad body, got %d", resp.StatusCode)
	}

	// 404 when consent registry is nil (agentic loop disabled).
	hd := New(&config.Config{}, nil) // no MCP_SERVERS -> agentConsent stays nil
	mux2 := http.NewServeMux()
	hd.RegisterRoutes(mux2)
	srv2 := httptest.NewServer(mux2)
	defer srv2.Close()
	resp, err = http.Post(srv2.URL+"/v1/agent/confirm/anything", "application/json", bytes.NewReader([]byte(`{"allowed":true}`)))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 when agentic disabled, got %d", resp.StatusCode)
	}
	// Drain so json.* doesn't complain.
	_ = json.NewDecoder(resp.Body)
}
