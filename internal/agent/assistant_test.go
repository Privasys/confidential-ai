package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestDispatcher_AssistantAuth verifies the §8.7 RAG-in-enclave auth path:
// an assistant-mode server call carries `Authorization: Assistant <token>`
// and names the acting user via X-Privasys-On-Behalf-Of (from the context).
func TestDispatcher_AssistantAuth(t *testing.T) {
	var gotAuth, gotOBO string
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/mcp/tools", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"tools": []Tool{{Name: "search_semantic"}}})
	})
	mux.HandleFunc("/api/v1/mcp/tools/", func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotOBO = r.Header.Get("X-Privasys-On-Behalf-Of")
		w.WriteHeader(200)
		w.Write([]byte(`{"hits":[]}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	cat := NewCatalog([]Server{{
		Name:           "drive",
		BaseURL:        srv.URL,
		Transport:      TransportPrivasysHTTP,
		AuthMode:       AuthModeAssistant,
		AssistantToken: "shared-secret",
	}}, nil, time.Hour)
	if _, err := cat.Tools(context.Background()); err != nil {
		t.Fatal(err)
	}
	d := NewDispatcher(cat, nil)

	ctx := WithOnBehalfOf(context.Background(), "user-42")
	res := d.Call(ctx, "drive__search_semantic", json.RawMessage(`{"query":"x"}`), "")
	if res.Status != "ok" {
		t.Fatalf("status %q error %q", res.Status, res.Error)
	}
	if gotAuth != "Assistant shared-secret" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Assistant shared-secret")
	}
	if gotOBO != "user-42" {
		t.Fatalf("X-Privasys-On-Behalf-Of = %q, want %q", gotOBO, "user-42")
	}

	// Without an acting user in context, the header is omitted (no forged
	// empty assertion).
	ctxNone := context.Background()
	_ = d.Call(ctxNone, "drive__search_semantic", json.RawMessage(`{}`), "")
	if gotOBO != "" {
		t.Fatalf("on-behalf-of should be empty without a caller, got %q", gotOBO)
	}
}
