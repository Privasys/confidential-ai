// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// toolServer serves a privasys_http MCP manifest + tool calls.
func toolServer(t *testing.T, tools []string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/mcp/tools", func(w http.ResponseWriter, _ *http.Request) {
		var list []map[string]any
		for _, n := range tools {
			list = append(list, map[string]any{"name": n, "description": n, "input_schema": map[string]any{"type": "object"}})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"tools": list})
	})
	mux.HandleFunc("/api/v1/mcp/tools/", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{"echo": r.URL.Path})
	})
	return httptest.NewServer(mux)
}

func TestDispatcherResolvesBareNames(t *testing.T) {
	srv := toolServer(t, []string{"kv-store", "kv-read", "hello"})
	defer srv.Close()

	servers := []Server{{Name: "kv_store", BaseURL: srv.URL, Transport: TransportPrivasysHTTP}}
	cat := NewCatalog(servers, srv.Client(), time.Minute)
	d := NewDispatcher(cat, srv.Client())
	ctx := context.Background()

	// Underscore/dash normalisation: kv_store means the kv-store tool
	// (unique fuzzy match), not the 3-tool server.
	res := d.Call(ctx, "kv_store", json.RawMessage(`{"key":"k","value":"v"}`), "")
	if res.Status != "ok" {
		t.Fatalf("kv_store call = %s (%s)", res.Status, res.Error)
	}

	// Exact bare tool name.
	res = d.Call(ctx, "hello", nil, "")
	if res.Status != "ok" {
		t.Fatalf("hello call = %s (%s)", res.Status, res.Error)
	}

	// Qualified names keep working.
	res = d.Call(ctx, "kv_store__kv-read", json.RawMessage(`{"key":"k"}`), "")
	if res.Status != "ok" {
		t.Fatalf("qualified call = %s (%s)", res.Status, res.Error)
	}

	// Genuinely unknown stays an error.
	res = d.Call(ctx, "nonexistent", nil, "")
	if res.Status != "error" {
		t.Fatalf("unknown name must fail, got %s", res.Status)
	}
}
