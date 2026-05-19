package specsync

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/privasys/confidential-ai/internal/agent"
)

// fakeSpecServer serves a JSON spec body and counts hits. Callers can
// mutate the served spec while the server runs to exercise the hot-
// reload code path.
type fakeSpecServer struct {
	srv      *httptest.Server
	hits     atomic.Int64
	spec     atomic.Pointer[Response]
	wantAuth atomic.Pointer[string]
}

func newFakeSpec(t *testing.T) *fakeSpecServer {
	t.Helper()
	f := &fakeSpecServer{}
	mux := http.NewServeMux()
	mux.HandleFunc("/spec", func(w http.ResponseWriter, r *http.Request) {
		f.hits.Add(1)
		if want := f.wantAuth.Load(); want != nil {
			if got := r.Header.Get("Authorization"); got != *want {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
		}
		sp := f.spec.Load()
		if sp == nil {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		// Hand-roll JSON so we don't pull encoding/json into the
		// minimum dependency set of this fixture file.
		_, _ = w.Write([]byte(`{"spec":"` + sp.Spec + `","generation":"` + sp.Generation + `"}`))
	})
	f.srv = httptest.NewServer(mux)
	t.Cleanup(f.srv.Close)
	return f
}

func TestSyncerAppliesAndCachesByGeneration(t *testing.T) {
	f := newFakeSpec(t)
	f.spec.Store(&Response{Spec: "rag=https://rag.example.com?transport=privasys_http", Generation: "gen-1"})

	cat := agent.NewCatalog(nil, nil, time.Hour)
	s := New(f.srv.URL+"/spec", "", 10*time.Millisecond, nil, cat)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	// Wait for first apply.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(cat.Servers()) == 1 && s.LastGeneration() == "gen-1" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := len(cat.Servers()); got != 1 {
		t.Fatalf("expected 1 server after initial apply, got %d (lastGen=%q lastErr=%v)", got, s.LastGeneration(), s.LastError())
	}

	// Same generation must be a no-op even though the server is hit.
	// Mutate spec text but keep gen the same; catalog stays at 1.
	f.spec.Store(&Response{Spec: "rag=https://rag.example.com,extra=https://extra.example.com", Generation: "gen-1"})
	time.Sleep(60 * time.Millisecond)
	if got := len(cat.Servers()); got != 1 {
		t.Fatalf("expected catalogue unchanged on same generation, got %d servers", got)
	}

	// New generation triggers re-parse and replace.
	f.spec.Store(&Response{Spec: "rag=https://rag.example.com,extra=https://extra.example.com", Generation: "gen-2"})
	deadline = time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(cat.Servers()) == 2 && s.LastGeneration() == "gen-2" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := len(cat.Servers()); got != 2 {
		t.Fatalf("expected 2 servers after new generation, got %d", got)
	}

	// Empty spec is valid and clears the catalogue.
	f.spec.Store(&Response{Spec: "", Generation: "gen-3"})
	deadline = time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(cat.Servers()) == 0 && s.LastGeneration() == "gen-3" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := len(cat.Servers()); got != 0 {
		t.Fatalf("expected empty catalogue after empty spec, got %d servers", got)
	}
}

func TestSyncerForwardsBearerToken(t *testing.T) {
	f := newFakeSpec(t)
	want := "Bearer secret-xyz"
	f.wantAuth.Store(&want)
	f.spec.Store(&Response{Spec: "", Generation: "g"})

	cat := agent.NewCatalog(nil, nil, time.Hour)
	s := New(f.srv.URL+"/spec", "secret-xyz", 5*time.Millisecond, nil, cat)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	deadline := time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		if s.LastGeneration() == "g" {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("syncer did not authenticate (lastErr=%v)", s.LastError())
}

func TestSyncerRecordsHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("upstream down"))
	}))
	defer srv.Close()

	cat := agent.NewCatalog(nil, nil, time.Hour)
	s := New(srv.URL, "", 5*time.Millisecond, nil, cat)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	s.Run(ctx)

	if s.LastError() == nil {
		t.Fatal("expected LastError to be set after non-2xx response")
	}
	if s.LastGeneration() != "" {
		t.Fatalf("expected LastGeneration empty on error, got %q", s.LastGeneration())
	}
}
