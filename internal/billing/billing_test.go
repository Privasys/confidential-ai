package billing

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestNewDisabled(t *testing.T) {
	if r := New(Config{}); r != nil {
		t.Fatal("New with empty config should return nil (disabled)")
	}
	if r := New(Config{AccountID: "acc"}); r != nil {
		t.Fatal("New without ReportURL should return nil (disabled)")
	}
	// A nil reporter is safe to use.
	var r *Reporter
	if r.Frozen() {
		t.Fatal("nil reporter must not be frozen")
	}
	r.Record("req", "", "", 1, 1) // must not panic
	r.Start(context.Background())
}

func TestReporterReportsAndTracksFrozen(t *testing.T) {
	var mu sync.Mutex
	var got reportRequest
	gotCh := make(chan struct{}, 8)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer tok" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		var req reportRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		if len(req.Requests) > 0 {
			got.AccountID = req.AccountID
			got.Model = req.Model
			got.Requests = append(got.Requests, req.Requests...)
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(reportResponse{Frozen: true, Metered: len(req.Requests)})
		select {
		case gotCh <- struct{}{}:
		default:
		}
	}))
	defer srv.Close()

	r := New(Config{
		AccountID:   "acc-1",
		ReportURL:   srv.URL,
		ReportToken: "tok",
		Model:       "qwen36-35b-a3b-fp8",
	})
	if r == nil {
		t.Fatal("New should return a reporter when configured")
	}
	// Speed up the loop for the test.
	r.flushEvery = 10 * time.Millisecond
	r.probeEvery = 10 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r.Start(ctx)

	r.Record("req-1", "alice", "", 100, 50)
	// A pooling instance's slug rides per-line; the default model is elided.
	r.Record("req-2", "alice", "qwen3-embedding-06b", 40, 0)
	r.Record("req-3", "alice", "qwen36-35b-a3b-fp8", 10, 5) // == default → elided

	// Wait for all three lines to land (they may split across flushes).
	deadlineLines := time.Now().Add(2 * time.Second)
	for {
		select {
		case <-gotCh:
		case <-time.After(2 * time.Second):
			t.Fatal("timed out waiting for usage report")
		}
		mu.Lock()
		n := len(got.Requests)
		mu.Unlock()
		if n >= 3 {
			break
		}
		if time.Now().After(deadlineLines) {
			t.Fatalf("only %d usage lines arrived", n)
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if got.AccountID != "acc-1" || got.Model != "qwen36-35b-a3b-fp8" {
		t.Fatalf("unexpected report header: %+v", got)
	}
	if len(got.Requests) != 3 || got.Requests[0].RequestID != "req-1" ||
		got.Requests[0].InputTokens != 100 || got.Requests[0].OutputTokens != 50 {
		t.Fatalf("unexpected report lines: %+v", got.Requests)
	}
	if got.Requests[0].Model != "" || got.Requests[2].Model != "" {
		t.Fatalf("default-model lines must elide the per-line model: %+v", got.Requests)
	}
	if got.Requests[1].Model != "qwen3-embedding-06b" {
		t.Fatalf("pooling line must carry its own model slug: %+v", got.Requests[1])
	}

	// Wait until the cached freeze flag reflects the server's response.
	deadline := time.Now().Add(2 * time.Second)
	for !r.Frozen() {
		if time.Now().After(deadline) {
			t.Fatal("reporter never observed frozen=true")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestRecordIgnoresZeroTokens(t *testing.T) {
	r := New(Config{AccountID: "a", ReportURL: "http://x"})
	r.Record("req", "", "", 0, 0) // dropped, no enqueue
	select {
	case <-r.queue:
		t.Fatal("zero-token sample should not be enqueued")
	default:
	}
}
