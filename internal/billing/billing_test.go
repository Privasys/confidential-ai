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
	r.Record("req", 1, 1) // must not panic
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
			got = req
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

	r.Record("req-1", 100, 50)

	// Wait for the report to land.
	select {
	case <-gotCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for usage report")
	}

	mu.Lock()
	defer mu.Unlock()
	if got.AccountID != "acc-1" || got.Model != "qwen36-35b-a3b-fp8" {
		t.Fatalf("unexpected report header: %+v", got)
	}
	if len(got.Requests) != 1 || got.Requests[0].RequestID != "req-1" ||
		got.Requests[0].InputTokens != 100 || got.Requests[0].OutputTokens != 50 {
		t.Fatalf("unexpected report lines: %+v", got.Requests)
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
	r.Record("req", 0, 0) // dropped, no enqueue
	select {
	case <-r.queue:
		t.Fatal("zero-token sample should not be enqueued")
	default:
	}
}
