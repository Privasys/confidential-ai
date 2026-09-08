package billing

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestCallerBillable_VerdictCachedAndTokenSent(t *testing.T) {
	var asks atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asks.Add(1)
		if r.URL.Path != "/api/v1/enclave/billable-caller" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer tok" {
			t.Errorf("bearer = %q", r.Header.Get("Authorization"))
		}
		switch r.URL.Query().Get("sub") {
		case "has-account":
			w.Write([]byte(`{"billable":true}`))
		default:
			w.Write([]byte(`{"billable":false}`))
		}
	}))
	defer srv.Close()

	rep := New(Config{AccountID: "acc", ReportURL: srv.URL + "/api/v1/enclave/ai-usage", ReportToken: "tok"})
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		if ok, err := rep.CallerBillable(ctx, "has-account"); err != nil || !ok {
			t.Fatalf("has-account: ok=%v err=%v", ok, err)
		}
	}
	if ok, err := rep.CallerBillable(ctx, "nobody"); err != nil || ok {
		t.Fatalf("nobody must be unbillable: ok=%v err=%v", ok, err)
	}
	if got := asks.Load(); got != 2 {
		t.Fatalf("expected one ask per caller (cached), got %d", got)
	}

	// An expired negative verdict is re-asked.
	rep.billableMu.Lock()
	rep.billable["nobody"] = billableEntry{billable: false, until: time.Now().Add(-time.Second)}
	rep.billableMu.Unlock()
	rep.CallerBillable(ctx, "nobody")
	if got := asks.Load(); got != 3 {
		t.Fatalf("expired verdict must be re-asked, asks=%d", got)
	}
}

func TestCallerBillable_OutageUsesStaleElseUnknown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
	}))
	defer srv.Close()
	rep := New(Config{AccountID: "acc", ReportURL: srv.URL + "/api/v1/enclave/ai-usage"})
	ctx := context.Background()

	if _, err := rep.CallerBillable(ctx, "fresh"); err == nil {
		t.Fatal("no cache + outage must report unknown")
	}
	rep.billableMu.Lock()
	rep.billable["known"] = billableEntry{billable: true, until: time.Now().Add(-time.Minute)}
	rep.billableMu.Unlock()
	if ok, err := rep.CallerBillable(ctx, "known"); err != nil || !ok {
		t.Fatalf("stale verdict must be reused during an outage: ok=%v err=%v", ok, err)
	}

	// Metering off: nothing to check, everyone servable.
	var nilRep *Reporter
	if ok, err := nilRep.CallerBillable(ctx, "anyone"); err != nil || !ok {
		t.Fatalf("nil reporter: ok=%v err=%v", ok, err)
	}
}
