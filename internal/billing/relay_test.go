// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package billing

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// fakeEndpoint answers usage reports and billable-caller checks with a fixed
// status, counts the calls it got, and keeps the last report body and bearer.
type fakeEndpoint struct {
	status   int
	frozen   bool
	billable bool
	calls    atomic.Int32
	lastBody map[string]any
	lastAuth string
	srv      *httptest.Server
}

func newFake(t *testing.T, status int) *fakeEndpoint {
	t.Helper()
	f := &fakeEndpoint{status: status, frozen: true, billable: true}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls.Add(1)
		f.lastAuth = r.Header.Get("Authorization")
		if r.Method == http.MethodPost {
			f.lastBody = map[string]any{}
			_ = json.NewDecoder(r.Body).Decode(&f.lastBody)
		}
		w.WriteHeader(f.status)
		if f.status == http.StatusOK {
			if r.Method == http.MethodPost {
				_ = json.NewEncoder(w).Encode(map[string]any{"frozen": f.frozen})
			} else {
				_ = json.NewEncoder(w).Encode(map[string]any{"billable": f.billable})
			}
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func reporterFor(relay, legacy *fakeEndpoint) *Reporter {
	cfg := Config{Model: "m"}
	if relay != nil {
		cfg.RelayURL, cfg.RelayToken = relay.srv.URL+"/api/v1/containers/c", "container-token"
	}
	if legacy != nil {
		cfg.AccountID, cfg.ReportURL, cfg.ReportToken = "acct", legacy.srv.URL+"/enclave/ai-usage", "usage-token"
	}
	return New(cfg)
}

func sample() []usage {
	return []usage{{RequestID: "r1", InputTokens: 10, OutputTokens: 5}}
}

func TestRelayReportsWithoutAccountAndWithTheContainerToken(t *testing.T) {
	relay, legacy := newFake(t, http.StatusOK), newFake(t, http.StatusOK)
	r := reporterFor(relay, legacy)
	r.send(context.Background(), sample())

	if relay.calls.Load() != 1 || legacy.calls.Load() != 0 {
		t.Fatalf("relay calls %d, direct calls %d: want 1 and 0", relay.calls.Load(), legacy.calls.Load())
	}
	if _, has := relay.lastBody["account_id"]; has {
		t.Fatalf("the relayed body must not name an account: %v", relay.lastBody)
	}
	if relay.lastAuth != "Bearer container-token" {
		t.Fatalf("relay auth %q", relay.lastAuth)
	}
	if !r.Frozen() {
		t.Fatal("the freeze state must come from the relay's answer")
	}
}

func TestRelayOnlyConfigIsEnabled(t *testing.T) {
	relay := newFake(t, http.StatusOK)
	if r := reporterFor(relay, nil); r == nil {
		t.Fatal("a relay without a billing config must still meter")
	}
}

func TestRelayUnavailableFallsBackToTheDirectEndpoint(t *testing.T) {
	for _, status := range []int{http.StatusNotFound, http.StatusServiceUnavailable, http.StatusForbidden} {
		relay, legacy := newFake(t, status), newFake(t, http.StatusOK)
		r := reporterFor(relay, legacy)
		r.send(context.Background(), sample())
		r.send(context.Background(), sample())

		if relay.calls.Load() != 1 {
			t.Fatalf("status %d: relay asked %d times, want once (then the direct endpoint until the retry)", status, relay.calls.Load())
		}
		if legacy.calls.Load() != 2 {
			t.Fatalf("status %d: direct calls %d, want 2", status, legacy.calls.Load())
		}
		if legacy.lastBody["account_id"] != "acct" || legacy.lastAuth != "Bearer usage-token" {
			t.Fatalf("status %d: direct report body %v auth %q", status, legacy.lastBody, legacy.lastAuth)
		}
	}
}

func TestRelayUpstreamFailureIsTransient(t *testing.T) {
	relay, legacy := newFake(t, http.StatusBadGateway), newFake(t, http.StatusOK)
	r := reporterFor(relay, legacy)
	r.send(context.Background(), sample())
	r.send(context.Background(), sample())

	if legacy.calls.Load() != 0 {
		t.Fatal("a 502 from the relay is transient: no fallback to the direct endpoint")
	}
	if relay.calls.Load() != 2 {
		t.Fatalf("the relay must be tried again after a 502: %d calls", relay.calls.Load())
	}
}

func TestBillableCallerThroughTheRelayAndItsFallback(t *testing.T) {
	relay, legacy := newFake(t, http.StatusOK), newFake(t, http.StatusOK)
	relay.billable = false
	r := reporterFor(relay, legacy)
	if ok, err := r.CallerBillable(context.Background(), "sub-a"); err != nil || ok {
		t.Fatalf("relay verdict: ok=%v err=%v, want false from the relay", ok, err)
	}
	if relay.calls.Load() != 1 || legacy.calls.Load() != 0 {
		t.Fatalf("relay %d, direct %d", relay.calls.Load(), legacy.calls.Load())
	}

	relay2, legacy2 := newFake(t, http.StatusNotFound), newFake(t, http.StatusOK)
	r2 := reporterFor(relay2, legacy2)
	if ok, err := r2.CallerBillable(context.Background(), "sub-b"); err != nil || !ok {
		t.Fatalf("fallback verdict: ok=%v err=%v, want true from the direct endpoint", ok, err)
	}
	if legacy2.calls.Load() != 1 {
		t.Fatalf("the direct endpoint must answer when the relay is absent: %d calls", legacy2.calls.Load())
	}
}
