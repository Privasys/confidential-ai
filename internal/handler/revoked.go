package handler

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"sync"
	"time"
)

// revokedSet is the set of revoked session ids, refreshed by polling the IdP's
// /sessions/revoked feed. An inference token whose sid is revoked is rejected —
// this kills a leaked/rotated API key WITHOUT a per-request callout, so a prompt
// never leaves the TEE for an auth lookup (the confidential invariant).
//
// The set is additive within a process lifetime (a sid, once revoked, stays
// revoked even if it later ages out of the IdP's retention window), so a
// long-lived key cannot be un-revoked by retention expiry.
type revokedSet struct {
	url      string
	interval time.Duration
	client   *http.Client

	mu  sync.RWMutex
	set map[string]struct{}
}

func newRevokedSet(url string, interval time.Duration) *revokedSet {
	if interval <= 0 {
		interval = 60 * time.Second
	}
	return &revokedSet{
		url:      url,
		interval: interval,
		client:   &http.Client{Timeout: 10 * time.Second},
		set:      map[string]struct{}{},
	}
}

// Has reports whether sid has been revoked as of the last successful poll. A
// nil set (feature disabled) never revokes.
func (r *revokedSet) Has(sid string) bool {
	if r == nil || sid == "" {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.set[sid]
	return ok
}

// Start runs the background poll loop until ctx is cancelled. No-op when the
// feed URL is unset.
func (r *revokedSet) Start(ctx context.Context) {
	if r == nil || r.url == "" {
		return
	}
	go func() {
		t := time.NewTicker(r.interval)
		defer t.Stop()
		r.refresh(ctx)
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				r.refresh(ctx)
			}
		}
	}()
}

func (r *revokedSet) refresh(ctx context.Context) {
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, r.url, nil)
	if err != nil {
		return
	}
	resp, err := r.client.Do(req)
	if err != nil {
		log.Printf("[revoked] poll failed: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("[revoked] poll status %d", resp.StatusCode)
		return
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return
	}
	var doc struct {
		Revoked []string `json:"revoked"`
	}
	if json.Unmarshal(body, &doc) != nil {
		return
	}
	r.mu.Lock()
	for _, sid := range doc.Revoked {
		r.set[sid] = struct{}{}
	}
	r.mu.Unlock()
}
