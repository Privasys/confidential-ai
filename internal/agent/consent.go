package agent

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ConsentDecision is the answer the front-end sends back when the user
// either approves or denies a privileged tool invocation.
type ConsentDecision struct {
	Allowed bool   `json:"allowed"`
	Reason  string `json:"reason,omitempty"`
}

// ConsentRegistry keeps the set of in-flight consent requests so that
// the HTTP confirm endpoint (running in a different request goroutine)
// can deliver the user's decision back to the agent loop that is
// blocked waiting for it.
//
// Lifetime: an entry exists between Wait being called and either the
// decision arriving, the context being cancelled, or the per-entry
// timeout firing. Resolve is a no-op when no entry exists for the id.
type ConsentRegistry struct {
	mu      sync.Mutex
	pending map[string]chan ConsentDecision
}

// NewConsentRegistry returns an empty registry. Safe for concurrent use.
func NewConsentRegistry() *ConsentRegistry {
	return &ConsentRegistry{pending: make(map[string]chan ConsentDecision)}
}

// ErrConsentTimeout is returned by Wait when the per-call timeout
// elapses before a decision arrives.
var ErrConsentTimeout = errors.New("consent timed out")

// ErrConsentDenied is returned by Wait when the user explicitly denied
// the call.
var ErrConsentDenied = errors.New("user denied tool call")

// Wait blocks until Resolve(id, ...) is called, the context is done, or
// `timeout` elapses. The id is typically the tool_call id emitted on
// the SSE stream. If the same id is registered twice the second
// registration replaces the first (and the first waiter unblocks with
// ErrConsentTimeout via context cancellation by the caller).
func (r *ConsentRegistry) Wait(ctx context.Context, id string, timeout time.Duration) (ConsentDecision, error) {
	ch := make(chan ConsentDecision, 1)
	r.mu.Lock()
	r.pending[id] = ch
	r.mu.Unlock()
	defer func() {
		r.mu.Lock()
		// Only remove if still ours. A racing Resolve already drained
		// the channel; a second Wait with the same id would have
		// replaced our entry, so we leave the newer one in place.
		if cur, ok := r.pending[id]; ok && cur == ch {
			delete(r.pending, id)
		}
		r.mu.Unlock()
	}()

	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case d := <-ch:
		return d, nil
	case <-ctx.Done():
		return ConsentDecision{}, ctx.Err()
	case <-timer.C:
		return ConsentDecision{}, ErrConsentTimeout
	}
}

// Resolve delivers the user's decision to the goroutine that called
// Wait(id, ...). Returns true when a waiter was found, false when the
// id is unknown (already resolved, expired, or never registered).
func (r *ConsentRegistry) Resolve(id string, d ConsentDecision) bool {
	r.mu.Lock()
	ch, ok := r.pending[id]
	if ok {
		delete(r.pending, id)
	}
	r.mu.Unlock()
	if !ok {
		return false
	}
	select {
	case ch <- d:
		return true
	default:
		return false
	}
}

// Pending returns the number of currently registered waiters. Test-only.
func (r *ConsentRegistry) Pending() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.pending)
}
