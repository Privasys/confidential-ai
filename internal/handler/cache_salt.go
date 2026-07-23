package handler

import (
	"crypto/hmac"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

// KV-cache scoping (vLLM automatic prefix caching).
//
// vLLM V1 keeps prefix caching enabled: identical token prefixes reuse
// their KV blocks instead of re-prefilling. Reuse is scoped per request
// with vLLM's `cache_salt` — the salt is folded into the KV-block hash,
// so only requests carrying the same salt can share cached blocks.
//
//   - "session" (default): the salt is an HMAC of the verified caller
//     subject under a per-process key. A caller's own requests (multi-turn
//     conversations, repeated system prompts) get full prefix reuse, while
//     callers are cryptographically isolated from each other — which also
//     closes the prefix-cache timing side channel (an attacker inferring
//     another tenant's prompt from cache-hit latency).
//   - "strict" (X-Privasys-Reproducibility: strict): a fresh single-use
//     salt per request — zero cache reuse by construction, the whole
//     prompt is freshly prefilled. This restores the serialized-replay
//     determinism recipe on a shared, cache-enabled engine without a
//     separate deployment. Note it does NOT remove concurrent-batching
//     numeric noise; kernel-level batch invariance (tracked upstream)
//     remains the path to bitwise serve==replay under load.
//
// A replay is cache-cold by construction on either mode (fresh salt), so
// the recorded seed + dynamic context reproduce the response exactly as
// the strict contract promises.
const (
	kvCacheModeSession = "session"
	kvCacheModeStrict  = "strict"
)

// strictSaltCounter disambiguates fallback strict salts if crypto/rand
// ever fails (effectively unreachable).
var strictSaltCounter atomic.Uint64

// wantsStrictReproducibility reports whether the caller asked for the
// strict KV-cache mode. Any other non-empty value of the header keeps the
// existing meaning (opt in to the reproducibility block, session cache).
func wantsStrictReproducibility(r *http.Request) bool {
	return strings.EqualFold(strings.TrimSpace(r.Header.Get("X-Privasys-Reproducibility")), "strict")
}

// cacheSalt derives the vLLM cache_salt for this request. Session salts are
// stable per caller for the life of this proxy (and therefore of the vLLM
// engine it supervises — the cache does not outlive either). Strict salts
// are single-use.
func (h *Handler) cacheSalt(r *http.Request, mode string) string {
	if mode == kvCacheModeStrict {
		var b [16]byte
		if _, err := crand.Read(b[:]); err != nil {
			// Never reuse a strict salt: fall back to a monotonic value
			// rather than a constant.
			return fmt.Sprintf("strict-%d-%d", time.Now().UnixNano(), strictSaltCounter.Add(1))
		}
		return "strict-" + hex.EncodeToString(b[:])
	}
	mac := hmac.New(sha256.New, h.saltKey)
	caller := callerFromContext(r.Context())
	if caller == "" {
		// Inference auth is mandatory, so this is only reachable on paths
		// that skipped authorizeInference; keep such traffic in its own
		// partition rather than sharing one with any real caller.
		caller = "anonymous"
	}
	mac.Write([]byte(caller))
	return hex.EncodeToString(mac.Sum(nil))
}

// injectCacheSalt sets the request's cache_salt, overwriting any
// client-supplied value: salts are derived server-side from the verified
// caller identity, never chosen by callers, so a client cannot join (or
// probe the existence of) another tenant's cache partition.
func injectCacheSalt(body []byte, salt string) ([]byte, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	m["cache_salt"] = salt
	return json.Marshal(m)
}

// newSaltKey returns the per-process HMAC key for session salts.
func newSaltKey() []byte {
	k := make([]byte, 32)
	if _, err := crand.Read(k); err != nil {
		// Effectively unreachable; a predictable key only weakens the
		// unguessability of salt VALUES, not the per-caller partitioning.
		copy(k, fmt.Sprintf("fallback-%d", time.Now().UnixNano()))
	}
	return k
}
