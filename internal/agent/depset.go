package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	rc "enclave-os-mini/clients/go/ratls"
)

// DepSet is this workload's attested dependency set as the RUNTIME declares
// it — the same bytes the enclave manager stamps into our serving
// certificate at OID 1.3.6.1.4.1.65230.6.1.
//
// Sourcing the pins from here, rather than from the tool catalogue, is what
// makes "declared == enforced" true by construction: the platform decides
// the set (an operator signs it off, the manager stamps it, wallets consent
// to it), the app can only READ it, and every tool dial is verified against
// exactly what verifiers can see on our leaf. A catalogue that starts
// advertising a tool on an unpinned host is refused, which also bounds what
// a compromised control plane can redirect us to.
//
// Off platform (no PRIVASYS_MANAGER_URL / container token) the set is empty
// and Enabled() is false, leaving the legacy per-host digest pins in charge —
// the pre-6.1 behaviour, so dev and tests are unaffected.
type DepSet struct {
	mu      sync.RWMutex
	set     rc.DependencySet
	loaded  bool
	fetched time.Time

	managerURL string
	container  string
	token      string
	client     *http.Client
}

// NewDepSet builds the runtime dependency-set client from the container's
// environment. Returns a usable (disabled) value off platform.
func NewDepSet() *DepSet {
	return &DepSet{
		managerURL: strings.TrimRight(os.Getenv("PRIVASYS_MANAGER_URL"), "/"),
		container:  os.Getenv("PRIVASYS_CONTAINER_NAME"),
		token:      os.Getenv("PRIVASYS_CONTAINER_TOKEN"),
		client:     &http.Client{Timeout: 10 * time.Second},
	}
}

// Enabled reports whether a dependency set was successfully loaded and is
// non-empty. When false, callers keep their existing pinning behaviour.
func (d *DepSet) Enabled() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.loaded && len(d.set.Entries) > 0
}

// Set returns a copy of the current set.
func (d *DepSet) Set() rc.DependencySet {
	d.mu.RLock()
	defer d.mu.RUnlock()
	out := rc.DependencySet{Entries: make([]rc.DependencyEntry, len(d.set.Entries))}
	copy(out.Entries, d.set.Entries)
	return out
}

// Fold returns the identity fold of the current set: the value that
// distinguishes this tool surface from any other, surfaced in the
// reproducibility block so a response is attributable to the dependency set
// that served it even away from the certificate.
func (d *DepSet) Fold() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.loaded || len(d.set.Entries) == 0 {
		return ""
	}
	return rc.FoldIdentityHex(nil, nil, d.set)
}

// Refresh re-reads the set from the local enclave manager. Safe to call
// often; it is a loopback request. A failure leaves the previous set in
// place (a transient manager blip must not silently unpin our tools).
func (d *DepSet) Refresh() error {
	if d.managerURL == "" || d.container == "" || d.token == "" {
		return nil // off platform: stay disabled, no error
	}
	url := fmt.Sprintf("%s/api/v1/containers/%s/dependencies", d.managerURL, d.container)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+d.token)
	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("dependency set: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("dependency set: read: %w", err)
	}
	if resp.StatusCode == http.StatusNotFound {
		// Older runtime without the read API: stay disabled rather than
		// failing closed on every tool call.
		return nil
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("dependency set: manager returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var payload struct {
		Dependencies *rc.DependencySet `json:"dependencies"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return fmt.Errorf("dependency set: parse: %w", err)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.loaded = true
	d.fetched = time.Now()
	if payload.Dependencies == nil {
		d.set = rc.DependencySet{}
	} else {
		d.set = *payload.Dependencies
	}
	return nil
}

// Start loads the set once and refreshes it on an interval, so an
// operator-approved change (a re-mint on the running container) takes effect
// without a restart — the same latency the serving certificate has.
func (d *DepSet) Start(interval time.Duration) {
	if interval <= 0 {
		interval = time.Minute
	}
	if err := d.Refresh(); err != nil {
		log.Printf("[deps] initial dependency-set load failed (tool pins fall back to catalogue digests): %v", err)
	} else if d.Enabled() {
		log.Printf("[deps] runtime dependency set loaded: %d entries (fold=%s)", len(d.Set().Entries), d.Fold())
	}
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for range t.C {
			if err := d.Refresh(); err != nil {
				log.Printf("[deps] dependency-set refresh failed (keeping the previous set): %v", err)
			}
		}
	}()
}

// VerifyPeer enforces the declared set against an attested peer.
//
// Two legitimate pin sources exist and this is the single gate for both:
//   - PLATFORM tools (the fleet's defaults) are declared in the 6.1 set and
//     must match their entry: measured identity any-of + required OIDs.
//   - USER tools arrive through a signed per-request grant and are
//     deliberately NOT in the set (the set is per-app and durable; user
//     tools are per-user and per-session). They are pinned by the grant's
//     expected digest, which the caller enforces separately — so pass them
//     through here, signalled by grantPinned.
//
// A peer that is NEITHER declared NOR grant-pinned is refused. That is the
// point of the sign-off flow: a tool the catalogue starts advertising, on a
// host nobody approved, cannot receive tool data.
//
// Returns nil when no set is loaded (off platform / older runtime), leaving
// the legacy digest pinning in charge.
func (d *DepSet) VerifyPeer(peer rc.CertInfo, tee rc.TeeType, grantPinned bool) error {
	if !d.Enabled() {
		return nil
	}
	appID := rc.AppIDFromCert(peer)
	set := d.Set()
	for _, e := range set.Entries {
		if e.AppID == appID && appID != "" {
			return rc.MatchDependency(peer, tee, e)
		}
	}
	if grantPinned {
		return nil // user tool: the grant's expected digest is its pin
	}
	return fmt.Errorf("peer app-id %s is not a declared dependency of this enclave and carries no tool grant (fail closed)", orUnknownAppID(appID))
}

func orUnknownAppID(s string) string {
	if s == "" {
		return "(absent)"
	}
	return s
}
