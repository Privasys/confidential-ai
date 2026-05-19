// Package specsync polls the management-service tool-spec endpoint and
// hot-reloads the agent.Catalog whenever the fleet's tool set changes.
//
// This solves the bootstrap problem for confidential-ai running inside
// an enclave: at container start no tool spec is known (env-var
// injection was removed by enclave-os-virtual b92be10 to honour the
// "apps configure themselves at runtime" principle). The puller fetches
// the spec from a static URL configured via --tool-spec-url and applies
// it to a long-lived Catalog. The polling cadence is short enough
// (60s default) that user-visible tool changes take effect within one
// natural latency window without a restart.
//
// Wire format expected from the endpoint:
//
//	GET <tool-spec-url>
//	-> 200 {"spec": "name=https://...?transport=...&...", "generation": "<hex>"}
//
// `generation` is opaque; the puller treats any change as "apply".
// `spec` is the canonical MCP_SERVERS string consumed by
// agent.ParseServerSpec. An empty spec is valid and means "no tools".
package specsync

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/privasys/confidential-ai/internal/agent"
)

// Syncer periodically pulls a tool spec from a URL and applies it to a
// Catalog. One Syncer drives one Catalog. Construct with New and start
// with Run; Run blocks until ctx is cancelled.
type Syncer struct {
	url      string
	token    string
	interval time.Duration
	client   *http.Client
	catalog  *agent.Catalog

	mu      sync.Mutex
	lastGen string
	lastErr error
}

// Response is the JSON envelope returned by the tool-spec endpoint.
// The exported tag set matches management-service's GetEnclaveToolSpec
// handler (platform/management-service/ai_tools.go).
type Response struct {
	Spec       string `json:"spec"`
	Generation string `json:"generation"`
	FleetID    string `json:"fleet_id,omitempty"`
}

// New returns a Syncer configured against url. interval defaults to 60s
// when zero. client defaults to a 10s-timeout client when nil.
func New(url, token string, interval time.Duration, client *http.Client, cat *agent.Catalog) *Syncer {
	if interval <= 0 {
		interval = 60 * time.Second
	}
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	return &Syncer{
		url:      url,
		token:    token,
		interval: interval,
		client:   client,
		catalog:  cat,
	}
}

// LastError returns the most recent fetch/apply error, or nil. Used by
// /healthz to surface staleness without affecting the success of the
// last applied spec.
func (s *Syncer) LastError() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastErr
}

// LastGeneration returns the most recently applied generation tag.
// Empty before the first successful poll.
func (s *Syncer) LastGeneration() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastGen
}

// Run drives the polling loop. The first poll happens immediately so
// the catalogue is populated before the proxy accepts traffic; if it
// fails the loop still continues at interval cadence and logs the
// error. Returns when ctx is done.
func (s *Syncer) Run(ctx context.Context) {
	// First fetch is synchronous (in this goroutine) so callers can
	// log a single "initial pull completed" line before the polling
	// cadence takes over. A failure here is non-fatal: the proxy can
	// still serve plain pass-through chat completions.
	if err := s.pollOnce(ctx); err != nil {
		log.Printf("[tool-spec] initial pull failed: %v", err)
	}

	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := s.pollOnce(ctx); err != nil {
				log.Printf("[tool-spec] pull failed: %v", err)
			}
		}
	}
}

// pollOnce fetches the spec, parses it, and applies it to the catalog
// when the generation tag has changed. Errors are stored in s.lastErr
// for /healthz; the next tick will retry.
func (s *Syncer) pollOnce(ctx context.Context) error {
	resp, err := s.fetch(ctx)
	if err != nil {
		s.recordErr(err)
		return err
	}
	// Empty generation but non-empty spec is a server bug; treat the
	// spec as authoritative and apply on every poll. Empty spec is OK
	// (it means "no tools enabled on this fleet").
	gen := resp.Generation
	s.mu.Lock()
	same := gen != "" && gen == s.lastGen
	s.mu.Unlock()
	if same {
		s.recordOK(gen)
		return nil
	}
	servers, err := agent.ParseServerSpec(resp.Spec)
	if err != nil {
		err = fmt.Errorf("parse spec: %w", err)
		s.recordErr(err)
		return err
	}
	s.catalog.Replace(servers)
	s.recordOK(gen)
	log.Printf("[tool-spec] applied generation=%q servers=%d", gen, len(servers))
	return nil
}

func (s *Syncer) fetch(ctx context.Context) (*Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		return nil, err
	}
	if s.token != "" {
		req.Header.Set("Authorization", "Bearer "+s.token)
	}
	req.Header.Set("Accept", "application/json")
	r, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	if r.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(r.Body, 512))
		return nil, fmt.Errorf("HTTP %d: %s", r.StatusCode, strings.TrimSpace(string(body)))
	}
	var out Response
	if err := json.NewDecoder(r.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &out, nil
}

func (s *Syncer) recordOK(gen string) {
	s.mu.Lock()
	s.lastGen = gen
	s.lastErr = nil
	s.mu.Unlock()
}

func (s *Syncer) recordErr(err error) {
	s.mu.Lock()
	s.lastErr = err
	s.mu.Unlock()
}
