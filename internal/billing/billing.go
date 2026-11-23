// Package billing meters completed inference requests and reports the
// vLLM-observed token counts to the management-service AI-usage endpoint,
// which performs the priced credit-ledger debit (the pricing model).
//
// Design constraints:
//
//   - The GPU enclave never holds a ledger credential. It authenticates to
//     the management-service with a usage-only EnclaveToken; that service
//     owns the grant-capable ledger token and the account mapping.
//   - Metering must never block or fail user-facing inference. Reporting is
//     asynchronous and best-effort; a full queue or a down endpoint drops
//     usage rather than stalling the hot path.
//   - The proxy refuses inference once the account is frozen (zero balance).
//     The freeze state is learned from the usage endpoint's response and from
//     a periodic balance probe (an empty report), so a long-idle enclave still
//     learns it has gone negative.
package billing

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// Config configures a Reporter. Disabled() reports whether metering should run.
type Config struct {
	AccountID   string // deployment-owner account charged for inference
	ReportURL   string // management-service AI-usage endpoint
	ReportToken string // EnclaveToken bearer
	Model       string // model slug for price-book lookup

	// RelayURL is the enclave manager's per-container base
	// ($PRIVASYS_MANAGER_URL/api/v1/containers/$PRIVASYS_CONTAINER_NAME). When
	// set, usage and billable-caller checks go through it with RelayToken (the
	// container token): the manager forwards them with the enclave's own
	// attested credential and stamps the app id, so no account or usage token
	// is needed here. The legacy fields above remain the fallback while a host
	// still runs a runtime without the relay.
	RelayURL   string
	RelayToken string
}

// relay reports whether the manager relay is configured.
func (c Config) relay() bool { return c.RelayURL != "" && c.RelayToken != "" }

// legacy reports whether the direct AI-usage endpoint is configured.
func (c Config) legacy() bool { return c.AccountID != "" && c.ReportURL != "" }

// Disabled reports whether the reporter should be a no-op (missing config).
func (c Config) Disabled() bool {
	return !c.relay() && !c.legacy()
}

// usage is one completed inference request's token counts, plus the caller it
// is attributed to (empty = charge the deployment-owner account) and the
// model slug it was served by (empty = the reporter's default model — the
// chat LLM; the embed/rerank instances pass their own slug so each model
// meters under its own ledger resource).
type usage struct {
	RequestID    string
	Caller       Caller
	Model        string
	InputTokens  int64
	OutputTokens int64
}

// Caller is who a request is attributed to: the verified end-user subject
// (empty = the deployment-owner account) and, when the user reached us
// through an app they allowed to spend for them (a spend token), that app
// and the consent session, so the ledger records the spend under the app
// and the user's per-app cap sees it.
type Caller struct {
	Sub string
	App string // undashed hex app id (spend-token azp), or ""
	SID string // consent session, or ""
	// Ledger is the spend token's ledger handle: how the platform bills the
	// caller without a subject, or "".
	Ledger string
}

// reportRequest is the JSON body posted to the AI-usage endpoint. It mirrors
// management-service's aiUsageRequest.
type reportRequest struct {
	// AccountID is omitted on the relay: the control plane charges the
	// app's owner account.
	AccountID string            `json:"account_id,omitempty"`
	Model     string            `json:"model"`
	Requests  []reportRequestLn `json:"requests"`
}

type reportRequestLn struct {
	RequestID string `json:"request_id"`
	CallerSub string `json:"caller_sub,omitempty"`
	// CallerLedger is the spend token's ledger handle; the platform debits
	// by it and ignores caller_sub when it is present.
	CallerLedger string `json:"caller_ledger,omitempty"`
	// PayerApp / SID: the spender app and consent session behind a
	// spend-token caller (see Caller).
	PayerApp string `json:"payer_app,omitempty"`
	SID      string `json:"sid,omitempty"`
	// Model overrides the batch-level model slug for this request
	// (embed/rerank usage in a multi-model fleet). Empty means the
	// batch-level model.
	Model        string `json:"model,omitempty"`
	InputTokens  int64  `json:"input_tokens"`
	OutputTokens int64  `json:"output_tokens"`
}

// reportResponse is the AI-usage endpoint's reply.
type reportResponse struct {
	Frozen  bool `json:"frozen"`
	Metered int  `json:"metered"`
}

// Reporter batches inference token usage and pushes it to the management
// service. The zero value is not usable; construct with New.
type Reporter struct {
	cfg    Config
	client *http.Client
	queue  chan usage

	// Per-caller billability cache (see callercheck.go). Lazily built.
	billableMu sync.Mutex
	billable   map[string]billableEntry
	frozen     atomic.Bool

	// relayDownUntil (unix nanoseconds) is set when the manager answered
	// that it cannot relay (404: a runtime without the route; 503: relay not
	// configured on the enclave). Until then the legacy endpoint is used,
	// then the relay is tried again.
	relayDownUntil atomic.Int64

	// probeEvery is how often an idle reporter posts an empty batch to refresh
	// the freeze state. Kept short enough that a newly-exhausted account is
	// gated within roughly one interval even without traffic.
	probeEvery time.Duration
	// flushEvery bounds reporting latency when usage is trickling in.
	flushEvery time.Duration
	// maxBatch caps how many requests are coalesced into a single POST.
	maxBatch int
}

// New returns a Reporter, or nil when metering is disabled by config. Call
// Start to begin the background reporting loop.
func New(cfg Config) *Reporter {
	if cfg.Disabled() {
		return nil
	}
	return &Reporter{
		cfg:        cfg,
		client:     &http.Client{Timeout: 10 * time.Second},
		queue:      make(chan usage, 4096),
		probeEvery: 30 * time.Second,
		flushEvery: 2 * time.Second,
		maxBatch:   256,
	}
}

// Frozen reports whether the account is currently out of credit. Inference
// should be refused while true. A nil Reporter is never frozen (metering off).
func (r *Reporter) Frozen() bool {
	if r == nil {
		return false
	}
	return r.frozen.Load()
}

// Record enqueues a completed request's token usage, attributed to caller (the
// verified end-user subject; empty charges the deployment-owner account) and
// to model (the serving instance's slug; empty means the reporter's default
// model). It never blocks: if the queue is full the sample is dropped
// (best-effort metering). A nil Reporter and zero-token samples are ignored.
func (r *Reporter) Record(requestID, caller, model string, inputTokens, outputTokens int64) {
	r.RecordFor(requestID, Caller{Sub: caller}, model, inputTokens, outputTokens)
}

// RecordFor is Record with the full caller attribution (spender app and
// consent session for a spend-token caller).
func (r *Reporter) RecordFor(requestID string, caller Caller, model string, inputTokens, outputTokens int64) {
	if r == nil || (inputTokens <= 0 && outputTokens <= 0) {
		return
	}
	if model == r.cfg.Model {
		model = "" // batch-level default; keeps the wire body compact
	}
	select {
	case r.queue <- usage{RequestID: requestID, Caller: caller, Model: model, InputTokens: inputTokens, OutputTokens: outputTokens}:
	default:
		log.Printf("[billing] usage queue full; dropping sample (req=%s)", requestID)
	}
}

// Start runs the background batch-and-report loop until ctx is cancelled.
func (r *Reporter) Start(ctx context.Context) {
	if r == nil {
		return
	}
	go r.loop(ctx)
}

func (r *Reporter) loop(ctx context.Context) {
	flush := time.NewTicker(r.flushEvery)
	probe := time.NewTicker(r.probeEvery)
	defer flush.Stop()
	defer probe.Stop()

	var batch []usage
	drain := func() {
		// Pull everything currently queued, up to maxBatch.
		for len(batch) < r.maxBatch {
			select {
			case u := <-r.queue:
				batch = append(batch, u)
			default:
				return
			}
		}
	}

	for {
		select {
		case <-ctx.Done():
			drain()
			if len(batch) > 0 {
				r.send(context.Background(), batch)
			}
			return
		case u := <-r.queue:
			batch = append(batch, u)
			drain()
			if len(batch) >= r.maxBatch {
				r.send(ctx, batch)
				batch = batch[:0]
			}
		case <-flush.C:
			drain()
			if len(batch) > 0 {
				r.send(ctx, batch)
				batch = batch[:0]
			}
		case <-probe.C:
			// Refresh the freeze state even when idle.
			if len(batch) == 0 {
				r.send(ctx, nil)
			}
		}
	}
}

// send posts a batch to the AI-usage endpoint and updates the freeze state
// from the response. A nil/empty batch is a pure balance probe.
func (r *Reporter) send(ctx context.Context, batch []usage) {
	body := reportRequest{
		AccountID: r.cfg.AccountID,
		Model:     r.cfg.Model,
		Requests:  make([]reportRequestLn, 0, len(batch)),
	}
	for _, u := range batch {
		body.Requests = append(body.Requests, reportRequestLn{
			RequestID:    u.RequestID,
			CallerSub:    u.Caller.Sub,
			CallerLedger: u.Caller.Ledger,
			PayerApp:     u.Caller.App,
			SID:          u.Caller.SID,
			Model:        u.Model,
			InputTokens:  u.InputTokens,
			OutputTokens: u.OutputTokens,
		})
	}
	if r.useRelay() {
		relayed := body
		relayed.AccountID = ""
		status, out, err := r.post(ctx, r.cfg.RelayURL+"/usage", r.cfg.RelayToken, relayed)
		switch {
		case err != nil:
			log.Printf("[billing] usage relay failed: %v", err)
			return
		case status == http.StatusOK:
			r.frozen.Store(out.Frozen)
			return
		case !r.relayUnavailable(status):
			// 502: the control plane is unreachable from the enclave, a
			// transient failure like a failed direct call.
			log.Printf("[billing] usage relay status %d", status)
			return
		}
		// The manager cannot relay: report directly, as before.
	}
	if !r.cfg.legacy() {
		log.Printf("[billing] usage not reported: no relay and no direct endpoint")
		return
	}
	status, out, err := r.post(ctx, r.cfg.ReportURL, r.cfg.ReportToken, body)
	if err != nil {
		log.Printf("[billing] usage report failed: %v", err)
		return
	}
	if status != http.StatusOK {
		log.Printf("[billing] usage report status %d", status)
		return
	}
	r.frozen.Store(out.Frozen)
}

// useRelay reports whether the next call should go through the manager.
func (r *Reporter) useRelay() bool {
	return r.cfg.relay() && time.Now().UnixNano() >= r.relayDownUntil.Load()
}

// relayRetry is how long the legacy endpoint is used after the manager said
// it cannot relay, before the relay is tried again.
const relayRetry = 10 * time.Minute

// relayUnavailable records a manager answer that means "this enclave cannot
// relay" and reports whether status was one: 404 (a runtime without the
// relay route) or 503 (relay not configured on the enclave), and 403 (the
// container was loaded without an app id), which is a deployment fault
// logged loudly. The caller then falls back to the direct endpoint.
func (r *Reporter) relayUnavailable(status int) bool {
	switch status {
	case http.StatusNotFound, http.StatusServiceUnavailable:
	case http.StatusForbidden:
		log.Printf("[billing] ERROR: the manager refused to relay usage (403): this container was loaded without an app id")
	default:
		return false
	}
	now := time.Now()
	if prev := r.relayDownUntil.Swap(now.Add(relayRetry).UnixNano()); prev <= now.UnixNano() {
		log.Printf("[billing] usage relay unavailable (status %d); reporting directly for %v", status, relayRetry)
	}
	return true
}

// post sends one JSON report and decodes the reply on a 200.
func (r *Reporter) post(ctx context.Context, url, token string, body reportRequest) (int, reportResponse, error) {
	var out reportResponse
	buf, err := json.Marshal(body)
	if err != nil {
		return 0, out, err
	}
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(buf))
	if err != nil {
		return 0, out, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return 0, out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		err = json.NewDecoder(resp.Body).Decode(&out)
	}
	return resp.StatusCode, out, err
}
