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
	"sync/atomic"
	"time"
)

// Config configures a Reporter. Disabled() reports whether metering should run.
type Config struct {
	AccountID   string // deployment-owner account charged for inference
	ReportURL   string // management-service AI-usage endpoint
	ReportToken string // EnclaveToken bearer
	Model       string // model slug for price-book lookup
}

// Disabled reports whether the reporter should be a no-op (missing config).
func (c Config) Disabled() bool {
	return c.AccountID == "" || c.ReportURL == ""
}

// usage is one completed inference request's token counts, plus the caller it
// is attributed to (empty = charge the deployment-owner account) and the
// model slug it was served by (empty = the reporter's default model — the
// chat LLM; the embed/rerank instances pass their own slug so each model
// meters under its own ledger resource).
type usage struct {
	RequestID    string
	Caller       string
	Model        string
	InputTokens  int64
	OutputTokens int64
}

// reportRequest is the JSON body posted to the AI-usage endpoint. It mirrors
// management-service's aiUsageRequest.
type reportRequest struct {
	AccountID string            `json:"account_id"`
	Model     string            `json:"model"`
	Requests  []reportRequestLn `json:"requests"`
}

type reportRequestLn struct {
	RequestID string `json:"request_id"`
	CallerSub string `json:"caller_sub,omitempty"`
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
	frozen atomic.Bool

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
			CallerSub:    u.Caller,
			Model:        u.Model,
			InputTokens:  u.InputTokens,
			OutputTokens: u.OutputTokens,
		})
	}
	buf, err := json.Marshal(body)
	if err != nil {
		return
	}
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, r.cfg.ReportURL, bytes.NewReader(buf))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if r.cfg.ReportToken != "" {
		req.Header.Set("Authorization", "Bearer "+r.cfg.ReportToken)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		log.Printf("[billing] usage report failed: %v", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Printf("[billing] usage report status %d", resp.StatusCode)
		return
	}
	var out reportResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return
	}
	r.frozen.Store(out.Frozen)
}
