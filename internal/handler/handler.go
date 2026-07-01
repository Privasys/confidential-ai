package handler

import (
	"bufio"
	"bytes"
	"context"
	crand "crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/privasys/confidential-ai/internal/agent"
	"github.com/privasys/confidential-ai/internal/billing"
	"github.com/privasys/confidential-ai/internal/config"
	"github.com/privasys/confidential-ai/internal/models"
	"github.com/privasys/confidential-ai/internal/reproducibility"
)

// Handler is the HTTP handler that proxies to vLLM and injects
// reproducibility metadata into every response.
type Handler struct {
	cfg      *config.Config
	client   *http.Client
	modelMgr *models.Manager

	// agentCatalog + agentDispatcher are non-nil when MCPServers is set
	// in config. When set, /v1/chat/completions runs through the
	// agentic loop instead of a straight pass-through.
	agentCatalog    *agent.Catalog
	agentDispatcher *agent.Dispatcher
	agentConsent    *agent.ConsentRegistry

	// grantVerifier is non-nil when ToolGrantJWKSURL is configured. It
	// verifies the per-request X-Privasys-Tool-Grant header so a user's
	// own tools can be unioned with the configured catalogue for that
	// request only.
	grantVerifier *agent.GrantVerifier

	// oidcVerifier is non-nil when OIDCIssuer is configured. It validates
	// platform bearer tokens offline (JWKS) so privileged endpoints
	// (model load/unload) can require the manager role, mirroring the
	// enclave manager's own auth model.
	oidcVerifier *OIDCVerifier

	// billing meters completed inference and gates requests at zero
	// balance (the pricing model). Held behind an atomic pointer
	// because POST /configure can swap the live Reporter at runtime
	// (env vars are not deliverable to container apps; billing config
	// arrives via the configure-then-freeze pattern). nil/Load()==nil
	// means metering is not configured. All Reporter methods are
	// nil-safe.
	billing atomic.Pointer[billing.Reporter]
	// billingMu serializes Reconfigure/Start so the stop-old/start-new
	// swap is atomic with respect to itself.
	billingMu sync.Mutex
	// billingBaseCtx is the server lifetime context the reporter loops
	// derive from; set by StartBilling. billingStop cancels the loop of
	// the currently-installed Reporter.
	billingBaseCtx context.Context
	billingStop    context.CancelFunc

	// ready is set to 1 once the vLLM upstream health check succeeds.
	// Used only in legacy mode (when model is loaded at boot via entrypoint.sh).
	ready atomic.Int32

	// Metrics
	requestsTotal  atomic.Int64
	requestsFailed atomic.Int64
}

// New creates a Handler with the given config and model manager.
// If modelMgr is nil, falls back to legacy mode (polling vLLM at startup).
func New(cfg *config.Config, modelMgr *models.Manager) *Handler {
	h := &Handler{
		cfg:      cfg,
		modelMgr: modelMgr,
		client: &http.Client{
			Timeout: 5 * time.Minute, // LLM inference can be slow
		},
	}
	// Legacy mode: poll vLLM health at startup if model manager is not used.
	if modelMgr == nil && cfg.ModelName != "" {
		go h.pollUpstreamReady()
	}
	if servers, err := agent.ParseServerSpec(cfg.MCPServers); err != nil {
		log.Printf("[agent] MCP_SERVERS parse error: %v (agentic loop disabled)", err)
	} else if len(servers) > 0 || cfg.ToolSpecURL != "" || cfg.ToolGrantJWKSURL != "" {
		// The catalogue is always created when the puller is enabled,
		// even if the static MCP_SERVERS is empty: the puller will
		// populate it on first poll and may continue to mutate it as
		// the fleet's tool set changes. It is likewise created when only
		// per-request tool grants are enabled, so the agentic path is
		// reachable for fleets whose tools are all user-supplied.
		h.agentCatalog = agent.NewCatalog(servers, &http.Client{Timeout: 10 * time.Second}, 60*time.Second)
		h.agentDispatcher = agent.NewDispatcher(h.agentCatalog, &http.Client{Timeout: 60 * time.Second})
		h.agentConsent = agent.NewConsentRegistry()
		log.Printf("[agent] enabled (static-servers=%d, puller=%v, grants=%v)", len(servers), cfg.ToolSpecURL != "", cfg.ToolGrantJWKSURL != "")
	}
	if cfg.ToolGrantJWKSURL != "" {
		h.grantVerifier = agent.NewGrantVerifier(cfg.ToolGrantJWKSURL, cfg.ToolGrantAudience)
		log.Printf("[agent] per-request tool grants enabled (jwks=%s, aud=%q)", cfg.ToolGrantJWKSURL, cfg.ToolGrantAudience)
	}
	if cfg.OIDCIssuer != "" {
		h.oidcVerifier = NewOIDCVerifier(cfg.OIDCIssuer, cfg.OIDCAudience)
		log.Printf("[auth] OIDC load/unload gate enabled (issuer=%s, role=%s)", cfg.OIDCIssuer, cfg.ManagerRole)
	}
	modelSlug := cfg.BillingModel
	if modelSlug == "" {
		modelSlug = cfg.ModelName
	}
	if rep := billing.New(billing.Config{
		AccountID:   cfg.BillingAccountID,
		ReportURL:   cfg.UsageReportURL,
		ReportToken: cfg.UsageReportToken,
		Model:       modelSlug,
	}); rep != nil {
		h.billing.Store(rep)
		log.Printf("[billing] inference metering enabled from env (account=%s, model=%s)", cfg.BillingAccountID, modelSlug)
	}
	return h
}

// StartBilling starts the background usage-reporting loop, if metering is
// enabled. It records the server lifetime context so that a later
// ReconfigureBilling (via POST /configure) can start the swapped-in
// Reporter. Safe to call when metering is disabled (no-op loop).
func (h *Handler) StartBilling(ctx context.Context) {
	h.billingMu.Lock()
	defer h.billingMu.Unlock()
	h.billingBaseCtx = ctx
	if rep := h.billing.Load(); rep != nil {
		loopCtx, cancel := context.WithCancel(ctx)
		rep.Start(loopCtx)
		h.billingStop = cancel
	}
}

// ReconfigureBilling atomically swaps the live billing Reporter to one
// built from cfg (or disables metering when cfg is disabled). The
// previous Reporter's loop is cancelled and the new one is started under
// the server lifetime context recorded by StartBilling. Safe for
// concurrent callers; the hot path reads h.billing.Load() lock-free.
func (h *Handler) ReconfigureBilling(cfg billing.Config) {
	h.billingMu.Lock()
	defer h.billingMu.Unlock()
	if h.billingStop != nil {
		h.billingStop()
		h.billingStop = nil
	}
	rep := billing.New(cfg)
	if rep != nil && h.billingBaseCtx != nil {
		loopCtx, cancel := context.WithCancel(h.billingBaseCtx)
		rep.Start(loopCtx)
		h.billingStop = cancel
	}
	h.billing.Store(rep)
	if rep != nil {
		log.Printf("[billing] inference metering (re)configured (account=%s, model=%s)", cfg.AccountID, cfg.Model)
	} else {
		log.Printf("[billing] inference metering disabled by configure")
	}
}

// billingReporter returns the currently-installed Reporter (possibly nil).
// All Reporter methods are nil-safe, so callers can chain directly.
func (h *Handler) billingReporter() *billing.Reporter { return h.billing.Load() }

// AgentCatalog exposes the live agent catalogue so external goroutines
// (e.g. the tool-spec puller in cmd/server) can mutate it via
// Catalog.Replace. Returns nil when the agentic loop is disabled (no
// MCP_SERVERS and no --tool-spec-url).
func (h *Handler) AgentCatalog() *agent.Catalog { return h.agentCatalog }

// pollUpstreamReady polls vLLM's /health endpoint until it returns 200,
// then sets h.ready to 1. This runs in the background so the proxy can
// serve user-friendly "loading" responses while the model warms up.
func (h *Handler) pollUpstreamReady() {
	client := &http.Client{Timeout: 5 * time.Second}
	for {
		resp, err := client.Get(h.cfg.VLLMUpstream + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				h.ready.Store(1)
				return
			}
		}
		time.Sleep(5 * time.Second)
	}
}

// IsReady returns whether inference is available (model loaded and serving).
func (h *Handler) IsReady() bool {
	if h.modelMgr != nil {
		return h.modelMgr.IsReady()
	}
	return h.ready.Load() == 1
}

// NotReadyMessage describes WHY inference is unavailable. The historic
// catch-all "Model is loading" was actively misleading when the
// manager was idle (no load ever requested) or a load had failed.
func (h *Handler) NotReadyMessage() string {
	if h.modelMgr == nil {
		return "Model is loading, please wait..."
	}
	switch s := h.modelMgr.Status(); s.State {
	case models.StateLoading:
		return "Model is loading, please wait..."
	case models.StateFailed:
		return "Model load failed: " + s.Error + ". Retry via POST /v1/models/load."
	default:
		return "No model loaded. Load one via POST /v1/models/load."
	}
}

// RegisterRoutes adds all endpoints to the given mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/chat/completions", h.chatCompletions)
	mux.HandleFunc("POST /v1/completions", h.completions)
	mux.HandleFunc("GET /v1/models", h.models)
	mux.HandleFunc("POST /v1/models", h.models)
	mux.HandleFunc("POST /v1/models/load", h.requireLoadAuth(h.modelsLoad))
	mux.HandleFunc("GET /v1/models/status", h.modelsStatus)
	mux.HandleFunc("POST /v1/models/unload", h.requireLoadAuth(h.modelsUnload))
	mux.HandleFunc("GET /readiness", h.readiness)
	mux.HandleFunc("GET /health", h.health)
	mux.HandleFunc("POST /health", h.health)
	mux.HandleFunc("GET /healthz", h.health)
	mux.HandleFunc("POST /healthz", h.health)
	mux.HandleFunc("GET /.well-known/attestation-extensions", h.attestationExtensions)
	mux.HandleFunc("GET /metrics", h.metrics)
	mux.HandleFunc("POST /v1/agent/confirm/{id}", h.agentConfirm)
	mux.HandleFunc("POST /configure", h.configure)
}

// requireLoadAuth gates model load/unload. Authorisation order:
//
//  1. When an OIDC verifier is configured (the default), the caller must
//     present a platform bearer carrying the manager role. This is what the
//     management-service service account presents, and mirrors the enclave
//     manager's own auth model, so no per-app shared secret is needed.
//  2. A non-empty static LoadToken is accepted as a LEGACY FALLBACK for the
//     direct CLI/owner path during migration.
//  3. When neither is configured the endpoint is open (dev mode).
//
// The end user never holds either credential; load/unload is manager-only.
func (h *Handler) requireLoadAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.oidcVerifier == nil && h.cfg.LoadToken == "" {
			next(w, r) // dev mode: no auth configured
			return
		}
		authz := r.Header.Get("Authorization")
		if !strings.HasPrefix(authz, "Bearer ") {
			writeError(w, http.StatusUnauthorized, "missing bearer token")
			return
		}
		token := strings.TrimPrefix(authz, "Bearer ")

		if h.oidcVerifier != nil {
			claims, err := h.oidcVerifier.Verify(r.Context(), token)
			if err == nil {
				if claims.HasRole(h.cfg.ManagerRole) {
					next(w, r)
					return
				}
				writeError(w, http.StatusForbidden, "requires "+h.cfg.ManagerRole+" role")
				return
			}
			// OIDC verification failed: fall back to the legacy static token.
			if h.cfg.LoadToken != "" && subtleEq(token, h.cfg.LoadToken) {
				next(w, r)
				return
			}
			writeError(w, http.StatusUnauthorized, "invalid token")
			return
		}

		// Legacy-only path (OIDC disabled, static token configured).
		if subtleEq(token, h.cfg.LoadToken) {
			next(w, r)
			return
		}
		writeError(w, http.StatusUnauthorized, "invalid token")
	}
}

// subtleEq is a constant-time string comparison.
func subtleEq(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	var v byte
	for i := 0; i < len(a); i++ {
		v |= a[i] ^ b[i]
	}
	return v == 0
}

// chatCompletions proxies to vLLM /v1/chat/completions and injects
// reproducibility metadata into the response.
//
// When an MCP-backed agent dispatcher is configured we normally run the
// bounded tool-call loop so the server can execute tools on behalf of
// the client. That assumes ALL `tools` in the request are server-side
// (the chat UI sends none and lets us inject our catalogue). Clients
// like Zed send their OWN client-side tools (e.g. `list_directory`) and
// expect to dispatch them themselves: they want the raw OpenAI SSE with
// `tool_calls` deltas, NOT our `tool_call` / `tool_result` events, and
// our dispatcher would otherwise reject the unknown names with
// "malformed tool name (expected <server>__<tool>)". Detect that case
// and use the plain pass-through proxy instead.
func (h *Handler) chatCompletions(w http.ResponseWriter, r *http.Request) {
	if h.agentDispatcher != nil {
		body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20))
		if err != nil {
			h.requestsFailed.Add(1)
			writeError(w, http.StatusBadRequest, "failed to read request body")
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		if !hasClientTools(body) {
			h.chatCompletionsAgentic(w, r)
			return
		}
		// Client supplied its own tools (e.g. Zed). It speaks plain
		// OpenAI streaming and cannot parse our `reproducibility` SSE
		// event — its stream parser rejects any non-chat-chunk `data:`
		// frame. Use the strict pass-through path.
		h.proxyPassthrough(w, r, "/v1/chat/completions")
		return
	}
	h.proxyWithReproducibility(w, r, "/v1/chat/completions")
}

// hasClientTools reports whether the request body contains a non-empty
// `tools` array, indicating the caller is supplying client-side tools
// and expects vendor-standard OpenAI tool_call streaming (no server-
// side dispatch).
func hasClientTools(body []byte) bool {
	var probe struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return false
	}
	return len(probe.Tools) > 0
}

// completions proxies to vLLM /v1/completions with reproducibility.
func (h *Handler) completions(w http.ResponseWriter, r *http.Request) {
	h.proxyWithReproducibility(w, r, "/v1/completions")
}

// proxyPassthrough forwards the request to vLLM and streams the
// response back verbatim, WITHOUT injecting the reproducibility
// metadata event. Used for OpenAI-compatible clients that supply
// their own tools and whose strict stream parsers reject any `data:`
// frame that is not a `chat.completion.chunk`.
func (h *Handler) proxyPassthrough(w http.ResponseWriter, r *http.Request, path string) {
	if !h.IsReady() {
		writeError(w, http.StatusServiceUnavailable, h.NotReadyMessage())
		return
	}
	// Balance gate (the pricing model): refuse inference at zero balance
	// on the tool-carrying path too, so it can't be used to bypass billing.
	if h.billingReporter().Frozen() {
		writeError(w, http.StatusPaymentRequired, "Account is out of credit. Add credit to continue.")
		return
	}
	h.requestsTotal.Add(1)

	body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20))
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	// Inject a random seed default (echoed back) so unseeded requests stay
	// replayable while still varying naturally on the pass-through path.
	var reqParams requestParams
	if err := json.Unmarshal(body, &reqParams); err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "invalid JSON request body")
		return
	}
	seed := newSeed()
	if reqParams.Seed != nil {
		seed = *reqParams.Seed
	}
	reqWithSeed, err := injectSeed(body, seed)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject seed")
		return
	}

	upstream := h.cfg.VLLMUpstream + path
	proxyReq, err := http.NewRequestWithContext(r.Context(), "POST", upstream, bytes.NewReader(reqWithSeed))
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(proxyReq)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadGateway, fmt.Sprintf("vLLM upstream error: %v", err))
		return
	}
	defer resp.Body.Close()

	if !reqParams.Stream {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 50<<20))
		if rep := h.billingReporter(); resp.StatusCode == http.StatusOK && rep != nil {
			if id, in, out, ok := extractUsage(respBody); ok {
				rep.Record(id, in, out)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(body)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	buf := make([]byte, 32*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			flusher.Flush()
		}
		if rerr != nil {
			return
		}
	}
}

// models proxies the /v1/models endpoint directly.
func (h *Handler) models(w http.ResponseWriter, r *http.Request) {
	if !h.IsReady() {
		writeError(w, http.StatusServiceUnavailable, h.NotReadyMessage())
		return
	}
	h.proxyDirect(w, r, "/v1/models")
}

// proxyWithReproducibility reads the request, proxies to vLLM, then
// wraps the response with reproducibility metadata.
// Supports both streaming (SSE) and non-streaming responses.
func (h *Handler) proxyWithReproducibility(w http.ResponseWriter, r *http.Request, path string) {
	if !h.IsReady() {
		writeError(w, http.StatusServiceUnavailable, h.NotReadyMessage())
		return
	}

	// Balance gate (the pricing model): refuse inference once the
	// account is out of credit. The freeze state is maintained by the
	// billing reporter from the management-service's usage responses.
	if h.billingReporter().Frozen() {
		writeError(w, http.StatusPaymentRequired, "Account is out of credit. Add credit to continue.")
		return
	}

	h.requestsTotal.Add(1)

	// Read the incoming request body
	body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20)) // 10 MiB limit
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}

	// Parse request to extract sampling parameters for metadata
	var reqParams requestParams
	if err := json.Unmarshal(body, &reqParams); err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "invalid JSON request body")
		return
	}

	// Default to a random seed (echoed in the reproducibility block) so
	// chat output varies naturally yet every response stays replayable.
	if reqParams.Seed == nil {
		defaultSeed := newSeed()
		reqParams.Seed = &defaultSeed
	}
	if reqParams.Temperature == 0 {
		reqParams.Temperature = 1.0
	}
	if reqParams.TopP == 0 {
		reqParams.TopP = 1.0
	}

	// Ensure seed is always sent to vLLM for reproducibility
	reqWithSeed, err := injectSeed(body, *reqParams.Seed)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject seed")
		return
	}

	// Inject the per-request dynamic context (wall clock) just before the
	// latest user turn, and record it for reproducible replay. Kept out of the
	// static system prompt so that prompt stays a stable, cacheable prefix.
	dynCtx := dynamicContext(r)
	reqWithSeed, err = injectDynamicContext(reqWithSeed, dynCtx)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject context")
		return
	}

	// Billing (the pricing model): when metering is enabled, streaming
	// responses carry no token usage unless we ask vLLM for it. Inject
	// stream_options.include_usage so the upstream emits a final usage
	// chunk; we record it and (unless the client opted in itself) strip it
	// from the forwarded stream so the response shape is unchanged.
	var meter *meterCtx
	if rep := h.billingReporter(); rep != nil {
		meter = &meterCtx{reporter: rep}
		if reqParams.Stream {
			clientHadUsage := false
			reqWithSeed, clientHadUsage = injectStreamUsage(reqWithSeed)
			meter.suppressUsage = !clientHadUsage
		}
	}

	// The `model` field is forwarded verbatim. The proxy never silently
	// rewrites it: callers are expected to send the canonical name that
	// vLLM is serving (which is exactly what `GET /v1/models` and the
	// management-service instance discovery endpoint publish). If the
	// name does not match what vLLM has loaded, vLLM returns its own
	// 404 - that is the correct behaviour, since silently switching the
	// model would invalidate the reproducibility metadata that this
	// endpoint promises.

	// Forward to vLLM
	upstream := h.cfg.VLLMUpstream + path
	proxyReq, err := http.NewRequestWithContext(r.Context(), "POST", upstream, bytes.NewReader(reqWithSeed))
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		return
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(proxyReq)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadGateway, fmt.Sprintf("vLLM upstream error: %v", err))
		return
	}
	defer resp.Body.Close()

	// Build reproducibility metadata.
	// Use dynamic model info from manager if available, else fall back to config.
	modelName := h.cfg.ModelName
	quantization := h.cfg.Quantization
	if h.modelMgr != nil {
		if n := h.modelMgr.ModelName(); n != "" {
			modelName = n
		}
		if q := h.modelMgr.Quantization(); q != "" {
			quantization = q
		}
	}
	meta := reproducibility.NewMetadata(
		*reqParams.Seed,
		reqParams.Temperature,
		reqParams.TopP,
		reqParams.TopK,
		reqParams.MaxTokens,
		modelName,
		quantization,
		h.cfg.VLLMVersion,
		h.cfg.CUDAVersion,
		h.cfg.GPUType,
		h.cfg.ImageDigest,
		h.cfg.TeeType,
	)
	meta.DynamicContext = dynCtx

	wantRepro := wantsReproducibility(r)
	if reqParams.Stream {
		h.proxyStream(w, resp, meta, wantRepro, meter)
	} else {
		h.proxyNonStream(w, resp, meta, wantRepro, meter)
	}
}

// meterCtx carries per-request billing state into the proxy response path.
// nil when inference metering is disabled.
type meterCtx struct {
	reporter *billing.Reporter
	// suppressUsage strips the synthetic stream usage chunk we injected via
	// stream_options.include_usage so a client that did not opt in still sees
	// an unchanged stream.
	suppressUsage bool
}

// record forwards extracted token counts to the reporter. Safe on nil.
func (m *meterCtx) record(requestID string, in, out int64) {
	if m == nil {
		return
	}
	m.reporter.Record(requestID, in, out)
}

// wantsReproducibility reports whether the caller opted in to the
// privasys-specific reproducibility extension. Without the opt-in we
// emit a stock OpenAI response so strict clients (e.g. Zed) don't
// reject the trailing `data: {"reproducibility":...}` SSE frame.
// Our chat front-end and SDK set this header; OpenAI-compatible
// integrations don't.
func wantsReproducibility(r *http.Request) bool {
	v := r.Header.Get("X-Privasys-Reproducibility")
	return v != "" && v != "0" && !strings.EqualFold(v, "false")
}

// proxyNonStream handles non-streaming responses: read full body, inject metadata.
func (h *Handler) proxyNonStream(w http.ResponseWriter, resp *http.Response, meta *reproducibility.Metadata, wantRepro bool, meter *meterCtx) {
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 50<<20)) // 50 MiB limit
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadGateway, "failed to read upstream response")
		return
	}

	// If vLLM returned an error, pass it through
	if resp.StatusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}

	// Meter token usage from the upstream body (present in every successful
	// vLLM completion, regardless of the reproducibility opt-in).
	if id, in, out, ok := extractUsage(respBody); ok {
		meter.record(id, in, out)
	}

	if !wantRepro {
		// Pass the upstream body through verbatim.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(respBody)
		return
	}

	// Parse vLLM response and inject metadata
	var vllmResp map[string]any
	if err := json.Unmarshal(respBody, &vllmResp); err != nil {
		// If we can't parse, still return the original + metadata as wrapper
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{
			"response":        json.RawMessage(respBody),
			"reproducibility": meta,
		})
		return
	}

	vllmResp["reproducibility"] = meta

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(vllmResp)
}

// proxyStream handles streaming SSE responses from vLLM.
//
// Forwarding strategy: read **whole SSE events** (`data: …\n\n` /
// `event: …\ndata: …\n\n`) and forward each one as a single Write +
// Flush. Two reasons:
//
//   - Each downstream sealed frame (sessionrelay) becomes one complete
//     SSE event. The browser SSE parser fires per frame, so the chat
//     UI sees one delta per pacing tick instead of "data line then
//     blank line" arriving as two AEAD frames (which doubled per-token
//     overhead and added nothing usable).
//   - The previous implementation used a `bufio.Scanner` line loop and
//     an `fmt.Fprintf("%s\n", line)` per line, which split each event
//     across two Write+Flush calls. Switching to event-aligned
//     forwarding removes that latency artefact and the redundant
//     formatting cost on the hot path.
//
// We also short-circuit `data: [DONE]` so the metadata event is
// emitted *before* the sentinel even when vLLM packs both lines into
// one TCP read.
func (h *Handler) proxyStream(w http.ResponseWriter, resp *http.Response, meta *reproducibility.Metadata, wantRepro bool, meter *meterCtx) {
	if resp.StatusCode != http.StatusOK {
		// Error responses are not streamed; read and pass through.
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(body)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	br := bufio.NewReaderSize(resp.Body, 64*1024)

	emitMeta := func() {
		if !wantRepro {
			return
		}
		metaJSON, err := json.Marshal(map[string]any{"reproducibility": meta})
		if err != nil {
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", metaJSON)
		flusher.Flush()
	}

	// Buffer for accumulating one SSE event (terminated by "\n\n").
	var event []byte
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			event = append(event, line...)
			// Event terminator: a blank line, i.e. the buffer ends in
			// "\n\n". We compare on raw bytes so CRLF separators (rare
			// in practice; vLLM emits LF) still work.
			if bytes.HasSuffix(event, []byte("\n\n")) || bytes.Equal(event, []byte("\n")) {
				// Billing: vLLM's final include_usage chunk carries the
				// token totals with an empty choices array. Record it and,
				// unless the client opted in to include_usage itself, drop
				// it from the forwarded stream so the response is unchanged.
				if meter != nil {
					if id, in, out, ok := extractStreamUsage(event); ok {
						meter.record(id, in, out)
						if meter.suppressUsage {
							event = event[:0]
							continue
						}
					}
				}
				if isDoneEvent(event) {
					emitMeta()
					_, _ = w.Write(event)
					flusher.Flush()
					event = event[:0]
					continue
				}
				if _, werr := w.Write(event); werr != nil {
					return
				}
				flusher.Flush()
				event = event[:0]
			}
		}
		if err != nil {
			// Best-effort tail flush of an unterminated trailing event.
			if len(event) > 0 {
				_, _ = w.Write(event)
				flusher.Flush()
			}
			if err != io.EOF {
				h.requestsFailed.Add(1)
			}
			return
		}
	}
}

// isDoneEvent reports whether the SSE event block ends with the
// `data: [DONE]` sentinel (ignoring trailing whitespace and any
// preceding event:/id: lines).
func isDoneEvent(event []byte) bool {
	for _, line := range bytes.Split(bytes.TrimRight(event, "\n"), []byte("\n")) {
		if bytes.Equal(bytes.TrimSpace(line), []byte("data: [DONE]")) {
			return true
		}
	}
	return false
}

// proxyDirect forwards the request to vLLM without modification.
func (h *Handler) proxyDirect(w http.ResponseWriter, r *http.Request, path string) {
	upstream := h.cfg.VLLMUpstream + path
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstream, r.Body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		return
	}
	for k, vv := range r.Header {
		for _, v := range vv {
			proxyReq.Header.Add(k, v)
		}
	}

	resp, err := h.client.Do(proxyReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, fmt.Sprintf("vLLM upstream error: %v", err))
		return
	}
	defer resp.Body.Close()

	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// health returns server health status. Always returns 200 to indicate the
// container is alive. Use /readiness to check if a model is serving.
func (h *Handler) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	modelName := h.cfg.ModelName
	modelDigest := h.cfg.ModelDigest
	quantization := h.cfg.Quantization
	modelState := "unknown"

	if h.modelMgr != nil {
		status := h.modelMgr.Status()
		modelState = string(status.State)
		if status.Model != "" {
			modelName = status.Model
		}
		if status.ModelDigest != "" {
			modelDigest = status.ModelDigest
		}
		if q := h.modelMgr.Quantization(); q != "" {
			quantization = q
		}
	} else if h.IsReady() {
		modelState = "ready"
	} else {
		modelState = "loading"
	}

	json.NewEncoder(w).Encode(map[string]any{
		"status":       "ok",
		"model_state":  modelState,
		"model":        modelName,
		"model_digest": modelDigest,
		"quantization": quantization,
		"gpu":          h.cfg.GPUType,
		"tee":          h.cfg.TeeType,
		"vllm_version": h.cfg.VLLMVersion,
		"image_digest": h.cfg.ImageDigest,
	})
}

// readiness returns 200 when a model is loaded and serving, 503 otherwise.
// Use this for load balancing or readiness probes.
func (h *Handler) readiness(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if !h.IsReady() {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "not_ready",
			"message": "No model loaded or model is still loading",
		})
		return
	}
	json.NewEncoder(w).Encode(map[string]string{"status": "ready"})
}

// modelsLoad handles POST /v1/models/load - starts loading a model.
func (h *Handler) modelsLoad(w http.ResponseWriter, r *http.Request) {
	if h.modelMgr == nil {
		writeError(w, http.StatusNotImplemented, "dynamic model loading not available (legacy mode)")
		return
	}

	var req models.LoadRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model field is required")
		return
	}

	if err := h.modelMgr.Load(req); err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(h.modelMgr.Status())
}

// modelsStatus handles GET /v1/models/status - returns model loading state
// plus the slugs of models available on disk (so an orchestrator/portal can
// offer a pick-list without a separate round-trip). `available` is omitted
// on read errors rather than failing the status call.
func (h *Handler) modelsStatus(w http.ResponseWriter, _ *http.Request) {
	if h.modelMgr == nil {
		writeError(w, http.StatusNotImplemented, "dynamic model loading not available (legacy mode)")
		return
	}

	avail, _ := h.modelMgr.ListAvailable()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(struct {
		models.Status
		Available []string `json:"available,omitempty"`
	}{h.modelMgr.Status(), avail})
}

// modelsUnload handles POST /v1/models/unload - stops vLLM and frees resources.
func (h *Handler) modelsUnload(w http.ResponseWriter, _ *http.Request) {
	if h.modelMgr == nil {
		writeError(w, http.StatusNotImplemented, "dynamic model loading not available (legacy mode)")
		return
	}

	if err := h.modelMgr.Unload(); err != nil {
		writeError(w, http.StatusInternalServerError, "unload failed: "+err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "idle"})
}

// attestationExtensions serves custom OID extensions for RA-TLS certificates.
// This is the Virtual equivalent of enclave-os-mini's custom_oids() trait
// method: the container declares its own attestation OIDs, and Caddy's RA-TLS
// module pulls them at certificate issuance time.
//
// Serves OID 1.3.6.1.4.1.65230.3.5 (MODEL_DIGEST) when a model digest is
// available. The digest is sourced, in order of preference:
//
//  1. The dm-verity root hash of the mounted model disk
//     (<RoothashDir>/<model>.roothash, written by disk-mounter).
//  2. The dynamic digest computed by the model manager from the
//     safetensors index (legacy fallback).
//  3. The static --model-digest config value.
func (h *Handler) attestationExtensions(w http.ResponseWriter, _ *http.Request) {
	type entry struct {
		OID   string `json:"oid"`
		Value string `json:"value"`
	}
	var exts []entry

	// Use dynamic model digest from manager if available, else fall back to config.
	digest := h.cfg.ModelDigest
	if h.modelMgr != nil {
		if d := h.modelMgr.ModelDigest(); d != "" {
			digest = d
		}
	}

	if digest != "" {
		digestBytes, err := hex.DecodeString(digest)
		if err == nil && len(digestBytes) > 0 {
			exts = append(exts, entry{
				OID:   "1.3.6.1.4.1.65230.3.5",
				Value: base64.StdEncoding.EncodeToString(digestBytes),
			})
		}
	}
	// OID 1.3.6.1.4.1.65230.3.7 (TOOLS_DIGEST): sha256 over the
	// canonical, sorted JSON of the configured MCP servers (name,
	// base_url, transport, auth_mode, audience, confirm). A verifier
	// can recompute this from the management-service ai_tools rows
	// to prove the container is exposing the expected toolset.
	if h.agentCatalog != nil {
		if td := h.agentCatalog.ServersDigest(); td != "" {
			if tdBytes, err := hex.DecodeString(td); err == nil && len(tdBytes) > 0 {
				exts = append(exts, entry{
					OID:   "1.3.6.1.4.1.65230.3.7",
					Value: base64.StdEncoding.EncodeToString(tdBytes),
				})
			}
		}
	}
	if exts == nil {
		exts = []entry{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(exts)
}

// metrics returns Prometheus-compatible metrics.
func (h *Handler) metrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	fmt.Fprintf(w, "# HELP confidential_ai_requests_total Total inference requests.\n")
	fmt.Fprintf(w, "# TYPE confidential_ai_requests_total counter\n")
	fmt.Fprintf(w, "confidential_ai_requests_total %d\n", h.requestsTotal.Load())
	fmt.Fprintf(w, "# HELP confidential_ai_requests_failed_total Failed inference requests.\n")
	fmt.Fprintf(w, "# TYPE confidential_ai_requests_failed_total counter\n")
	fmt.Fprintf(w, "confidential_ai_requests_failed_total %d\n", h.requestsFailed.Load())
}

// requestParams captures the sampling parameters from an OpenAI-compatible request.
type requestParams struct {
	Seed        *int64  `json:"seed"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	TopK        int     `json:"top_k"`
	MaxTokens   int     `json:"max_tokens"`
	Model       string  `json:"model"`
	Stream      bool    `json:"stream"`
}

// newSeed returns a fresh random seed in numpy's valid range [0, 2^32).
// vLLM feeds the seed to numpy/torch RNG, and numpy rejects seeds outside
// that range. We default unseeded requests to a random seed (rather than a
// fixed 0) so chat output varies naturally; the seed is echoed in the
// reproducibility block, so any response remains replayable token-for-token.
func newSeed() int64 {
	var b [4]byte
	if _, err := crand.Read(b[:]); err != nil {
		// crypto/rand failure is effectively unreachable; fall back to a
		// non-zero constant rather than silently pinning every request to 0.
		return 1
	}
	return int64(binary.BigEndian.Uint32(b[:]))
}

// injectSeed ensures the "seed" field is present in the request JSON.
func injectSeed(body []byte, seed int64) ([]byte, error) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	if _, ok := m["seed"]; !ok {
		m["seed"] = seed
	}
	return json.Marshal(m)
}

// dynamicContext returns the per-request context block injected just before the
// latest user turn — currently the wall-clock time, so the model can answer
// "what time is it?" without baking a timestamp into the (cacheable) system
// prompt. A replay passes the recorded value back via the
// X-Privasys-Dynamic-Context header so the prompt is reconstructed
// byte-for-byte and the response stays reproducible; otherwise it is built from
// the current UTC time. Returns "" only if the override header is explicitly
// blank-after-trim AND there is no clock (unreachable), in practice non-empty.
func dynamicContext(r *http.Request) string {
	if v := strings.TrimSpace(r.Header.Get("X-Privasys-Dynamic-Context")); v != "" {
		return v
	}
	return "The current date and time is " + time.Now().UTC().Format(time.RFC3339) +
		" (UTC). Treat this as the present moment when answering."
}

// injectDynamicContext inserts a dedicated system message carrying the
// per-request context (the wall clock) immediately BEFORE the last user turn.
// A reasoning model disregards a context line buried inside the user message
// (it falls back to "I have no clock"), so the context must arrive with system
// authority, the way OpenAI/Anthropic feed the current date. Placing it right
// before the last user turn keeps the static system prompt and the conversation
// history a stable, cacheable prefix (the time, which changes every request,
// never perturbs that prefix). The exact string is recorded in the
// reproducibility metadata (DynamicContext) for deterministic replay.
//
// Returns the body unchanged when ctx is empty or there is no user message.
func injectDynamicContext(body []byte, ctx string) ([]byte, error) {
	if ctx == "" {
		return body, nil
	}
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	msgs, ok := m["messages"].([]any)
	if !ok || len(msgs) == 0 {
		return body, nil
	}
	last := -1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msg, ok := msgs[i].(map[string]any); ok && msg["role"] == "user" {
			last = i
			break
		}
	}
	if last < 0 {
		return body, nil // no user message to anchor against
	}
	sysMsg := map[string]any{"role": "system", "content": ctx}
	// Insert sysMsg at index `last`, shifting the user turn (and anything
	// after it) right by one.
	msgs = append(msgs, nil)
	copy(msgs[last+1:], msgs[last:])
	msgs[last] = sysMsg
	m["messages"] = msgs
	return json.Marshal(m)
}

// usageEnvelope is the minimal shape parsed off a vLLM completion (streaming
// or non-streaming) to extract billable token counts for metering.
type usageEnvelope struct {
	ID    string `json:"id"`
	Usage *struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
}

// extractUsage parses the token counts from a non-streaming vLLM completion
// body. ok is false when the body has no usage block (e.g. an error response).
func extractUsage(body []byte) (id string, in, out int64, ok bool) {
	var e usageEnvelope
	if err := json.Unmarshal(body, &e); err != nil || e.Usage == nil {
		return "", 0, 0, false
	}
	return e.ID, e.Usage.PromptTokens, e.Usage.CompletionTokens, true
}

// extractStreamUsage parses the token counts from a single SSE event. vLLM
// emits the usage totals (with an empty choices array) as the final chunk when
// stream_options.include_usage is set. Returns ok=false for ordinary delta
// chunks, the [DONE] sentinel, and the injected reproducibility frame.
func extractStreamUsage(event []byte) (id string, in, out int64, ok bool) {
	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data:"):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var e usageEnvelope
		if err := json.Unmarshal(data, &e); err != nil || e.Usage == nil {
			continue
		}
		// A usage-bearing chunk: prompt_tokens is only populated on the
		// final include_usage frame, not on per-token deltas.
		if e.Usage.PromptTokens > 0 || e.Usage.CompletionTokens > 0 {
			return e.ID, e.Usage.PromptTokens, e.Usage.CompletionTokens, true
		}
	}
	return "", 0, 0, false
}

// injectStreamUsage sets stream_options.include_usage=true so vLLM appends a
// final usage chunk we can meter. It returns the modified body and whether the
// client had already requested include_usage (in which case the usage chunk
// must NOT be stripped from the forwarded stream). On parse failure the body is
// returned unchanged.
func injectStreamUsage(body []byte) ([]byte, bool) {
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		return body, false
	}
	clientHadUsage := false
	opts, _ := m["stream_options"].(map[string]any)
	if opts == nil {
		opts = map[string]any{}
	} else if v, ok := opts["include_usage"].(bool); ok && v {
		clientHadUsage = true
	}
	opts["include_usage"] = true
	m["stream_options"] = opts
	out, err := json.Marshal(m)
	if err != nil {
		return body, clientHadUsage
	}
	return out, clientHadUsage
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
