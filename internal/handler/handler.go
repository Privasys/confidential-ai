package handler

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/privasys/confidential-ai/internal/agent"
	"github.com/privasys/confidential-ai/internal/config"
	"github.com/privasys/confidential-ai/internal/models"
	"github.com/privasys/confidential-ai/internal/reproducibility"
)

// Handler is the HTTP handler that proxies to vLLM and injects
// reproducibility metadata into every response.
type Handler struct {
	cfg    *config.Config
	client *http.Client
	modelMgr *models.Manager

	// agentCatalog + agentDispatcher are non-nil when MCPServers is set
	// in config. When set, /v1/chat/completions runs through the
	// agentic loop instead of a straight pass-through.
	agentCatalog    *agent.Catalog
	agentDispatcher *agent.Dispatcher
	agentConsent    *agent.ConsentRegistry

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
	} else if len(servers) > 0 || cfg.ToolSpecURL != "" {
		// The catalogue is always created when the puller is enabled,
		// even if the static MCP_SERVERS is empty: the puller will
		// populate it on first poll and may continue to mutate it as
		// the fleet's tool set changes.
		h.agentCatalog = agent.NewCatalog(servers, &http.Client{Timeout: 10 * time.Second}, 60*time.Second)
		h.agentDispatcher = agent.NewDispatcher(h.agentCatalog, &http.Client{Timeout: 60 * time.Second})
		h.agentConsent = agent.NewConsentRegistry()
		log.Printf("[agent] enabled (static-servers=%d, puller=%v)", len(servers), cfg.ToolSpecURL != "")
	}
	return h
}

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

// RegisterRoutes adds all endpoints to the given mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/chat/completions", h.chatCompletions)
	mux.HandleFunc("POST /v1/completions", h.completions)
	mux.HandleFunc("GET /v1/models", h.models)
	mux.HandleFunc("POST /v1/models", h.models)
	mux.HandleFunc("POST /v1/models/load", h.requireLoadToken(h.modelsLoad))
	mux.HandleFunc("GET /v1/models/status", h.modelsStatus)
	mux.HandleFunc("POST /v1/models/unload", h.requireLoadToken(h.modelsUnload))
	mux.HandleFunc("GET /readiness", h.readiness)
	mux.HandleFunc("GET /health", h.health)
	mux.HandleFunc("POST /health", h.health)
	mux.HandleFunc("GET /healthz", h.health)
	mux.HandleFunc("POST /healthz", h.health)
	mux.HandleFunc("GET /.well-known/attestation-extensions", h.attestationExtensions)
	mux.HandleFunc("GET /metrics", h.metrics)
	mux.HandleFunc("POST /v1/agent/confirm/{id}", h.agentConfirm)
}

// requireLoadToken gates a handler behind the static load token defined in
// config.LoadToken. When LoadToken is empty the handler is reachable
// without authentication (legacy / dev). When set, callers must present
// `Authorization: Bearer <LoadToken>`. The fleet manager / orchestrator
// holds this token; end users never do.
func (h *Handler) requireLoadToken(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.cfg.LoadToken == "" {
			next(w, r)
			return
		}
		authz := r.Header.Get("Authorization")
		if !strings.HasPrefix(authz, "Bearer ") {
			writeError(w, http.StatusUnauthorized, "missing bearer token")
			return
		}
		token := strings.TrimPrefix(authz, "Bearer ")
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
		writeError(w, http.StatusServiceUnavailable, "Model is loading, please wait...")
		return
	}
	h.requestsTotal.Add(1)

	body, err := io.ReadAll(io.LimitReader(r.Body, 10<<20))
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	// Still inject seed=0 default so determinism guarantees hold even
	// on the pass-through path.
	var reqParams requestParams
	if err := json.Unmarshal(body, &reqParams); err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "invalid JSON request body")
		return
	}
	seed := int64(0)
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
		writeError(w, http.StatusServiceUnavailable, "Model is loading, please wait...")
		return
	}
	h.proxyDirect(w, r, "/v1/models")
}

// proxyWithReproducibility reads the request, proxies to vLLM, then
// wraps the response with reproducibility metadata.
// Supports both streaming (SSE) and non-streaming responses.
func (h *Handler) proxyWithReproducibility(w http.ResponseWriter, r *http.Request, path string) {
	if !h.IsReady() {
		writeError(w, http.StatusServiceUnavailable, "Model is loading, please wait...")
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

	// Defaults matching vLLM V1
	if reqParams.Seed == nil {
		defaultSeed := int64(0)
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

	wantRepro := wantsReproducibility(r)
	if reqParams.Stream {
		h.proxyStream(w, resp, meta, wantRepro)
	} else {
		h.proxyNonStream(w, resp, meta, wantRepro)
	}
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
func (h *Handler) proxyNonStream(w http.ResponseWriter, resp *http.Response, meta *reproducibility.Metadata, wantRepro bool) {
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
func (h *Handler) proxyStream(w http.ResponseWriter, resp *http.Response, meta *reproducibility.Metadata, wantRepro bool) {
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

// modelsStatus handles GET /v1/models/status - returns model loading state.
func (h *Handler) modelsStatus(w http.ResponseWriter, _ *http.Request) {
	if h.modelMgr == nil {
		writeError(w, http.StatusNotImplemented, "dynamic model loading not available (legacy mode)")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(h.modelMgr.Status())
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

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
