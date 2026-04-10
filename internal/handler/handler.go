package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/privasys/confidential-ai/internal/config"
	"github.com/privasys/confidential-ai/internal/reproducibility"
)

// Handler is the HTTP handler that proxies to vLLM and injects
// reproducibility metadata into every response.
type Handler struct {
	cfg    *config.Config
	client *http.Client

	// Metrics
	requestsTotal  atomic.Int64
	requestsFailed atomic.Int64
}

// New creates a Handler with the given config.
func New(cfg *config.Config) *Handler {
	return &Handler{
		cfg: cfg,
		client: &http.Client{
			Timeout: 5 * time.Minute, // LLM inference can be slow
		},
	}
}

// RegisterRoutes adds all endpoints to the given mux.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/chat/completions", h.chatCompletions)
	mux.HandleFunc("POST /v1/completions", h.completions)
	mux.HandleFunc("GET /v1/models", h.models)
	mux.HandleFunc("GET /health", h.health)
	mux.HandleFunc("GET /healthz", h.health)
	mux.HandleFunc("GET /metrics", h.metrics)
}

// chatCompletions proxies to vLLM /v1/chat/completions and injects
// reproducibility metadata into the response.
func (h *Handler) chatCompletions(w http.ResponseWriter, r *http.Request) {
	h.proxyWithReproducibility(w, r, "/v1/chat/completions")
}

// completions proxies to vLLM /v1/completions with reproducibility.
func (h *Handler) completions(w http.ResponseWriter, r *http.Request) {
	h.proxyWithReproducibility(w, r, "/v1/completions")
}

// models proxies the /v1/models endpoint directly.
func (h *Handler) models(w http.ResponseWriter, r *http.Request) {
	h.proxyDirect(w, r, "/v1/models")
}

// proxyWithReproducibility reads the request, proxies to vLLM, then
// wraps the response with reproducibility metadata.
// Supports both streaming (SSE) and non-streaming responses.
func (h *Handler) proxyWithReproducibility(w http.ResponseWriter, r *http.Request, path string) {
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

	// Build reproducibility metadata
	meta := reproducibility.NewMetadata(
		*reqParams.Seed,
		reqParams.Temperature,
		reqParams.TopP,
		reqParams.TopK,
		reqParams.MaxTokens,
		h.cfg.ModelName,
		h.cfg.Quantization,
		h.cfg.VLLMVersion,
		h.cfg.CUDAVersion,
		h.cfg.GPUType,
		h.cfg.ImageDigest,
		h.cfg.TeeType,
	)

	if reqParams.Stream {
		h.proxyStream(w, resp, meta)
	} else {
		h.proxyNonStream(w, resp, meta)
	}
}

// proxyNonStream handles non-streaming responses: read full body, inject metadata.
func (h *Handler) proxyNonStream(w http.ResponseWriter, resp *http.Response, meta *reproducibility.Metadata) {
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
// It forwards each chunk as-is, then emits a final reproducibility event
// before the [DONE] sentinel.
func (h *Handler) proxyStream(w http.ResponseWriter, resp *http.Response, meta *reproducibility.Metadata) {
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

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20) // up to 1 MiB per line

	for scanner.Scan() {
		line := scanner.Text()

		// Forward the [DONE] sentinel last, after our metadata event.
		if strings.TrimSpace(line) == "data: [DONE]" {
			// Emit reproducibility metadata as a final SSE event
			metaJSON, err := json.Marshal(map[string]any{"reproducibility": meta})
			if err == nil {
				fmt.Fprintf(w, "data: %s\n\n", metaJSON)
				flusher.Flush()
			}
			// Now send [DONE]
			fmt.Fprintf(w, "%s\n", line)
			flusher.Flush()
			continue
		}

		// Forward every other line verbatim
		fmt.Fprintf(w, "%s\n", line)

		// Flush on data lines (SSE events are terminated by blank lines)
		if line == "" {
			flusher.Flush()
		}
	}

	// If the stream ended without [DONE] (e.g. connection dropped),
	// still try to emit metadata.
	if err := scanner.Err(); err != nil {
		h.requestsFailed.Add(1)
	}
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

// health returns server status and config metadata.
func (h *Handler) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":       "ok",
		"model":        h.cfg.ModelName,
		"quantization": h.cfg.Quantization,
		"gpu":          h.cfg.GPUType,
		"tee":          h.cfg.TeeType,
		"vllm_version": h.cfg.VLLMVersion,
		"image_digest": h.cfg.ImageDigest,
	})
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
