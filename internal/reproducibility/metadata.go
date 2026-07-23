package reproducibility

import (
	"time"

	"github.com/google/uuid"
)

// Metadata captures all parameters needed for exact inference reproduction.
type Metadata struct {
	RequestID          string  `json:"request_id"`
	Seed               int64   `json:"seed"`
	Temperature        float64 `json:"temperature"`
	TopP               float64 `json:"top_p"`
	TopK               int     `json:"top_k,omitempty"`
	MaxTokens          int     `json:"max_tokens,omitempty"`
	Model              string  `json:"model"`
	Quantization       string  `json:"quantization"`
	VLLMVersion        string  `json:"vllm_version"`
	CUDAVersion        string  `json:"cuda_version"`
	GPU                string  `json:"gpu"`
	TensorParallelSize int     `json:"tensor_parallel_size"`
	BatchInvariance    bool    `json:"batch_invariance"`
	ImageDigest        string  `json:"image_digest,omitempty"`
	TeeType            string  `json:"tee_type"`
	Timestamp          string  `json:"timestamp"`

	// DynamicContext is the per-request context block the proxy injected at
	// the tail of the last user message (currently the wall-clock time). It
	// is recorded verbatim so a replay can reconstruct the exact prompt:
	// re-issue the request with the X-Privasys-Dynamic-Context header set to
	// this value (and the recorded seed) to reproduce the response
	// token-for-token. Empty when nothing was injected.
	DynamicContext string `json:"dynamic_context,omitempty"`

	// KVCacheMode records how the vLLM prefix (KV) cache was scoped for
	// this request: "session" (caller-partitioned cache_salt — KV blocks
	// may be reused across this caller's own requests) or "strict"
	// (single-use salt — zero cache reuse, the full prompt was freshly
	// prefilled). A replay is always cache-cold regardless of the serving
	// mode.
	KVCacheMode string `json:"kv_cache_mode,omitempty"`

	// CachedTokens is the number of prompt tokens vLLM served from the
	// prefix cache (usage.prompt_tokens_details.cached_tokens). 0 in
	// strict mode by construction. When 0, a serialized replay must match
	// token-for-token; when >0, cached KV blocks computed under a
	// different batch composition were reused, so a replay is expected to
	// match but can diverge at logit near-ties until kernel-level batch
	// invariance ships. Nil when the upstream did not report the detail.
	CachedTokens *int64 `json:"cached_tokens,omitempty"`

	// ToolCalls, when non-nil, lists the MCP tool invocations that
	// served this response (populated by the agentic loop). Each entry
	// is a compact descriptor: {name, status, duration_ms, error?}.
	// Unset for non-agentic completions.
	ToolCalls []ToolCallSummary `json:"tool_calls,omitempty"`
}

// ToolCallSummary is the per-tool entry surfaced in the reproducibility
// block so a verifier can replay the conversation and check both the
// model AND the retrieval boundary.
type ToolCallSummary struct {
	Name       string `json:"name"`
	Status     string `json:"status"`
	DurationMs int64  `json:"duration_ms"`
	Error      string `json:"error,omitempty"`
}

// PoolingMetadata is the compact reproducibility block for the pooling
// endpoints (/v1/embeddings, /v1/rerank). Pooling inference has no
// sampling parameters — the attested facts are WHICH model produced the
// vectors/scores (digest = the dm-verity root hash, OID-3.5-style) and
// the serving stack it ran on. Drive records this alongside its index
// rows so a reindex-triggering model change is detectable.
type PoolingMetadata struct {
	RequestID   string `json:"request_id"`
	Task        string `json:"task"`
	Model       string `json:"model"`
	ModelDigest string `json:"model_digest,omitempty"`
	VLLMVersion string `json:"vllm_version"`
	CUDAVersion string `json:"cuda_version"`
	GPU         string `json:"gpu"`
	ImageDigest string `json:"image_digest,omitempty"`
	TeeType     string `json:"tee_type"`
	Timestamp   string `json:"timestamp"`
}

// NewPoolingMetadata creates the reproducibility block for an
// embeddings/rerank response.
func NewPoolingMetadata(task, model, modelDigest, vllmVersion, cudaVersion, gpu, imageDigest, teeType string) *PoolingMetadata {
	return &PoolingMetadata{
		RequestID:   uuid.New().String(),
		Task:        task,
		Model:       model,
		ModelDigest: modelDigest,
		VLLMVersion: vllmVersion,
		CUDAVersion: cudaVersion,
		GPU:         gpu,
		ImageDigest: imageDigest,
		TeeType:     teeType,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
	}
}

// NewMetadata creates reproducibility metadata from request parameters
// and system configuration.
func NewMetadata(
	seed int64,
	temperature float64,
	topP float64,
	topK int,
	maxTokens int,
	model string,
	quantization string,
	vllmVersion string,
	cudaVersion string,
	gpu string,
	imageDigest string,
	teeType string,
) *Metadata {
	return &Metadata{
		RequestID:          uuid.New().String(),
		Seed:               seed,
		Temperature:        temperature,
		TopP:               topP,
		TopK:               topK,
		MaxTokens:          maxTokens,
		Model:              model,
		Quantization:       quantization,
		VLLMVersion:        vllmVersion,
		CUDAVersion:        cudaVersion,
		GPU:                gpu,
		TensorParallelSize: 1,
		// Honest value: kernel-level batch invariance is NOT enabled (we
		// run stock vLLM kernels; the upstream VLLM_BATCH_INVARIANT work
		// is tracked separately). What we guarantee is serialized replay
		// determinism from the recorded seed + prompt, not bitwise
		// equality under concurrent batching. This field was previously
		// hardcoded true, which overstated the contract.
		BatchInvariance: false,
		ImageDigest:        imageDigest,
		TeeType:            teeType,
		Timestamp:          time.Now().UTC().Format(time.RFC3339),
	}
}
