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
		BatchInvariance:    true,
		ImageDigest:        imageDigest,
		TeeType:            teeType,
		Timestamp:          time.Now().UTC().Format(time.RFC3339),
	}
}
