package config

import (
	"flag"
	"os"
	"time"
)

// Config holds all server configuration.
type Config struct {
	Listen       string // HTTP listen address
	VLLMUpstream string // vLLM backend URL
	ModelsDir    string // directory containing model subdirectories (e.g. /models)
	VLLMPort     int    // port for vLLM subprocess (default 8000)
	ModelName    string // model identifier for metadata (legacy, dynamic in new mode)
	ModelDigest  string // SHA-256 of model weights index (legacy, dynamic in new mode)
	Quantization string // quantization method (awq, gptq, fp8, etc.)
	GPUType      string // GPU hardware identifier
	ImageDigest  string // CVM image SHA256 digest
	CUDAVersion  string // CUDA version string
	VLLMVersion  string // vLLM version string
	TeeType      string // TEE type: tdx, sev-snp

	// RoothashDir is the directory written by disk-mounter on the host
	// containing per-model dm-verity root hashes (one file per model,
	// named <model>.roothash). When a load request resolves to a model
	// whose roothash file is present, the proxy publishes that root
	// hash as the OID 3.5 (MODEL_DIGEST) attestation extension instead
	// of hashing the safetensors index.
	RoothashDir string

	// StateFile is the path on the per-container encrypted volume where
	// the most recently successful /v1/models/load request is
	// persisted. On container start the proxy reads this file and
	// auto-issues the same Load so a restart (typically after a Spot-VM
	// reboot) recovers the previously-served model without manual
	// orchestration. Empty disables the feature.
	StateFile string

	// LoadToken, when non-empty, is required as Bearer credential on
	// POST /v1/models/load and POST /v1/models/unload. Issued to the
	// fleet manager / orchestrator only. When empty, the endpoints
	// remain open (legacy / dev mode).
	LoadToken string

	// MCPServers, when non-empty, enables the agentic tool-call loop
	// on POST /v1/chat/completions. Format (env MCP_SERVERS):
	//
	//   name1=https://url1[?bearer=1],name2=https://url2,...
	//
	// `bearer=1` opts the server in to receiving the user's
	// Authorization header from the original chat request (required
	// for private-rag, optional for stateless tools like lightpanda).
	// When MCPServers is empty the proxy behaves exactly as before
	// (pure pass-through to vLLM).
	MCPServers string

	// ToolSpecURL, when non-empty, enables the background tool-spec
	// puller. The proxy polls this URL every ToolSpecInterval and
	// atomically replaces the agent.Catalog's server list with the
	// returned spec string (see internal/agent/spec.go::ParseServerSpec
	// for format). This is how managed instances (e.g. confidential-ai
	// running inside an enclave alongside the workload manager) pick
	// up fleet-level tool-set changes without a container restart.
	//
	// The endpoint must return JSON: {"spec":"...","generation":"..."}.
	// When ToolSpecURL is empty the puller is disabled and the tool
	// catalogue is fixed to whatever MCPServers resolved to at startup.
	ToolSpecURL string

	// ToolSpecToken, when non-empty, is sent as `Authorization: Bearer
	// <token>` on every ToolSpecURL poll. Typically a static
	// machine-to-machine credential issued to the enclave by the
	// management service.
	ToolSpecToken string

	// ToolSpecInterval is the polling cadence for ToolSpecURL.
	// Defaults to 60s when zero. Ignored when ToolSpecURL is empty.
	ToolSpecInterval time.Duration

	// CORSOrigins is a comma-separated allowlist of HTTP Origins that
	// receive Access-Control-Allow-* response headers. Defaults to the
	// Privasys chat front-ends. Empty disables CORS entirely (browser
	// requests from any origin will be blocked by the SOP).
	CORSOrigins string
}

// Parse reads configuration from flags and environment, returning it.
func Parse(args []string) (*Config, error) {
	fs := flag.NewFlagSet("confidential-ai", flag.ContinueOnError)

	cfg := &Config{}

	fs.StringVar(&cfg.Listen, "listen", envOr("LISTEN_ADDR", ":8080"),
		"HTTP listen address (env: LISTEN_ADDR)")
	fs.StringVar(&cfg.VLLMUpstream, "vllm-upstream", envOr("VLLM_UPSTREAM", "http://localhost:8000"),
		"vLLM backend URL (env: VLLM_UPSTREAM)")
	fs.StringVar(&cfg.ModelsDir, "models-dir", envOr("MODELS_DIR", "/models"),
		"Directory containing model subdirectories (env: MODELS_DIR)")
	fs.IntVar(&cfg.VLLMPort, "vllm-port", 8000,
		"Port for vLLM subprocess to listen on")
	fs.StringVar(&cfg.ModelName, "model", envOr("MODEL_NAME", ""),
		"Model name for reproducibility metadata (env: MODEL_NAME)")
	fs.StringVar(&cfg.ModelDigest, "model-digest", envOr("MODEL_DIGEST", ""),
		"SHA-256 of model weights index for attestation (env: MODEL_DIGEST)")
	fs.StringVar(&cfg.Quantization, "quantization", envOr("QUANTIZATION", ""),
		"Quantization method (env: QUANTIZATION)")
	fs.StringVar(&cfg.GPUType, "gpu-type", envOr("GPU_TYPE", "H100-80GB"),
		"GPU hardware type (env: GPU_TYPE)")
	fs.StringVar(&cfg.ImageDigest, "image-digest", envOr("IMAGE_DIGEST", ""),
		"CVM image SHA256 digest (env: IMAGE_DIGEST)")
	fs.StringVar(&cfg.CUDAVersion, "cuda-version", envOr("CUDA_VERSION", "13.0"),
		"CUDA version (env: CUDA_VERSION)")
	fs.StringVar(&cfg.VLLMVersion, "vllm-version", envOr("VLLM_VERSION", "0.19.1"),
		"vLLM version (env: VLLM_VERSION)")
	fs.StringVar(&cfg.TeeType, "tee-type", envOr("TEE_TYPE", "tdx"),
		"TEE type: tdx or sev-snp (env: TEE_TYPE)")
	fs.StringVar(&cfg.RoothashDir, "roothash-dir", envOr("ROOTHASH_DIR", "/var/lib/enclave-os/model-roothashes"),
		"Directory of per-model dm-verity root hashes (env: ROOTHASH_DIR)")
	fs.StringVar(&cfg.StateFile, "state-file", envOr("STATE_FILE", "/data/last-load.json"),
		"Path where the last successful Load request is persisted for auto-restore on restart (env: STATE_FILE; empty disables)")
	fs.StringVar(&cfg.LoadToken, "load-token", envOr("LOAD_TOKEN", ""),
		"Bearer token required on /v1/models/{load,unload}; empty disables auth (env: LOAD_TOKEN)")
	fs.StringVar(&cfg.MCPServers, "mcp-servers", envOr("MCP_SERVERS", ""),
		"Comma-separated <name>=<url>[?bearer=1] list of MCP servers to expose as tools (env: MCP_SERVERS)")
	fs.StringVar(&cfg.ToolSpecURL, "tool-spec-url", envOr("TOOL_SPEC_URL", ""),
		"When set, the proxy polls this URL for {spec,generation} and hot-reloads the tool catalogue (env: TOOL_SPEC_URL)")
	fs.StringVar(&cfg.ToolSpecToken, "tool-spec-token", envOr("TOOL_SPEC_TOKEN", ""),
		"Bearer token sent on every tool-spec-url poll (env: TOOL_SPEC_TOKEN)")
	fs.DurationVar(&cfg.ToolSpecInterval, "tool-spec-interval", envDuration("TOOL_SPEC_INTERVAL", 60*time.Second),
		"How often to poll tool-spec-url (env: TOOL_SPEC_INTERVAL, e.g. 30s)")
	fs.StringVar(&cfg.CORSOrigins, "cors-origins", envOr("CORS_ORIGINS", "https://chat.privasys.org,https://chat-test.privasys.org,http://localhost:4210,http://localhost:3000"),
		"Comma-separated CORS Origin allowlist (env: CORS_ORIGINS)")

	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}
	return d
}
