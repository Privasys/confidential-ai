package config

import (
	"flag"
	"os"
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
	fs.StringVar(&cfg.VLLMVersion, "vllm-version", envOr("VLLM_VERSION", "0.19.0"),
		"vLLM version (env: VLLM_VERSION)")
	fs.StringVar(&cfg.TeeType, "tee-type", envOr("TEE_TYPE", "tdx"),
		"TEE type: tdx or sev-snp (env: TEE_TYPE)")

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
