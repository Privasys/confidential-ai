package models

import (
	"slices"
	"testing"
)

// flagValue returns the argument following flag, or "" if absent.
func flagValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func TestBuildVLLMArgs_SingleBlockPrefillDefault(t *testing.T) {
	req := LoadRequest{Model: "qwen36-35b-a3b-fp8", Dtype: "auto", MaxModelLen: 8192, GPUMemoryUtilization: 0.90}
	args := buildVLLMArgs(req, "/models/qwen36-35b-a3b-fp8", 8000)

	if !slices.Contains(args, "--no-enable-chunked-prefill") {
		t.Fatalf("default policy must disable chunked prefill: %v", args)
	}
	if got := flagValue(args, "--max-num-batched-tokens"); got != "16384" {
		t.Fatalf("MNBT floor: got %s, want 16384", got)
	}
}

func TestBuildVLLMArgs_SingleBlockClampsMNBTToModelLen(t *testing.T) {
	// vLLM rejects MNBT < max_model_len when chunked prefill is off;
	// the builder must clamp up even if the caller passed less.
	req := LoadRequest{Model: "m", Dtype: "auto", MaxModelLen: 200000, GPUMemoryUtilization: 0.93, MaxNumBatchedTokens: 32768}
	args := buildVLLMArgs(req, "/models/m", 8000)

	if got := flagValue(args, "--max-num-batched-tokens"); got != "200000" {
		t.Fatalf("MNBT must clamp to max_model_len in single-block mode: got %s", got)
	}
}

func TestBuildVLLMArgs_ChunkedPrefill(t *testing.T) {
	req := LoadRequest{Model: "m", Dtype: "auto", MaxModelLen: 262144, GPUMemoryUtilization: 0.92, EnableChunkedPrefill: true}
	args := buildVLLMArgs(req, "/models/m", 8000)

	if !slices.Contains(args, "--enable-chunked-prefill") {
		t.Fatalf("expected --enable-chunked-prefill: %v", args)
	}
	if slices.Contains(args, "--no-enable-chunked-prefill") {
		t.Fatalf("both prefill flags present: %v", args)
	}
	// Bounded default decoupled from max_model_len.
	if got := flagValue(args, "--max-num-batched-tokens"); got != "32768" {
		t.Fatalf("chunked MNBT default: got %s, want 32768", got)
	}

	req.MaxNumBatchedTokens = 16384
	args = buildVLLMArgs(req, "/models/m", 8000)
	if got := flagValue(args, "--max-num-batched-tokens"); got != "16384" {
		t.Fatalf("explicit MNBT must win under chunked prefill: got %s", got)
	}
}

func TestBuildVLLMArgs_MemoryTuningFlags(t *testing.T) {
	req := LoadRequest{
		Model: "m", Dtype: "auto", MaxModelLen: 262144, GPUMemoryUtilization: 0.92,
		EnableChunkedPrefill:    true,
		MaxNumSeqs:              64,
		KVCacheDtype:            "fp8",
		MambaSSMCacheDtype:      "bfloat16",
		MaxCudagraphCaptureSize: 64,
	}
	args := buildVLLMArgs(req, "/models/m", 8000)

	for flag, want := range map[string]string{
		"--max-num-seqs":              "64",
		"--kv-cache-dtype":            "fp8",
		"--mamba-ssm-cache-dtype":     "bfloat16",
		"--max-cudagraph-capture-size": "64",
	} {
		if got := flagValue(args, flag); got != want {
			t.Errorf("%s: got %q, want %q", flag, got, want)
		}
	}
}

func TestBuildVLLMArgs_OmitsTuningFlagsByDefault(t *testing.T) {
	req := LoadRequest{Model: "m", Dtype: "auto", MaxModelLen: 8192, GPUMemoryUtilization: 0.90}
	args := buildVLLMArgs(req, "/models/m", 8000)

	for _, flag := range []string{"--kv-cache-dtype", "--mamba-ssm-cache-dtype", "--max-cudagraph-capture-size", "--max-num-seqs"} {
		if slices.Contains(args, flag) {
			t.Errorf("unset field must not emit %s: %v", flag, args)
		}
	}
}

func TestBuildVLLMArgs_ServedNameBasename(t *testing.T) {
	req := LoadRequest{Model: "/models/qwen36-35b-a3b-fp8", Dtype: "auto", MaxModelLen: 8192, GPUMemoryUtilization: 0.90}
	args := buildVLLMArgs(req, "/models/qwen36-35b-a3b-fp8", 8000)

	if got := flagValue(args, "--served-model-name"); got != "qwen36-35b-a3b-fp8" {
		t.Fatalf("served name must strip path: got %q", got)
	}
}
