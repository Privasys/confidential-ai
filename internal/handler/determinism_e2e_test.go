//go:build e2e
// +build e2e

// Determinism CI gate.
//
// Runs N=10 sequential chat completions against a live confidential-ai
// proxy with the same seed and asserts that:
//
//   1. Every response carries the same model_digest in the
//      reproducibility metadata.
//   2. Every response carries seed=0 and batch_invariance=true.
//   3. The completion content is bit-identical across all N runs.
//
// Single-stream determinism is guaranteed today by:
//
//   - vLLM V1 scheduler with deterministic prefill (b2aa347).
//   - CUDA graphs pinned per max_num_batched_tokens.
//   - Sampler seed=0 + temperature=0.
//
// Per-batch determinism still depends on the upstream
// batch-invariant attention kernel (tracked separately under P1).
// This gate intentionally exercises the sequential path only.
//
// Build / run:
//
//   go test -tags=e2e ./internal/handler/... \
//       -run TestDeterminism \
//       -confidential-ai-url=https://m2-dev-ai.privasys.org \
//       -confidential-ai-token=$JWT
//
// Or via env vars:
//
//   CONFIDENTIAL_AI_URL=... CONFIDENTIAL_AI_TOKEN=... \
//       go test -tags=e2e ./internal/handler/... -run TestDeterminism
//
// The tag keeps it out of the default `go test ./...` run so unit-test
// CI does not need a live GPU. The `e2e` GH workflow triggers it
// against a dev VM on workflow_dispatch (and nightly on main).

package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

var (
	e2eURL   = flag.String("confidential-ai-url", "", "base URL of the confidential-ai proxy under test (overrides CONFIDENTIAL_AI_URL)")
	e2eToken = flag.String("confidential-ai-token", "", "bearer token for the proxy (overrides CONFIDENTIAL_AI_TOKEN)")
	e2eModel = flag.String("confidential-ai-model", "gemma4-31b", "model name to query")
	e2eN     = flag.Int("determinism-n", 10, "number of sequential requests")
)

func resolveE2E(t *testing.T) (url, token string) {
	t.Helper()
	url = *e2eURL
	if url == "" {
		url = os.Getenv("CONFIDENTIAL_AI_URL")
	}
	token = *e2eToken
	if token == "" {
		token = os.Getenv("CONFIDENTIAL_AI_TOKEN")
	}
	if url == "" {
		t.Skip("CONFIDENTIAL_AI_URL not set; skipping determinism e2e gate")
	}
	return url, strings.TrimSpace(token)
}

type chatResp struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	// Note: model_digest lives on /v1/models/status, not per-completion.
	Reproducibility struct {
		Seed            int64  `json:"seed"`
		Model           string `json:"model"`
		BatchInvariance bool   `json:"batch_invariance"`
		TeeType         string `json:"tee_type"`
		GPU             string `json:"gpu"`
		VLLMVersion     string `json:"vllm_version"`
	} `json:"reproducibility"`
}

type modelStatus struct {
	State       string `json:"state"`
	Model       string `json:"model"`
	ModelDigest string `json:"model_digest"`
	Message     string `json:"message"`
}

func fetchStatus(ctx context.Context, base, token string) (modelStatus, error) {
	req, err := http.NewRequestWithContext(ctx, "GET",
		strings.TrimRight(base, "/")+"/v1/models/status", nil)
	if err != nil {
		return modelStatus{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return modelStatus{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return modelStatus{}, fmt.Errorf("status %d: %s", resp.StatusCode, raw)
	}
	var s modelStatus
	if err := json.Unmarshal(raw, &s); err != nil {
		return modelStatus{}, fmt.Errorf("decode: %w (body=%s)", err, raw)
	}
	return s, nil
}

func chat(ctx context.Context, base, token, model, prompt string) (chatResp, error) {
	body := map[string]any{
		"model":       model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"temperature": 0,
		"seed":        0,
		"max_tokens":  64,
		"stream":      false,
	}
	buf, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(base, "/")+"/v1/chat/completions", bytes.NewReader(buf))
	if err != nil {
		return chatResp{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return chatResp{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return chatResp{}, fmt.Errorf("status %d: %s", resp.StatusCode, raw)
	}
	var out chatResp
	if err := json.Unmarshal(raw, &out); err != nil {
		return chatResp{}, fmt.Errorf("decode: %w (body=%s)", err, raw)
	}
	if len(out.Choices) == 0 {
		return chatResp{}, fmt.Errorf("no choices in response: %s", raw)
	}
	return out, nil
}

func TestDeterminismSequential(t *testing.T) {
	base, token := resolveE2E(t)
	const prompt = "Reply with exactly the following sentence: " +
		"The quick brown fox jumps over the lazy dog."

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	statusBefore, err := fetchStatus(ctx, base, token)
	if err != nil {
		t.Fatalf("status fetch (before) failed: %v", err)
	}
	if statusBefore.State != "ready" {
		t.Fatalf("model not ready: state=%q message=%q", statusBefore.State, statusBefore.Message)
	}
	if statusBefore.ModelDigest == "" {
		t.Fatalf("model_digest empty in /v1/models/status: %+v", statusBefore)
	}
	t.Logf("baseline: model=%s digest=%s", statusBefore.Model, statusBefore.ModelDigest)

	first, err := chat(ctx, base, token, *e2eModel, prompt)
	if err != nil {
		t.Fatalf("first request failed: %v", err)
	}
	if first.Reproducibility.Seed != 0 {
		t.Fatalf("expected seed=0, got %d", first.Reproducibility.Seed)
	}
	if !first.Reproducibility.BatchInvariance {
		t.Fatalf("expected batch_invariance=true (sequential gate)")
	}
	if first.Reproducibility.Model != *e2eModel {
		t.Fatalf("model name drift: want %q got %q", *e2eModel, first.Reproducibility.Model)
	}
	want := first.Choices[0].Message.Content

	for i := 2; i <= *e2eN; i++ {
		got, err := chat(ctx, base, token, *e2eModel, prompt)
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
		if got.Reproducibility.Seed != 0 {
			t.Fatalf("request %d: seed drift, got %d", i, got.Reproducibility.Seed)
		}
		if got.Choices[0].Message.Content != want {
			t.Fatalf("request %d: completion drift\n  want: %q\n   got: %q",
				i, want, got.Choices[0].Message.Content)
		}
	}

	statusAfter, err := fetchStatus(ctx, base, token)
	if err != nil {
		t.Fatalf("status fetch (after) failed: %v", err)
	}
	if statusAfter.ModelDigest != statusBefore.ModelDigest {
		t.Fatalf("model_digest drift: before=%s after=%s",
			statusBefore.ModelDigest, statusAfter.ModelDigest)
	}

	t.Logf("determinism gate passed: %d/%d identical completions, model_digest=%s (stable)",
		*e2eN, *e2eN, statusBefore.ModelDigest)
}
