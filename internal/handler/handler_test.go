package handler

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/privasys/confidential-ai/internal/config"
)

func TestHealthEndpoint(t *testing.T) {
	h := New(&config.Config{
		ModelName:    "test-model",
		Quantization: "awq",
		GPUType:      "H100-80GB",
		TeeType:      "tdx",
		VLLMVersion:  "0.19.1",
		CUDAVersion:  "12.6",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/health", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "ok" {
		t.Fatalf("expected status ok, got %v", resp["status"])
	}
	if resp["model"] != "test-model" {
		t.Fatalf("expected model test-model, got %v", resp["model"])
	}
	if resp["tee"] != "tdx" {
		t.Fatalf("expected tee tdx, got %v", resp["tee"])
	}
}

func TestChatCompletionsInjectsReproducibility(t *testing.T) {
	// Mock vLLM backend
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		// Verify seed was injected
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["seed"] == nil {
			t.Error("seed was not injected into upstream request")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"choices": []map[string]any{{"message": map[string]string{"content": "hello"}}},
		})
	}))
	defer vllm.Close()

	h := New(&config.Config{
		VLLMUpstream: vllm.URL,
		ModelName:    "gpt-oss-120b",
		Quantization: "awq",
		GPUType:      "H100-80GB",
		TeeType:      "tdx",
		VLLMVersion:  "0.19.1",
		CUDAVersion:  "12.6",
		ImageDigest:  "sha256:abc123",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := `{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	repro, ok := resp["reproducibility"].(map[string]any)
	if !ok {
		t.Fatal("missing reproducibility metadata in response")
	}

	checks := map[string]any{
		"model":          "gpt-oss-120b",
		"quantization":   "awq",
		"gpu":            "H100-80GB",
		"tee_type":       "tdx",
		"vllm_version":   "0.19.1",
		"cuda_version":   "12.6",
		"image_digest":   "sha256:abc123",
	}
	for k, want := range checks {
		if got := repro[k]; got != want {
			t.Errorf("reproducibility[%q] = %v, want %v", k, got, want)
		}
	}

	// seed should be 0 (default)
	if seed, ok := repro["seed"].(float64); !ok || seed != 0 {
		t.Errorf("expected seed 0, got %v", repro["seed"])
	}

	// batch_invariance should be true
	if bi, ok := repro["batch_invariance"].(bool); !ok || !bi {
		t.Errorf("expected batch_invariance true, got %v", repro["batch_invariance"])
	}

	// tensor_parallel_size should be 1
	if tp, ok := repro["tensor_parallel_size"].(float64); !ok || tp != 1 {
		t.Errorf("expected tensor_parallel_size 1, got %v", repro["tensor_parallel_size"])
	}
}

func TestMetricsEndpoint(t *testing.T) {
	h := New(&config.Config{}, nil)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/metrics", nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "confidential_ai_requests_total") {
		t.Fatal("metrics response missing expected metric")
	}
}

func TestCompletionsEndpoint(t *testing.T) {
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "cmpl-test",
			"object":  "text_completion",
			"choices": []map[string]any{{"text": "world"}},
		})
	}))
	defer vllm.Close()

	h := New(&config.Config{
		VLLMUpstream: vllm.URL,
		ModelName:    "gpt-oss-120b",
		Quantization: "awq",
		GPUType:      "H100-80GB",
		TeeType:      "tdx",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := `{"prompt":"hello","temperature":0.5,"seed":42}`
	req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp map[string]any
	json.Unmarshal(rec.Body.Bytes(), &resp)
	repro := resp["reproducibility"].(map[string]any)

	// Should preserve the user-provided seed=42
	if seed := repro["seed"].(float64); seed != 42 {
		t.Errorf("expected seed 42, got %v", seed)
	}
	if temp := repro["temperature"].(float64); temp != 0.5 {
		t.Errorf("expected temperature 0.5, got %v", temp)
	}
}

func TestVLLMUpstreamError(t *testing.T) {
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"model not loaded"}`))
	}))
	defer vllm.Close()

	h := New(&config.Config{
		VLLMUpstream: vllm.URL,
		ModelName:    "test-model",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	// Should pass through the 500 from vLLM
	if rec.Code != 500 {
		t.Fatalf("expected 500, got %d", rec.Code)
	}
}

func TestStreamingChatCompletions(t *testing.T) {
	// Mock vLLM streaming backend (SSE)
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["seed"] == nil {
			t.Error("seed was not injected into streaming request")
		}
		if req["stream"] != true {
			t.Error("stream field should be true")
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("test server does not support flushing")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Simulate vLLM SSE: two content chunks + [DONE]
		chunks := []string{
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hel"},"index":0}]}`,
			`{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"lo"},"index":0}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer vllm.Close()

	h := New(&config.Config{
		VLLMUpstream: vllm.URL,
		ModelName:    "test-model",
		Quantization: "none",
		GPUType:      "H100-80GB",
		TeeType:      "tdx",
		VLLMVersion:  "0.19.1",
		CUDAVersion:  "13.0",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := `{"messages":[{"role":"user","content":"hi"}],"temperature":0.7,"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %s", ct)
	}

	// Parse SSE events
	scanner := bufio.NewScanner(rec.Body)
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		}
	}

	// Should have: chunk1, chunk2, reproducibility, [DONE]
	if len(dataLines) < 4 {
		t.Fatalf("expected at least 4 data lines, got %d: %v", len(dataLines), dataLines)
	}

	// Second-to-last should be reproducibility metadata
	reproLine := dataLines[len(dataLines)-2]
	var reproEvent map[string]any
	if err := json.Unmarshal([]byte(reproLine), &reproEvent); err != nil {
		t.Fatalf("failed to parse reproducibility event: %v (line: %s)", err, reproLine)
	}

	repro, ok := reproEvent["reproducibility"].(map[string]any)
	if !ok {
		t.Fatal("missing reproducibility key in metadata event")
	}

	if repro["model"] != "test-model" {
		t.Errorf("expected model test-model, got %v", repro["model"])
	}
	if repro["tee_type"] != "tdx" {
		t.Errorf("expected tee_type tdx, got %v", repro["tee_type"])
	}
	if seed, ok := repro["seed"].(float64); !ok || seed != 0 {
		t.Errorf("expected seed 0, got %v", repro["seed"])
	}

	// Last should be [DONE]
	lastLine := dataLines[len(dataLines)-1]
	if lastLine != "[DONE]" {
		t.Errorf("expected last data line to be [DONE], got %s", lastLine)
	}
}

func TestStreamingErrorPassthrough(t *testing.T) {
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"model not loaded"}`))
	}))
	defer vllm.Close()

	h := New(&config.Config{
		VLLMUpstream: vllm.URL,
		ModelName:    "test-model",
	}, nil)
	h.ready.Store(1)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := `{"messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 500 {
		t.Fatalf("expected 500, got %d", rec.Code)
	}
}
