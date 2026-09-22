// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/privasys/confidential-ai/internal/config"
)

// messagesHandler returns a handler wired to a mock vLLM that records what the
// Messages route forwarded upstream.
func messagesHandler(t *testing.T, reply func(w http.ResponseWriter, r *http.Request)) (*http.ServeMux, *Handler, *map[string]any) {
	t.Helper()
	forwarded := map[string]any{}
	vllm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		forwarded["path"] = r.URL.Path
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		for k, v := range req {
			forwarded[k] = v
		}
		reply(w, r)
	}))
	t.Cleanup(vllm.Close)

	h := New(&config.Config{
		VLLMUpstream: vllm.URL,
		ModelName:    "qwen36-35b-a3b-fp8",
		Quantization: "fp8",
		GPUType:      "H100-80GB",
		TeeType:      "tdx",
		VLLMVersion:  "0.29.0",
		CUDAVersion:  "12.6",
		ImageDigest:  "sha256:abc123",
	}, nil)
	h.ready.Store(1)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)
	return mux, h, &forwarded
}

// A Messages request reaches vLLM's own Anthropic route, carries the pinned
// seed our reproducibility contract needs, and comes back with the block.
func TestMessagesIsReproducibleLikeChat(t *testing.T) {
	mux, h, forwarded := messagesHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_test", "type": "message", "role": "assistant",
			"content": []map[string]any{{"type": "text", "text": "hello"}},
			"usage":   map[string]any{"input_tokens": 11, "output_tokens": 3},
		})
	})

	body := `{"model":"qwen36-35b-a3b-fp8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Privasys-Reproducibility", "1")
	req.Header.Set("X-App-Auth", authInference(t, h))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if (*forwarded)["path"] != "/v1/messages" {
		t.Fatalf("forwarded to %v, want vLLM's own Anthropic route", (*forwarded)["path"])
	}
	// The seed is what makes a reply replayable; the Anthropic wire has no
	// seed of its own, so the deployment's vLLM patch carries ours.
	if (*forwarded)["seed"] == nil {
		t.Error("no seed was injected into the Messages request")
	}
	if (*forwarded)["temperature"] == nil || (*forwarded)["top_p"] == nil {
		t.Error("the effective sampling must be explicit on this wire too")
	}
	if (*forwarded)["cache_salt"] == nil {
		t.Error("the prefix cache must be salted on this wire too")
	}
	// A streamless request must not be given the chat wire's opt-in.
	if _, present := (*forwarded)["stream_options"]; present {
		t.Error("stream_options belongs to the chat wire only")
	}

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if _, ok := resp["reproducibility"].(map[string]any); !ok {
		t.Fatalf("missing reproducibility block: %s", rec.Body.String())
	}
	if resp["type"] != "message" {
		t.Errorf("the Anthropic response shape must survive: %v", resp["type"])
	}
}

// The reproducibility block is not volunteered to a caller who did not ask.
func TestMessagesOmitsReproducibilityWithoutHeader(t *testing.T) {
	mux, h, _ := messagesHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_test", "type": "message",
			"usage": map[string]any{"input_tokens": 5, "output_tokens": 1},
		})
	})
	req := httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-App-Auth", authInference(t, h))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	var resp map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)
	if _, present := resp["reproducibility"]; present {
		t.Error("the block must be opt-in on this wire too")
	}
}

// An unauthenticated caller is refused, like on every inference route.
func TestMessagesRequiresAuth(t *testing.T) {
	mux, _, _ := messagesHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req := httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":8,"messages":[]}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", rec.Code)
	}
}

// A streamed Messages response keeps every event of the Anthropic protocol,
// and the reproducibility frame lands before message_stop.
func TestMessagesStreamKeepsItsEventsAndEndsWithTheBlock(t *testing.T) {
	events := strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":10,\"output_tokens\":4}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, "")
	mux, h, forwarded := messagesHandler(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(events))
	})

	req := httptest.NewRequest("POST", "/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("X-Privasys-Reproducibility", "1")
	req.Header.Set("X-App-Auth", authInference(t, h))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if _, present := (*forwarded)["stream_options"]; present {
		t.Error("vLLM's Anthropic converter asks for usage itself; we must not")
	}
	out := rec.Body.String()
	for _, want := range []string{"message_start", "content_block_delta", "message_delta", "message_stop"} {
		if !strings.Contains(out, want) {
			t.Errorf("the %s event was dropped from the stream", want)
		}
	}
	repro := strings.Index(out, "reproducibility")
	stop := strings.Index(out, "message_stop")
	if repro < 0 {
		t.Fatalf("no reproducibility frame in the stream: %s", out)
	}
	if repro > stop {
		t.Error("the reproducibility frame must precede message_stop")
	}
}

// The billable prompt total on the Messages wire includes the cached tokens,
// which Anthropic reports beside input_tokens rather than inside it. Billing
// the same request the same on either wire depends on this.
func TestMessagesUsageCountsCachedPromptTokens(t *testing.T) {
	body := []byte(`{"id":"msg_9","usage":{"input_tokens":30,"output_tokens":7,"cache_read_input_tokens":100,"cache_creation_input_tokens":20}}`)
	id, in, out, ok := messagesUsage(body)
	if !ok || id != "msg_9" || out != 7 {
		t.Fatalf("usage: id=%q in=%d out=%d ok=%v", id, in, out, ok)
	}
	if in != 150 {
		t.Errorf("prompt total = %d, want 30+100+20: a cached prompt must not be billed as a short one", in)
	}
	if c := messagesCachedTokens(body); c == nil || *c != 100 {
		t.Errorf("cached tokens = %v, want 100", c)
	}
}

// Streamed totals arrive across two events and are complete only at the end.
func TestMessagesStreamUsageFoldsBothEvents(t *testing.T) {
	var u messagesStreamUsage
	u.observe([]byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"usage\":{\"input_tokens\":12,\"output_tokens\":0,\"cache_read_input_tokens\":4}}}\n\n"))
	if _, in, out, _ := u.total(); in != 16 || out != 0 {
		t.Fatalf("after message_start: in=%d out=%d, want the prompt with its cache read", in, out)
	}
	u.observe([]byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":12,\"output_tokens\":9,\"cache_read_input_tokens\":4}}\n\n"))
	id, in, out, ok := u.total()
	if !ok || id != "msg_2" || in != 16 || out != 9 {
		t.Fatalf("after message_delta: id=%q in=%d out=%d ok=%v", id, in, out, ok)
	}
	if u.cached == nil || *u.cached != 4 {
		t.Errorf("cached = %v, want 4", u.cached)
	}
}

// Each wire knows its own end of stream, and neither answers to the other's.
func TestEachWireKnowsItsTerminalEvent(t *testing.T) {
	done := []byte("data: [DONE]\n\n")
	stop := []byte("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	delta := []byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n")

	if !wireChat.isTerminalEvent(done) || wireChat.isTerminalEvent(stop) {
		t.Error("the chat wire ends at [DONE] and nothing else")
	}
	if !wireMessages.isTerminalEvent(stop) || wireMessages.isTerminalEvent(delta) {
		t.Error("the Messages wire ends at message_stop and nothing else")
	}
	// A bare data frame naming the type, with no event: line, still counts.
	if !wireMessages.isTerminalEvent([]byte("data: {\"type\":\"message_stop\"}\n\n")) {
		t.Error("message_stop must be recognised without its event: line")
	}
}
