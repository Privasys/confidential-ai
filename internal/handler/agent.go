package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/privasys/confidential-ai/internal/agent"
	"github.com/privasys/confidential-ai/internal/reproducibility"
)

// chatCompletionsAgentic is the entry point when one or more upstream
// MCP servers are configured. It runs the bounded tool-call loop in
// agent.Run, synthesises an SSE stream that includes tool_call /
// tool_result events alongside the final assistant delta, and adds a
// tool_calls[] array to the reproducibility block.
//
// When the original request set stream=false, the response is a single
// JSON document mirroring vLLM's shape with reproducibility.tool_calls
// populated. When stream=true, every tool round trip is reported live.
func (h *Handler) chatCompletionsAgentic(w http.ResponseWriter, r *http.Request) {
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
	var reqParams requestParams
	if err := json.Unmarshal(body, &reqParams); err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusBadRequest, "invalid JSON request body")
		return
	}
	if reqParams.Seed == nil {
		s := int64(0)
		reqParams.Seed = &s
	}
	if reqParams.Temperature == 0 {
		reqParams.Temperature = 1.0
	}
	if reqParams.TopP == 0 {
		reqParams.TopP = 1.0
	}
	body, err = injectSeed(body, *reqParams.Seed)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject seed")
		return
	}

	// Fetch tool catalogue and inject into request.
	tools, _ := h.agentCatalog.Tools(r.Context())
	body, err = agent.InjectTools(body, tools)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject tools")
		return
	}

	bearer := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	wantStream := reqParams.Stream

	// Set up the SSE stream early when the client wants one, so events
	// can flow as the loop progresses.
	var (
		flusher http.Flusher
		streaming bool
	)
	if wantStream {
		f, ok := w.(http.Flusher)
		if !ok {
			h.requestsFailed.Add(1)
			writeError(w, http.StatusInternalServerError, "streaming not supported")
			return
		}
		flusher = f
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		streaming = true
	}

	emit := func(eventName string, payload any) {
		if !streaming {
			return
		}
		b, err := json.Marshal(payload)
		if err != nil {
			return
		}
		// SSE allows custom event names alongside the default `message`.
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventName, b)
		flusher.Flush()
	}

	finalBody, results, err := agent.Run(r.Context(), h.agentDispatcher, body, agent.LoopOptions{
		Bearer:    bearer,
		EmitEvent: emit,
		Invoke: func(ctx context.Context, b []byte) ([]byte, error) {
			return h.callVLLM(ctx, b)
		},
	})
	if err != nil {
		h.requestsFailed.Add(1)
		if streaming {
			emit("error", map[string]string{"message": err.Error()})
			fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	meta := h.buildMetadata(reqParams)
	addToolCallsToMeta(meta, results)

	if !streaming {
		var vllmResp map[string]any
		if err := json.Unmarshal(finalBody, &vllmResp); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]any{
				"response":        json.RawMessage(finalBody),
				"reproducibility": meta,
			})
			return
		}
		vllmResp["reproducibility"] = meta
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(vllmResp)
		return
	}

	// Streaming: synthesise OpenAI-compatible deltas from the final
	// assistant message. We emit the entire content as ONE delta - the
	// real model already finished by the time we get here. A future
	// improvement is to re-invoke vLLM with stream=true on the final
	// turn so the user sees real-time tokens after tool calls; for now
	// the tool_call / tool_result events are the live signal.
	emitFinalAssistant(w, flusher, finalBody)
	metaJSON, _ := json.Marshal(map[string]any{"reproducibility": meta})
	fmt.Fprintf(w, "data: %s\n\n", metaJSON)
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// callVLLM is the low-level non-stream proxy used by the agent loop.
func (h *Handler) callVLLM(ctx context.Context, body []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.cfg.VLLMUpstream+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 50<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vllm status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}
	return respBody, nil
}

func (h *Handler) buildMetadata(p requestParams) *reproducibility.Metadata {
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
	return reproducibility.NewMetadata(
		*p.Seed, p.Temperature, p.TopP, p.TopK, p.MaxTokens,
		modelName, quantization,
		h.cfg.VLLMVersion, h.cfg.CUDAVersion, h.cfg.GPUType,
		h.cfg.ImageDigest, h.cfg.TeeType,
	)
}

// addToolCallsToMeta serialises the tool results onto the metadata's
// ToolCalls slice.
func addToolCallsToMeta(meta *reproducibility.Metadata, results []agent.ToolResult) {
	if len(results) == 0 {
		return
	}
	out := make([]reproducibility.ToolCallSummary, len(results))
	for i, r := range results {
		out[i] = reproducibility.ToolCallSummary{
			Name:       r.Name,
			Status:     r.Status,
			DurationMs: r.DurationMs,
			Error:      r.Error,
		}
	}
	meta.ToolCalls = out
}

// emitFinalAssistant unpacks the assistant message text from the final
// vLLM response and emits it as a single OpenAI-compatible streaming
// delta.
func emitFinalAssistant(w http.ResponseWriter, flusher http.Flusher, finalBody []byte) {
	var resp struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(finalBody, &resp); err != nil || len(resp.Choices) == 0 {
		return
	}
	chunk := map[string]any{
		"id":      resp.ID,
		"object":  "chat.completion.chunk",
		"created": resp.Created,
		"model":   resp.Model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{"role": "assistant", "content": resp.Choices[0].Message.Content},
			"finish_reason": resp.Choices[0].FinishReason,
		}},
	}
	b, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	flusher.Flush()
}
