package handler

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

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
		writeError(w, http.StatusServiceUnavailable, h.NotReadyMessage())
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
		s := newSeed()
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

	// Inject the per-request dynamic context (wall clock) just before the
	// latest user turn, recorded below for reproducible replay. Out of the
	// static system prompt so that prefix stays cacheable.
	dynCtx := dynamicContext(r)
	body, err = injectDynamicContext(body, dynCtx)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject context")
		return
	}

	// Resolve the catalogue for this request. By default it is the
	// configured (admin/whitelist) catalogue. When a valid tool-grant is
	// present, union the user's authorised tools into an ephemeral,
	// request-scoped catalogue + dispatcher so concurrent sessions never
	// see each other's tools.
	cat := h.agentCatalog
	disp := h.agentDispatcher
	if gv := h.grantVerifier.Load(); gv != nil {
		if gtok := r.Header.Get("X-Privasys-Tool-Grant"); gtok != "" {
			gservers, gerr := gv.GrantServers(r.Context(), gtok)
			if gerr != nil {
				log.Printf("[agent] tool-grant rejected: %v", gerr)
			} else if len(gservers) > 0 {
				merged := agent.MergeServers(h.agentCatalog.Servers(), gservers)
				// Route transport per tool kind. Enclave tools (grant
				// carries their workload digest) MUST use the admin
				// catalogue's attested RA-TLS transport — the gateway's
				// terminated leg refuses plain HTTP
				// (sealed-transport-required); with a plain client every
				// granted enclave tool silently vanished from the union.
				// External tools (no digest) go over WebPKI TLS with an
				// SSRF-guarded dialer instead — RA-TLS would refuse them,
				// and their URL is user input reaching out from inside
				// the enclave. Admin (fleet) servers carry no digest but
				// only external GRANT servers are registered on the
				// external path, so fleet tools stay attested.
				var enclaveRT http.RoundTripper
				if h.agentCatClient != nil {
					enclaveRT = h.agentCatClient.Transport
				}
				// Pin each granted enclave tool to the workload digest the
				// user admitted (fails closed if the app changed since).
				enclaveRT = agent.PinnedEnclaveTransport(enclaveRT, gservers)
				router := agent.NewKindRouter(enclaveRT, agent.ExternalHostsOf(gservers))
				cat = agent.NewCatalog(merged, &http.Client{Timeout: 15 * time.Second, Transport: router}, 60*time.Second)
				disp = agent.NewDispatcher(cat, &http.Client{Timeout: 60 * time.Second, Transport: router})
				defer cat.Close()
			}
		}
	}

	// Fetch tool catalogue and inject into request.
	tools, _ := cat.Tools(r.Context())
	tools = filterToolsByHeader(tools, r.Header.Get("X-Privasys-Tools"))
	body, err = agent.InjectTools(body, tools)
	if err != nil {
		h.requestsFailed.Add(1)
		writeError(w, http.StatusInternalServerError, "failed to inject tools")
		return
	}

	// Only a real Bearer credential may flow to tools. On the sealed
	// transport the Authorization header carries the session relay's OWN
	// scheme ("PrivasysSession <id>") — TrimPrefix left that whole string
	// as the "bearer", so every tool call forwarded
	// "Bearer PrivasysSession …" and the tool's OIDC layer refused it
	// ("invalid or expired token") even for public functions. No bearer →
	// no Authorization header on the tool call: public tool functions
	// work; authed ones correctly refuse the anonymous caller.
	// TODO(ai-tools): propagate the sealed caller's identity to tools
	// (IdP exchange keyed on the relay-asserted sub) so authed tool
	// functions work from the chat too.
	bearer := ""
	if v := r.Header.Get("Authorization"); strings.HasPrefix(v, "Bearer ") {
		bearer = strings.TrimPrefix(v, "Bearer ")
	}
	wantStream := reqParams.Stream

	// Set up the SSE stream early when the client wants one, so events
	// can flow as the loop progresses.
	var (
		flusher   http.Flusher
		streaming bool
		writeMu   sync.Mutex // serialises all writes to w during streaming
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
		writeMu.Lock()
		defer writeMu.Unlock()
		// SSE allows custom event names alongside the default `message`.
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventName, b)
		flusher.Flush()
	}

	// Billing (the pricing model): the agentic path makes one or more
	// upstream vLLM calls per request (one per tool-loop turn). Each call's
	// token usage must be metered, exactly like the plain proxy paths. We
	// record usage from every upstream response body — both the synthesised
	// stream body (which embeds the include_usage totals) and the native
	// non-stream body carry a `usage` block. nil reporter ⇒ metering off.
	rep := h.billingReporter()

	// Pick the upstream invoker based on whether the client wants a
	// stream. Both implementations return a non-stream-shaped response
	// body so agent.Run can inspect tool_calls between iterations.
	var invoke func(ctx context.Context, b []byte) ([]byte, error)
	if streaming {
		// Streaming path: forward content/finish chunks live to the
		// client while accumulating the full assistant message (and any
		// tool-call argument fragments) into a synthesised non-stream
		// body. tool_call deltas are intentionally suppressed from the
		// relayed stream — the front-end gets richer `tool_call` /
		// `tool_result` SSE events instead.
		invoke = func(ctx context.Context, b []byte) ([]byte, error) {
			b, err := setStreamMode(b, true)
			if err != nil {
				return nil, err
			}
			// Ask vLLM for the final usage chunk so the synthesised body
			// carries token totals to meter. The usage-only chunk is
			// captured by callVLLMStream and never forwarded to the
			// client, so the relayed stream is unchanged.
			if rep != nil {
				b, _ = injectStreamUsage(b)
			}
			out, err := h.callVLLMStream(ctx, b, func(chunk []byte) {
				writeMu.Lock()
				defer writeMu.Unlock()
				w.Write(chunk)
				flusher.Flush()
			})
			if err == nil && rep != nil {
				if id, in, o, ok := extractUsage(out); ok {
					rep.Record(id, callerFromContext(r.Context()), in, o)
				}
			}
			return out, err
		}
	} else {
		// Non-stream path: vLLM rejects `stream_options` whenever
		// `stream != true`, so strip it when forcing stream=false.
		invoke = func(ctx context.Context, b []byte) ([]byte, error) {
			b, err := setStreamMode(b, false)
			if err != nil {
				return nil, err
			}
			out, err := h.callVLLM(ctx, b)
			if err == nil && rep != nil {
				if id, in, o, ok := extractUsage(out); ok {
					rep.Record(id, callerFromContext(r.Context()), in, o)
				}
			}
			return out, err
		}
	}

	finalBody, results, err := agent.Run(r.Context(), disp, body, agent.LoopOptions{
		Bearer:    bearer,
		EmitEvent: emit,
		Invoke:    invoke,
		WaitConsent: func(ctx context.Context, callID, name string, args []byte) (bool, error) {
			if h.agentConsent == nil {
				// Consent registry not initialised; default to allow so
				// existing behaviour is preserved.
				return true, nil
			}
			dec, werr := h.agentConsent.Wait(ctx, callID, 2*time.Minute)
			if werr != nil {
				return false, werr
			}
			return dec.Allowed, nil
		},
	})
	if err != nil {
		h.requestsFailed.Add(1)
		if streaming {
			emit("error", map[string]string{"message": err.Error()})
			writeMu.Lock()
			fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
			writeMu.Unlock()
			return
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	meta := h.buildMetadata(reqParams)
	meta.DynamicContext = dynCtx
	addToolCallsToMeta(meta, results)
	wantRepro := wantsReproducibility(r)

	if !streaming {
		var vllmResp map[string]any
		if err := json.Unmarshal(finalBody, &vllmResp); err != nil {
			if !wantRepro {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				w.Write(finalBody)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]any{
				"response":        json.RawMessage(finalBody),
				"reproducibility": meta,
			})
			return
		}
		if wantRepro {
			vllmResp["reproducibility"] = meta
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(vllmResp)
		return
	}

	// Streaming: the assistant content was already forwarded chunk-by-
	// chunk to the client by callVLLMStream. We just need to append the
	// reproducibility metadata (opt-in) and the terminator.
	writeMu.Lock()
	if wantRepro {
		metaJSON, _ := json.Marshal(map[string]any{"reproducibility": meta})
		fmt.Fprintf(w, "data: %s\n\n", metaJSON)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	writeMu.Unlock()
}

// setStreamMode rewrites the request body's `stream` flag and, when
// forcing it off, also strips `stream_options` (vLLM 400s on the pair).
func setStreamMode(body []byte, stream bool) ([]byte, error) {
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	req["stream"] = stream
	if !stream {
		delete(req, "stream_options")
	}
	return json.Marshal(req)
}

// callVLLM is the low-level non-stream proxy used by the agent loop
// when the client did NOT request streaming.
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

// callVLLMStream POSTs `body` (which MUST already have stream=true) to
// vLLM, streams SSE chunks to `sink` for the chunks the client cares
// about (content/finish), and returns a synthesised non-stream-shaped
// response body the agent loop can parse for tool_calls.
//
// What's forwarded to `sink`:
//   - `data: {…delta.content…}` chunks (raw, byte-for-byte from vLLM)
//   - the final `data: {…finish_reason…}` chunk (with delta.tool_calls
//     stripped if present, so the front-end's chat parser doesn't try
//     to render half-formed tool calls)
//
// What's NOT forwarded:
//   - chunks whose only delta is `tool_calls`
//   - the terminal `data: [DONE]` (the caller emits its own after
//     appending the reproducibility block)
//   - the `data: [DONE]` line in general (we own the terminator)
//
// Accumulated into the returned body:
//   - assistant.role / assistant.content (concatenated content deltas)
//   - assistant.tool_calls[] (id, type, function.name, function.arguments
//     concatenated by index)
//   - finish_reason from the last chunk that carried one
func (h *Handler) callVLLMStream(ctx context.Context, body []byte, sink func([]byte)) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.cfg.VLLMUpstream+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		return nil, fmt.Errorf("vllm status %d: %s", resp.StatusCode, strings.TrimSpace(string(errBody)))
	}

	type tcAccum struct {
		ID      string
		Type    string
		Name    string
		ArgsBuf bytes.Buffer
	}
	var (
		contentBuf       bytes.Buffer
		role             = "assistant"
		finishReason     string
		idStr            string
		objectStr        = "chat.completion"
		createdNum       int64
		modelStr         string
		toolCalls        = map[int]*tcAccum{}
		promptTokens     int64
		completionTokens int64
		usageSeen        bool
	)

	// SSE lines can be long (a tool_call argument fragment may be a few
	// KB). 1MB buffer is way more than enough for any one chunk.
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := bytes.TrimSpace(line[len("data: "):])
		if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}

		// Parse the chunk to accumulate state.
		var chunk struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			Model   string `json:"model"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Role             string `json:"role"`
					Content          string `json:"content"`
					Reasoning        string `json:"reasoning"`
					ReasoningContent string `json:"reasoning_content"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int64 `json:"prompt_tokens"`
				CompletionTokens int64 `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(payload, &chunk); err != nil {
			// Unparseable — best effort: forward as-is so we don't
			// hide an upstream error event from the client.
			sink(append(append([]byte{}, line...), '\n', '\n'))
			continue
		}
		if idStr == "" {
			idStr = chunk.ID
		}
		if chunk.Object != "" {
			objectStr = chunk.Object
		}
		if chunk.Created != 0 {
			createdNum = chunk.Created
		}
		if modelStr == "" {
			modelStr = chunk.Model
		}

		// Billing: capture the include_usage totals (the final chunk
		// carries `usage` with an empty `choices` array). Record them
		// onto the synthesised body and never forward this usage-only
		// chunk to the client so the relayed stream stays unchanged.
		if chunk.Usage != nil && (chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0) {
			promptTokens = chunk.Usage.PromptTokens
			completionTokens = chunk.Usage.CompletionTokens
			usageSeen = true
		}
		if len(chunk.Choices) == 0 {
			// A usage-only / heartbeat chunk has nothing to forward.
			continue
		}

		hasContent := false
		hasToolCallDelta := false
		for _, ch := range chunk.Choices {
			if ch.Delta.Role != "" {
				role = ch.Delta.Role
			}
			if ch.Delta.Content != "" {
				contentBuf.WriteString(ch.Delta.Content)
				hasContent = true
			}
			// Native reasoning channel (vLLM `--reasoning-parser`):
			// surface to the client just like content. Without this
			// the agentic loop's stream filter swallowed every
			// reasoning_content/reasoning delta and the chat UI saw
			// only the final finish_reason chunk after the model had
			// burned its full max_tokens budget thinking.
			if ch.Delta.ReasoningContent != "" || ch.Delta.Reasoning != "" {
				hasContent = true
			}
			if ch.FinishReason != "" {
				finishReason = ch.FinishReason
			}
			for _, tc := range ch.Delta.ToolCalls {
				hasToolCallDelta = true
				acc, ok := toolCalls[tc.Index]
				if !ok {
					acc = &tcAccum{}
					toolCalls[tc.Index] = acc
				}
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Type != "" {
					acc.Type = tc.Type
				}
				if tc.Function.Name != "" {
					acc.Name = tc.Function.Name
				}
				if tc.Function.Arguments != "" {
					acc.ArgsBuf.WriteString(tc.Function.Arguments)
				}
			}
		}

		// Forward the chunk to the client only when it carries
		// user-visible content OR a finish_reason. Skip pure
		// tool_call-delta chunks (the front-end will see synthetic
		// tool_call / tool_result events instead).
		if hasContent || (finishReason != "" && !hasToolCallDelta) {
			sink(append(append([]byte{}, line...), '\n', '\n'))
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read sse: %w", err)
	}

	// Synthesise the non-stream OpenAI response shape the loop expects.
	msg := map[string]any{
		"role":    role,
		"content": contentBuf.String(),
	}
	if len(toolCalls) > 0 {
		// Order by Index to keep determinism.
		out := make([]map[string]any, 0, len(toolCalls))
		for i := 0; i < len(toolCalls); i++ {
			acc, ok := toolCalls[i]
			if !ok {
				continue
			}
			out = append(out, map[string]any{
				"id":   acc.ID,
				"type": acc.Type,
				"function": map[string]any{
					"name":      acc.Name,
					"arguments": acc.ArgsBuf.String(),
				},
			})
		}
		msg["tool_calls"] = out
	}
	full := map[string]any{
		"id":      idStr,
		"object":  objectStr,
		"created": createdNum,
		"model":   modelStr,
		"choices": []map[string]any{{
			"index":         0,
			"message":       msg,
			"finish_reason": finishReason,
		}},
	}
	// Surface the metered token totals so the agentic invoke wrapper can
	// extractUsage(...) from this synthesised body, matching the native
	// non-stream path. Absent when the client did not enable include_usage
	// and metering was off (no usage chunk requested).
	if usageSeen {
		full["usage"] = map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		}
	}
	return json.Marshal(full)
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

// filterToolsByHeader trims the catalogue to only the servers named in
// the X-Privasys-Tools header (comma-separated). An empty header keeps
// every server (default-on behaviour preserved). Unknown server names
// in the header are silently dropped.
func filterToolsByHeader(tools []agent.Tool, header string) []agent.Tool {
	header = strings.TrimSpace(header)
	if header == "" {
		return tools
	}
	allowed := map[string]bool{}
	for _, name := range strings.Split(header, ",") {
		if n := strings.TrimSpace(name); n != "" {
			allowed[n] = true
		}
	}
	if len(allowed) == 0 {
		return tools
	}
	out := tools[:0:0]
	for _, t := range tools {
		if allowed[t.Server] {
			out = append(out, t)
		}
	}
	return out
}
