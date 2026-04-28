package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
)

// MaxIterations bounds the tool-call loop. A model that keeps calling
// tools past this point gets its current state forcibly summarised back
// as a final assistant message.
const MaxIterations = 6

// LoopOptions tunes one Run.
type LoopOptions struct {
	// Bearer is the end-user's Authorization header value (without the
	// "Bearer " prefix). Forwarded to MCP servers that opted in.
	Bearer string

	// MaxIterations overrides the default cap. Zero means use MaxIterations.
	MaxIterations int

	// EmitEvent, if non-nil, is called for every tool_call (start) and
	// tool_result (end) so the calling layer can stream them as SSE.
	// Both calls happen in the goroutine that drives the loop.
	EmitEvent func(eventName string, payload any)

	// Invoke is the callback that drives one round-trip to vLLM.
	// Required. The implementation is provided by the handler layer
	// (it has the live http.Client + upstream URL).
	Invoke func(ctx context.Context, body []byte) ([]byte, error)

	// WaitConsent, if non-nil, is called BEFORE dispatching any tool
	// flagged with RequiresUserConfirmation. It blocks until the
	// front-end answers (via the confirm endpoint) or the context is
	// cancelled. When it returns allowed=false the call is short-
	// circuited with a synthetic "user_denied" tool result that is
	// fed back into the model so it can recover (apologise / try
	// something else). Errors are surfaced the same way.
	WaitConsent func(ctx context.Context, callID, name string, args []byte) (allowed bool, err error)
}

// Run executes the agentic loop on the given chat completion request.
//
// Inputs:
//   - body: the original (non-streaming) JSON request body, with the
//     `tools` array already injected by the caller.
//
// Outputs:
//   - the FINAL non-stream vLLM response body, with no tool_calls left.
//   - a map[name]ToolResult of every tool that ran, suitable for the
//     extended reproducibility block.
//   - an error if the loop is broken (transport failure, malformed JSON,
//     etc.). Tool errors are NOT loop errors: they are fed back into the
//     model as `tool` messages so it can recover.
func Run(ctx context.Context, dispatcher *Dispatcher, body []byte, opt LoopOptions) ([]byte, []ToolResult, error) {
	if opt.Invoke == nil {
		return nil, nil, errors.New("agent.Run: Invoke is required")
	}
	maxIter := opt.MaxIterations
	if maxIter <= 0 {
		maxIter = MaxIterations
	}

	// Decode once to a generic map; we mutate `messages` between rounds.
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, nil, fmt.Errorf("decode request: %w", err)
	}

	// Force non-stream upstream while iterating; the caller is responsible
	// for synthesising stream output if the original client wanted it.
	req["stream"] = false

	allResults := make([]ToolResult, 0, 4)

	for i := 0; i < maxIter; i++ {
		current, err := json.Marshal(req)
		if err != nil {
			return nil, allResults, fmt.Errorf("marshal request iter %d: %w", i, err)
		}
		respBody, err := opt.Invoke(ctx, current)
		if err != nil {
			return nil, allResults, fmt.Errorf("vllm invoke iter %d: %w", i, err)
		}

		toolCalls, assistantMsg, err := parseToolCalls(respBody)
		if err != nil {
			return nil, allResults, fmt.Errorf("parse response iter %d: %w", i, err)
		}
		if len(toolCalls) == 0 {
			// No more tool calls: return the model's final response.
			return respBody, allResults, nil
		}

		// Append the assistant message (with tool_calls) to history.
		messages, _ := req["messages"].([]any)
		messages = append(messages, assistantMsg)

		// Dispatch each tool call serially. Parallelism is a follow-up.
		for _, tc := range toolCalls {
			requiresConfirm := false
			if t, ok := dispatcher.catalog.Tool(tc.Function.Name); ok && t.RequiresUserConfirmation {
				requiresConfirm = true
			}
			if opt.EmitEvent != nil {
				ev := map[string]any{
					"id":   tc.ID,
					"name": tc.Function.Name,
					"args": tc.Function.Arguments,
				}
				// Surface the catalogue's "requires_user_confirmation"
				// flag so the front-end can mark the card as a
				// privileged action (write tools, send-email, etc.).
				if requiresConfirm {
					ev["requires_confirmation"] = true
				}
				opt.EmitEvent("tool_call", ev)
			}

			// Block-and-await consent for write-capable tools when the
			// caller wired a WaitConsent callback. If the user denies
			// (or the wait fails) we feed a synthetic error back to
			// the model instead of executing the tool.
			if requiresConfirm && opt.WaitConsent != nil {
				if opt.EmitEvent != nil {
					opt.EmitEvent("tool_confirm_request", map[string]any{
						"id":   tc.ID,
						"name": tc.Function.Name,
						"args": tc.Function.Arguments,
					})
				}
				allowed, werr := opt.WaitConsent(ctx, tc.ID, tc.Function.Name, []byte(tc.Function.Arguments))
				if werr != nil || !allowed {
					reason := "user_denied"
					if werr != nil {
						reason = werr.Error()
					}
					denied := ToolResult{
						Name:   tc.Function.Name,
						Status: "error",
						Error:  reason,
					}
					allResults = append(allResults, denied)
					if opt.EmitEvent != nil {
						opt.EmitEvent("tool_result", denied)
					}
					messages = append(messages, map[string]any{
						"role":         "tool",
						"tool_call_id": tc.ID,
						"name":         tc.Function.Name,
						"content":      denied.AsToolMessageContent(),
					})
					continue
				}
			}

			args := json.RawMessage(tc.Function.Arguments)
			if !json.Valid(args) {
				// vLLM emits arguments as a JSON STRING containing JSON.
				// Unwrap the outer string if so.
				var s string
				if err := json.Unmarshal([]byte(strconv.Quote(string(tc.Function.Arguments))), &s); err == nil && json.Valid([]byte(s)) {
					args = json.RawMessage(s)
				}
			}
			result := dispatcher.Call(ctx, tc.Function.Name, args, opt.Bearer)
			allResults = append(allResults, result)

			if opt.EmitEvent != nil {
				opt.EmitEvent("tool_result", result)
			}

			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": tc.ID,
				"name":         tc.Function.Name,
				"content":      result.AsToolMessageContent(),
			})
		}
		req["messages"] = messages
	}

	// Cap reached: make one final non-tool call by removing tools so the
	// model is forced to summarise.
	delete(req, "tools")
	delete(req, "tool_choice")
	final, err := json.Marshal(req)
	if err != nil {
		return nil, allResults, fmt.Errorf("marshal final: %w", err)
	}
	respBody, err := opt.Invoke(ctx, final)
	if err != nil {
		return nil, allResults, fmt.Errorf("vllm final invoke: %w", err)
	}
	return respBody, allResults, nil
}

// toolCall mirrors the OpenAI tool_calls[*] shape vLLM produces.
type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

// parseToolCalls returns the tool_calls (if any) AND the raw assistant
// message that should be appended to history before the tool messages.
func parseToolCalls(respBody []byte) ([]toolCall, map[string]any, error) {
	var resp struct {
		Choices []struct {
			Message map[string]any `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, nil, nil
	}
	msg := resp.Choices[0].Message
	rawCalls, ok := msg["tool_calls"].([]any)
	if !ok || len(rawCalls) == 0 {
		return nil, msg, nil
	}
	// Re-marshal -> unmarshal into typed shape.
	b, err := json.Marshal(rawCalls)
	if err != nil {
		return nil, msg, err
	}
	var calls []toolCall
	if err := json.Unmarshal(b, &calls); err != nil {
		return nil, msg, err
	}
	return calls, msg, nil
}

// InjectTools merges the catalogue into a chat completion request body in
// the OpenAI tools-array shape vLLM accepts. If the request already
// contains a non-empty `tools` array, the catalogue is appended (request
// wins on collisions, identified by qualified name).
func InjectTools(body []byte, tools []Tool) ([]byte, error) {
	if len(tools) == 0 {
		return body, nil
	}
	var req map[string]any
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	existing, _ := req["tools"].([]any)
	have := make(map[string]bool, len(existing))
	for _, t := range existing {
		if m, ok := t.(map[string]any); ok {
			if fn, ok := m["function"].(map[string]any); ok {
				if n, ok := fn["name"].(string); ok {
					have[n] = true
				}
			}
		}
	}
	for _, t := range tools {
		qname := t.QualifiedName()
		if have[qname] {
			continue
		}
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		existing = append(existing, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        qname,
				"description": t.Description,
				"parameters":  schema,
			},
		})
	}
	req["tools"] = existing
	if _, ok := req["tool_choice"]; !ok {
		req["tool_choice"] = "auto"
	}
	return json.Marshal(req)
}
