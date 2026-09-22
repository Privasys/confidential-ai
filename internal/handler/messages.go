// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// The Anthropic Messages wire, served beside the OpenAI chat-completions one.
//
// Both doors reach the same attested model in the same measured image: the
// request goes through the same reproducibility path (pinned seed, explicit
// sampling, injected clock, salted prefix cache) and the same meter. Only the
// JSON shapes differ, and this file holds every place they do:
//
//	                    chat completions            messages
//	request             seed, stream_options        seed (patched in, see
//	                                                patches/vllm/0003), no
//	                                                stream_options: vLLM's
//	                                                own converter asks the
//	                                                engine for usage
//	usage (body)        usage.prompt_tokens /       usage.input_tokens /
//	                    completion_tokens           output_tokens, with the
//	                                                cache counts SEPARATE
//	usage (stream)      one final chunk with an     message_start carries the
//	                    empty choices array         prompt, message_delta the
//	                                                completion
//	cache hits          usage.prompt_tokens_        usage.cache_read_input_
//	                    details.cached_tokens       tokens
//	end of stream       data: [DONE]                event: message_stop
//
// vLLM 0.29 serves /v1/messages itself (vllm/entrypoints/anthropic), so this
// proxy forwards rather than translates: what it adds is what it adds on the
// other wire — attestation, metering, and the reproducibility block.
// The wire is the request/response SHAPE; the upstream route stays a separate
// argument, because /v1/completions speaks the chat shapes on its own path.
type wire struct {
	// name is the wire's short name, used in logs and tests.
	name string
}

var (
	wireChat     = wire{name: "chat completions"}
	wireMessages = wire{name: "messages"}
)

// isMessages reports whether this is the Anthropic wire.
func (w wire) isMessages() bool { return w.name == wireMessages.name }

// reproEventType names the reproducibility frame on the Messages wire. The
// Anthropic protocol is a typed-event protocol: its readers refuse a frame
// whose payload carries no `type` (dsh 0.1.7 throws "SSE event type
// mismatch" and the whole turn fails), and they ignore a type they do not
// know. So the block travels as an event of its own rather than as the bare
// data frame the chat wire uses, where the reader instead drops any frame
// without `choices`. Vendor-prefixed, because it is not Anthropic's.
const reproEventType = "privasys_reproducibility"

// reproFrame renders the reproducibility block as one SSE event on this wire.
func (w wire) reproFrame(meta any) ([]byte, error) {
	if !w.isMessages() {
		body, err := json.Marshal(map[string]any{"reproducibility": meta})
		if err != nil {
			return nil, err
		}
		return []byte("data: " + string(body) + "\n\n"), nil
	}
	body, err := json.Marshal(map[string]any{"type": reproEventType, "reproducibility": meta})
	if err != nil {
		return nil, err
	}
	return []byte("event: " + reproEventType + "\n" + "data: " + string(body) + "\n\n"), nil
}

// isTerminalEvent reports whether this SSE event ends the stream on this
// wire. The reproducibility frame is emitted just before it.
func (w wire) isTerminalEvent(event []byte) bool {
	if w.isMessages() {
		return isMessagesTerminalEvent(event)
	}
	return isDoneEvent(event)
}

// usage reads the token totals from a complete response body on this wire.
func (w wire) usage(body []byte) (id string, in, out int64, ok bool) {
	if w.isMessages() {
		return messagesUsage(body)
	}
	return extractUsage(body)
}

// cachedTokens reads the prefix-cache hit count from a complete body.
func (w wire) cachedTokens(body []byte) *int64 {
	if w.isMessages() {
		return messagesCachedTokens(body)
	}
	return extractCachedTokens(body)
}

// messages serves POST /v1/messages: the Anthropic Messages wire onto the
// same model, with the same guarantees as chat completions.
func (h *Handler) messages(w http.ResponseWriter, r *http.Request) {
	h.generate(w, r, "/v1/messages", wireMessages)
}

// countTokens serves POST /v1/messages/count_tokens. It samples nothing and
// returns no content, so it neither meters nor carries a reproducibility
// block; it is forwarded so a Messages client can size a prompt before
// sending it.
func (h *Handler) countTokens(w http.ResponseWriter, r *http.Request) {
	if !h.IsReady() {
		writeError(w, http.StatusServiceUnavailable, h.NotReadyMessage())
		return
	}
	if _, ok := h.authorizeInference(w, r); !ok {
		return
	}
	h.proxyDirect(w, r, "/v1/messages/count_tokens")
}

// anthropicUsage is the usage object of the Messages wire. Anthropic reports
// the cached and newly-cached prompt tokens BESIDE input_tokens rather than
// inside it, so the billable prompt total is their sum — which is what the
// chat wire's prompt_tokens already is.
type anthropicUsage struct {
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheReadInputTokens     *int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens *int64 `json:"cache_creation_input_tokens"`
}

// prompt returns the billable prompt total, comparable to the chat wire's
// usage.prompt_tokens.
func (u anthropicUsage) prompt() int64 {
	total := u.InputTokens
	if u.CacheReadInputTokens != nil {
		total += *u.CacheReadInputTokens
	}
	if u.CacheCreationInputTokens != nil {
		total += *u.CacheCreationInputTokens
	}
	return total
}

// messagesEnvelope is the part of a Messages response, or of one streamed
// event, that carries identity and usage.
type messagesEnvelope struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Usage   *anthropicUsage `json:"usage"`
	Message *struct {
		ID    string          `json:"id"`
		Usage *anthropicUsage `json:"usage"`
	} `json:"message"`
}

// id returns the response id, wherever this shape carries it.
func (e messagesEnvelope) id() string {
	if e.ID != "" {
		return e.ID
	}
	if e.Message != nil {
		return e.Message.ID
	}
	return ""
}

// usage returns the usage this shape carries, if any.
func (e messagesEnvelope) usage() *anthropicUsage {
	if e.Usage != nil {
		return e.Usage
	}
	if e.Message != nil {
		return e.Message.Usage
	}
	return nil
}

// messagesUsage reads the token totals from a complete Messages response.
func messagesUsage(body []byte) (id string, in, out int64, ok bool) {
	var e messagesEnvelope
	if err := json.Unmarshal(body, &e); err != nil {
		return "", 0, 0, false
	}
	u := e.usage()
	if u == nil {
		return "", 0, 0, false
	}
	return e.id(), u.prompt(), u.OutputTokens, true
}

// messagesCachedTokens reads the prefix-cache hit count from a complete
// Messages response.
func messagesCachedTokens(body []byte) *int64 {
	var e messagesEnvelope
	if err := json.Unmarshal(body, &e); err != nil {
		return nil
	}
	if u := e.usage(); u != nil {
		return u.CacheReadInputTokens
	}
	return nil
}

// messagesStreamUsage accumulates the token totals of one streamed Messages
// response. Unlike the chat wire, which reports everything in a single final
// chunk, the Messages wire reports the prompt in `message_start` and the
// completion in `message_delta`, so the totals are only complete at the end
// of the stream.
type messagesStreamUsage struct {
	id     string
	in     int64
	out    int64
	cached *int64
	seen   bool
}

// observe takes one SSE event and folds any usage it carries into the totals.
func (m *messagesStreamUsage) observe(event []byte) {
	for _, data := range sseData(event) {
		var e messagesEnvelope
		if err := json.Unmarshal(data, &e); err != nil {
			continue
		}
		if id := e.id(); id != "" && m.id == "" {
			m.id = id
		}
		u := e.usage()
		if u == nil {
			continue
		}
		m.seen = true
		// Both events carry a prompt count and the later one is the
		// settled figure; the completion count only grows.
		if p := u.prompt(); p > m.in {
			m.in = p
		}
		if u.OutputTokens > m.out {
			m.out = u.OutputTokens
		}
		if u.CacheReadInputTokens != nil {
			m.cached = u.CacheReadInputTokens
		}
	}
}

// total reports the accumulated usage, and whether any was seen at all.
func (m *messagesStreamUsage) total() (id string, in, out int64, ok bool) {
	return m.id, m.in, m.out, m.seen
}

// sseData returns the payload of every `data:` line in one SSE event.
func sseData(event []byte) [][]byte {
	var out [][]byte
	for _, line := range bytes.Split(event, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data:"):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		out = append(out, data)
	}
	return out
}

// isMessagesTerminalEvent reports whether this SSE event ends a Messages
// stream. It is the Messages wire's `data: [DONE]`: the frame carrying the
// reproducibility block is emitted just before it, so a client that reads to
// the end of the stream has the block whichever wire it speaks.
func isMessagesTerminalEvent(event []byte) bool {
	for _, line := range bytes.Split(bytes.TrimRight(event, "\n"), []byte("\n")) {
		if bytes.Equal(bytes.TrimSpace(line), []byte("event: message_stop")) {
			return true
		}
	}
	for _, data := range sseData(event) {
		var e messagesEnvelope
		if err := json.Unmarshal(data, &e); err == nil && e.Type == "message_stop" {
			return true
		}
	}
	return false
}
