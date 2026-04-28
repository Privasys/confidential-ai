package handler

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/privasys/confidential-ai/internal/agent"
)

// agentConfirm receives the user's allow / deny decision for one
// in-flight tool call that was tagged `requires_user_confirmation`.
//
// Body: {"allowed": bool, "reason": string?}
//
// Returns 204 on success, 404 when the id is unknown (already
// resolved, expired, or never registered), 400 on malformed body.
//
// The id is the tool_call id surfaced on the SSE `tool_confirm_request`
// event. Auth is intentionally the same Authorization header the chat
// stream itself uses; the upstream gateway is responsible for tying
// the bearer to the right tenant.
func (h *Handler) agentConfirm(w http.ResponseWriter, r *http.Request) {
	if h.agentConsent == nil {
		http.Error(w, "agentic loop disabled", http.StatusNotFound)
		return
	}
	id := r.PathValue("id")
	if id == "" {
		http.Error(w, "missing id", http.StatusBadRequest)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 4<<10))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	var dec agent.ConsentDecision
	if len(body) > 0 {
		if err := json.Unmarshal(body, &dec); err != nil {
			http.Error(w, "invalid JSON body", http.StatusBadRequest)
			return
		}
	}
	if !h.agentConsent.Resolve(id, dec) {
		http.Error(w, "no pending confirmation for id", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
