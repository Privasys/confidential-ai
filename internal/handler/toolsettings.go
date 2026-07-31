package handler

// Per-user tool settings, proxied to the owning MCP server (the generic
// "complex tool integration" surface — chat-byo-mcp plan follow-up).
//
// A tool server may advertise a `settings` descriptor in its catalogue
// (title/icon/description); its per-user settings document then lives at
// <base>/api/v1/mcp/settings on the TOOL, enforced by the tool. These
// routes only (a) tell clients which servers have such a surface, and
// (b) carry the read/write through the enclave's attested channel with the
// sealed caller's identity in X-Privasys-On-Behalf-Of. Nothing here is
// specific to any one tool: Drive's Memory panel and any future rich tool
// render from the same contract.
//
// Auth: a settings request must carry an acting USER (relay-asserted
// sealed sub or a verified bearer). Attested app callers ("app:" subjects)
// are refused — settings are user preferences, and an app has none.

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/privasys/confidential-ai/internal/agent"
)

// toolServers serves GET /v1/tools: the configured MCP servers with their
// settings descriptors, so a chat client discovers rich integrations
// generically. Static fleet configuration only — safe without auth (the
// same facts are already public in the fleet's tool spec and in the
// attestation extension OID ...3.5.7 digest).
func (h *Handler) toolServers(w http.ResponseWriter, r *http.Request) {
	type serverInfo struct {
		Name        string          `json:"name"`
		HasSettings bool            `json:"has_settings"`
		Settings    json.RawMessage `json:"settings,omitempty"`
	}
	out := []serverInfo{}
	if h.agentCatalog != nil {
		descriptors := h.agentCatalog.SettingsDescriptors(r.Context())
		for _, s := range h.agentCatalog.Servers() {
			info := serverInfo{Name: s.Name}
			if sd, ok := descriptors[s.Name]; ok {
				info.HasSettings = true
				info.Settings = sd
			}
			out = append(out, info)
		}
	}
	writeJSONBody(w, http.StatusOK, map[string]interface{}{"servers": out})
}

// toolSettings serves GET+PUT /v1/tools/{server}/settings by proxying to
// the owning server's /api/v1/mcp/settings with the acting user named in
// X-Privasys-On-Behalf-Of. Status and body are forwarded verbatim: the
// attested tool is the sole authority on the document.
func (h *Handler) toolSettings(w http.ResponseWriter, r *http.Request) {
	if h.agentCatalog == nil || h.agentDispatcher == nil {
		writeError(w, http.StatusNotFound, "no tool servers configured")
		return
	}
	sub, err := h.resolveCaller(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, "invalid credential")
		return
	}
	if sub == "" || strings.HasPrefix(sub, "app:") {
		writeError(w, http.StatusUnauthorized, "authentication required")
		return
	}
	server := r.PathValue("server")
	if !h.agentCatalog.HasSettings(r.Context(), server) {
		writeError(w, http.StatusNotFound, "server has no settings surface")
		return
	}
	var body []byte
	if r.Method == http.MethodPut {
		body, err = io.ReadAll(io.LimitReader(r.Body, 256<<10))
		if err != nil {
			writeError(w, http.StatusBadRequest, "unreadable body")
			return
		}
	}
	ctx := agent.WithOnBehalfOf(r.Context(), sub)
	status, respBody, err := h.agentDispatcher.SettingsRoundTrip(ctx, server, r.Method, body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "settings proxy failed: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(respBody)
}

// writeJSONBody is writeError's success-path sibling.
func writeJSONBody(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
