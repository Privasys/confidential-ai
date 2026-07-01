package handler

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/privasys/confidential-ai/internal/billing"
)

// billingConfigPayload is the JSON body accepted by POST /configure and the
// shape persisted to the encrypted volume. It carries the four billing
// values that env vars used to deliver; container apps receive per-deploy
// configuration via the configure-then-freeze pattern instead (env vars are
// not carried by the container load envelope — see README / the pricing model
// §6.3).
//
// Trust model: the manager freezes all non-/configure paths with HTTP 503
// until this endpoint returns 2xx, and the only practical caller is the
// owner-authenticated management-service RPC relay
// (POST /api/v1/apps/{id}/rpc/configure → RA-TLS → Caddy → this handler).
// The endpoint therefore performs no additional bearer check; it validates
// and applies the payload.
type billingConfigPayload struct {
	BillingAccountID string `json:"billing_account_id"`
	UsageReportURL   string `json:"usage_report_url"`
	UsageReportToken string `json:"usage_report_token"`
	BillingModel     string `json:"billing_model"`

	// InferenceAuthRequired toggles inference-auth enforcement at runtime.
	// Env vars are not deliverable to container apps, so this is the only
	// way to turn on enforcement for a deployed instance. A pointer so an
	// omitted key leaves the current value untouched (a billing-only
	// configure must not silently flip enforcement); when present it is
	// applied and persisted, independent of the billing fields.
	InferenceAuthRequired *bool `json:"inference_auth_required,omitempty"`
}

// toConfig maps the wire payload to a billing.Config, applying the same
// model-slug fallback as New (payload → cfg.BillingModel → cfg.ModelName).
func (h *Handler) toConfig(p billingConfigPayload) billing.Config {
	model := p.BillingModel
	if model == "" {
		model = h.cfg.BillingModel
	}
	if model == "" {
		model = h.cfg.ModelName
	}
	return billing.Config{
		AccountID:   p.BillingAccountID,
		ReportURL:   p.UsageReportURL,
		ReportToken: p.UsageReportToken,
		Model:       model,
	}
}

// configure handles POST /configure. It accepts the billing configuration,
// persists it to the encrypted volume, and swaps the live metering Reporter.
// Returning 2xx lifts the manager's freeze gate so the proxy begins serving
// inference. The call is idempotent: re-delivering the same config (e.g.
// after a Spot-VM restart) simply re-applies it.
func (h *Handler) configure(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<10))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	var p billingConfigPayload
	if err := json.Unmarshal(body, &p); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON request body")
		return
	}

	// Both empty is an explicit "disable metering". Exactly one empty is a
	// misconfiguration (a typo would otherwise silently disable billing).
	hasAccount := p.BillingAccountID != ""
	hasURL := p.UsageReportURL != ""
	if hasAccount != hasURL {
		writeError(w, http.StatusBadRequest,
			"billing_account_id and usage_report_url must be set together")
		return
	}

	if err := h.persistBillingConfig(p); err != nil {
		log.Printf("[billing] failed to persist config: %v", err)
		writeError(w, http.StatusInternalServerError, "failed to persist configuration")
		return
	}

	cfg := h.toConfig(p)
	h.ReconfigureBilling(cfg)

	// Apply the enforcement toggle independently of billing (only when the
	// caller included the key).
	if p.InferenceAuthRequired != nil {
		h.inferenceAuth.Store(*p.InferenceAuthRequired)
		log.Printf("[auth] inference-auth enforcement set to %v via configure", *p.InferenceAuthRequired)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":                  "ok",
		"metering":                !cfg.Disabled(),
		"inference_auth_required": h.inferenceAuth.Load(),
	})
}

// persistBillingConfig atomically writes the payload to the configured
// persistence path with 0600 permissions. A no-op when persistence is
// disabled (empty path).
func (h *Handler) persistBillingConfig(p billingConfigPayload) error {
	path := h.cfg.BillingConfigFile
	if path == "" {
		return nil
	}
	buf, err := json.Marshal(p)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".billing-config-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(buf); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// RestorePersistedBilling reloads billing config from the encrypted volume
// on startup so metering survives a restart without waiting for the
// orchestrator to re-deliver via POST /configure. A missing file is not an
// error (metering stays as configured by env, typically disabled). Call
// after StartBilling so the reporter loop context is available.
func (h *Handler) RestorePersistedBilling() {
	path := h.cfg.BillingConfigFile
	if path == "" {
		return
	}
	buf, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[billing] failed to read persisted config: %v", err)
		}
		return
	}
	var p billingConfigPayload
	if err := json.Unmarshal(buf, &p); err != nil {
		log.Printf("[billing] persisted config is corrupt, ignoring: %v", err)
		return
	}
	// Restore the enforcement toggle first, so it survives a restart even when
	// billing is disabled (the early return below).
	if p.InferenceAuthRequired != nil {
		h.inferenceAuth.Store(*p.InferenceAuthRequired)
		log.Printf("[auth] restored inference-auth enforcement = %v", *p.InferenceAuthRequired)
	}
	cfg := h.toConfig(p)
	if cfg.Disabled() {
		return
	}
	h.ReconfigureBilling(cfg)
	log.Printf("[billing] restored persisted billing config (account=%s)", cfg.AccountID)
}
