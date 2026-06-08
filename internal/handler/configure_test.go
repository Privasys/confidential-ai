package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/privasys/confidential-ai/internal/config"
)

// newConfigureHandler builds a Handler with billing persistence pointed at a
// temp file and the reporter loop context started, as main.go does.
func newConfigureHandler(t *testing.T) (*Handler, string) {
	t.Helper()
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "billing-config.json")
	h := New(&config.Config{
		ModelName:         "test-model",
		BillingConfigFile: cfgPath,
	}, nil)
	h.StartBilling(context.Background())
	return h, cfgPath
}

func TestConfigureEnablesMeteringAndPersists(t *testing.T) {
	h, cfgPath := newConfigureHandler(t)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	if h.billingReporter() != nil {
		t.Fatal("metering should be disabled before configure")
	}

	body := `{"billing_account_id":"acct-1","usage_report_url":"https://m/api","usage_report_token":"tok","billing_model":"qwen36-35b-a3b-fp8"}`
	req := httptest.NewRequest("POST", "/configure", strings.NewReader(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("configure: expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["status"] != "ok" || resp["metering"] != true {
		t.Fatalf("unexpected configure response: %v", resp)
	}
	if h.billingReporter() == nil {
		t.Fatal("metering should be enabled after configure")
	}

	// Persisted file should contain the delivered config with 0600 perms.
	info, err := os.Stat(cfgPath)
	if err != nil {
		t.Fatalf("config file not written: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("config file perm = %o, want 600", perm)
	}
	raw, _ := os.ReadFile(cfgPath)
	var p billingConfigPayload
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatal(err)
	}
	if p.BillingAccountID != "acct-1" || p.UsageReportToken != "tok" {
		t.Fatalf("persisted config mismatch: %+v", p)
	}
}

func TestConfigurePartialIsRejected(t *testing.T) {
	h, _ := newConfigureHandler(t)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// account without url -> 400, no reporter installed.
	req := httptest.NewRequest("POST", "/configure", strings.NewReader(`{"billing_account_id":"acct-1"}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("partial configure: expected 400, got %d", rec.Code)
	}
	if h.billingReporter() != nil {
		t.Fatal("metering must stay disabled after a rejected configure")
	}
}

func TestConfigureEmptyDisablesMetering(t *testing.T) {
	h, _ := newConfigureHandler(t)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// First enable.
	mux.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/configure",
		strings.NewReader(`{"billing_account_id":"a","usage_report_url":"https://m"}`)))
	if h.billingReporter() == nil {
		t.Fatal("metering should be enabled")
	}
	// Then an empty payload disables.
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/configure", strings.NewReader(`{}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("empty configure: expected 200, got %d", rec.Code)
	}
	if h.billingReporter() != nil {
		t.Fatal("metering should be disabled after empty configure")
	}
}

func TestRestorePersistedBilling(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "billing-config.json")
	if err := os.WriteFile(cfgPath,
		[]byte(`{"billing_account_id":"acct-9","usage_report_url":"https://m","usage_report_token":"t","billing_model":"qwen36-35b-a3b-fp8"}`),
		0o600); err != nil {
		t.Fatal(err)
	}
	h := New(&config.Config{ModelName: "m", BillingConfigFile: cfgPath}, nil)
	if h.billingReporter() != nil {
		t.Fatal("reporter should be nil before restore")
	}
	h.StartBilling(context.Background())
	h.RestorePersistedBilling()
	if h.billingReporter() == nil {
		t.Fatal("reporter should be installed after restore")
	}
}
