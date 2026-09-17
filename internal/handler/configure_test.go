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
	"time"

	"github.com/privasys/confidential-ai/internal/config"
)

// configureAppID and configureOwnerRole identify the app these tests
// configure. The role is the canonical hex form the IdP grants.
const (
	configureAppID     = "3a545cb7-740e-4d31-839b-7341359631a2"
	configureOwnerRole = "privasys-platform:app:3a545cb7740e4d31839b7341359631a2:owner"
)

// newConfigureHandler builds a Handler with billing persistence pointed at a
// temp file and the reporter loop context started, as main.go does.
//
// It also wires a real verifier and returns a mint function. These tests are
// about configure SEMANTICS, not auth, but /configure is owner-gated and no
// longer has an unauthenticated mode to fall through — removing that was the
// point of the 2026-09-17 LOAD_TOKEN fix. So they now authenticate properly
// rather than relying on the gate being open.
func newConfigureHandler(t *testing.T) (*Handler, string, func() string) {
	t.Helper()
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "billing-config.json")
	issuer, mint := jwksTestIDP(t)
	h := New(&config.Config{
		ModelName:         "test-model",
		BillingConfigFile: cfgPath,
		OIDCIssuer:        issuer,
		OIDCAudience:      "privasys-platform",
		AppID:             configureAppID,
	}, nil)
	h.oidcVerifier = NewOIDCVerifier(issuer, "")
	h.StartBilling(context.Background())

	ownerToken := func() string {
		return mint(map[string]any{
			"iss":   issuer,
			"sub":   "owner-sub",
			"exp":   float64(time.Now().Add(time.Hour).Unix()),
			"roles": []string{configureOwnerRole},
		})
	}
	return h, cfgPath, ownerToken
}

// configureRequest builds an owner-authenticated POST /configure.
func configureRequest(ownerToken func() string, body string) *http.Request {
	req := httptest.NewRequest("POST", "/configure", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+ownerToken())
	return req
}

func TestConfigureEnablesMeteringAndPersists(t *testing.T) {
	h, cfgPath, ownerToken := newConfigureHandler(t)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	if h.billingReporter() != nil {
		t.Fatal("metering should be disabled before configure")
	}

	body := `{"billing_account_id":"acct-1","usage_report_url":"https://m/api","usage_report_token":"tok","billing_model":"qwen36-35b-a3b-fp8"}`
	req := configureRequest(ownerToken, body)
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
	h, _, ownerToken := newConfigureHandler(t)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// account without url -> 400, no reporter installed.
	req := configureRequest(ownerToken, `{"billing_account_id":"acct-1"}`)
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
	h, _, ownerToken := newConfigureHandler(t)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// First enable.
	mux.ServeHTTP(httptest.NewRecorder(),
		configureRequest(ownerToken, `{"billing_account_id":"a","usage_report_url":"https://m"}`))
	if h.billingReporter() == nil {
		t.Fatal("metering should be enabled")
	}
	// Then an empty payload disables.
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, configureRequest(ownerToken, `{}`))
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
