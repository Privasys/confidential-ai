package handler

import (
	"testing"

	"github.com/privasys/confidential-ai/internal/config"
)

// The role strings must be byte-identical to what the IdP grants
// (<audience>:app:<32-hex>:owner|admin), or an owner can never configure the
// instance. This is the same hex-vs-hyphen trap that bit the enclave manager.
func TestLoadRoles(t *testing.T) {
	h := &Handler{cfg: &config.Config{
		OIDCAudience: "privasys-platform",
		AppID:        "3a545cb7-740e-4d31-839b-7341359631a2", // hyphenated UUID, as injected
	}}
	got := h.loadRoles()
	want := map[string]bool{
		"privasys-platform:app:3a545cb7740e4d31839b7341359631a2:owner": true,
		"privasys-platform:app:3a545cb7740e4d31839b7341359631a2:admin": true,
	}
	if len(got) != len(want) {
		t.Fatalf("got %d roles, want %d: %v", len(got), len(want), got)
	}
	for _, r := range got {
		if !want[r] {
			t.Errorf("unexpected role %q (hyphens not stripped, or wrong shape)", r)
		}
	}
}

func TestLoadRolesFallbackAudience(t *testing.T) {
	// Empty OIDC_AUDIENCE must still build the canonical platform-audience role.
	h := &Handler{cfg: &config.Config{
		AppID: "3a545cb7-740e-4d31-839b-7341359631a2",
	}}
	found := false
	for _, r := range h.loadRoles() {
		if r == "privasys-platform:app:3a545cb7740e4d31839b7341359631a2:owner" {
			found = true
		}
	}
	if !found {
		t.Fatal("owner role not built with the default platform audience")
	}
}

func TestLoadRolesNoAppID(t *testing.T) {
	// Without an app id there is nothing to owner-gate on: fail closed (nil).
	h := &Handler{cfg: &config.Config{}}
	if got := h.loadRoles(); len(got) != 0 {
		t.Fatalf("expected no roles without an app id (fail closed), got %v", got)
	}
}
