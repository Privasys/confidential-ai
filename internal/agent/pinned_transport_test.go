// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package agent

import (
	"net/http"
	"testing"
)

func TestPinnedEnclaveTransport(t *testing.T) {
	base := NewRATLSTransport()
	servers := []Server{
		{Name: "a", BaseURL: "https://A.Apps.Privasys.org", ExpectedDigest: "AB12CD"},
		{Name: "ext", BaseURL: "https://mcp.example.com"}, // external: no pin
	}
	pinned, ok := PinnedEnclaveTransport(base, servers).(*RATLSTransport)
	if !ok {
		t.Fatal("expected a *RATLSTransport")
	}
	if pinned == base {
		t.Fatal("must clone, not mutate the shared base transport")
	}
	if base.ExpectedDigests != nil {
		t.Fatal("base transport must stay unpinned")
	}
	if got := pinned.ExpectedDigests["a.apps.privasys.org"]; got != "ab12cd" {
		t.Fatalf("digest for a.apps.privasys.org = %q, want ab12cd", got)
	}
	if _, exists := pinned.ExpectedDigests["mcp.example.com"]; exists {
		t.Fatal("external host must not be pinned")
	}

	// Non-RA-TLS base (dev fallback) passes through untouched.
	var plain http.RoundTripper = http.DefaultTransport
	if out := PinnedEnclaveTransport(plain, servers); out != plain {
		t.Fatal("non-RA-TLS base must be returned unchanged")
	}

	// No enclave servers → base returned as-is.
	if out := PinnedEnclaveTransport(base, servers[1:]); out != http.RoundTripper(base) {
		t.Fatal("no digests to pin → base unchanged")
	}
}
