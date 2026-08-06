package agent

import (
	"encoding/hex"
	"strings"
	"testing"

	rc "enclave-os-mini/clients/go/ratls"
)

func depSetWith(entries ...rc.DependencyEntry) *DepSet {
	d := &DepSet{}
	d.set = rc.DependencySet{Entries: entries}
	d.loaded = true
	return d
}

func peerWithAppID(appIDHex string) rc.CertInfo {
	raw, _ := hex.DecodeString(appIDHex)
	return rc.CertInfo{CustomOids: []rc.OidExtension{{OID: rc.OidWorkloadAppID, Value: raw}}}
}

// TestVerifyPeerDisabledIsNoop: off platform (nothing loaded) the gate must
// not reject anything — the legacy per-host digest pins stay in charge.
func TestVerifyPeerDisabledIsNoop(t *testing.T) {
	d := &DepSet{}
	if err := d.VerifyPeer(peerWithAppID("aa"+strings.Repeat("bb", 15)), rc.TeeTypeTDX, false); err != nil {
		t.Fatalf("disabled set must be a no-op, got %v", err)
	}
	if d.Enabled() {
		t.Fatal("empty set must report disabled")
	}
	if d.Fold() != "" {
		t.Fatal("empty set must have no fold")
	}
}

// TestVerifyPeerUndeclaredRefused pins the fail-closed core: a peer that is
// neither declared nor grant-pinned cannot receive tool data, even though it
// attested successfully. This is what stops a catalogue (or a compromised
// control plane) redirecting tool traffic to an unapproved host.
func TestVerifyPeerUndeclaredRefused(t *testing.T) {
	declared := strings.Repeat("11", 16)
	other := strings.Repeat("22", 16)
	d := depSetWith(rc.DependencyEntry{AppID: declared})

	err := d.VerifyPeer(peerWithAppID(other), rc.TeeTypeTDX, false)
	if err == nil {
		t.Fatal("an undeclared, ungranted peer must be refused")
	}
	if !strings.Contains(err.Error(), other) {
		t.Errorf("error should name the offending app-id, got: %v", err)
	}
}

// TestVerifyPeerGrantPinnedPassesThrough: user tools ride a signed grant and
// are deliberately NOT in the 6.1 set (the set is per-app and durable; grants
// are per-user and per-session). They must pass this gate and be enforced by
// their grant digest instead.
func TestVerifyPeerGrantPinnedPassesThrough(t *testing.T) {
	d := depSetWith(rc.DependencyEntry{AppID: strings.Repeat("11", 16)})
	if err := d.VerifyPeer(peerWithAppID(strings.Repeat("33", 16)), rc.TeeTypeTDX, true); err != nil {
		t.Fatalf("grant-pinned peer must pass the declared-set gate, got %v", err)
	}
}

// TestVerifyPeerDeclaredIsMatched: a declared peer is handed to the SDK
// matcher, which fails closed on an entry that pins no measurement — proving
// we do not accept a declared app-id on name alone.
func TestVerifyPeerDeclaredIsMatched(t *testing.T) {
	declared := strings.Repeat("11", 16)
	d := depSetWith(rc.DependencyEntry{AppID: declared}) // no measurements
	err := d.VerifyPeer(peerWithAppID(declared), rc.TeeTypeTDX, false)
	if err == nil {
		t.Fatal("a declared entry with no measurement must fail closed")
	}
	if !strings.Contains(err.Error(), "fail closed") {
		t.Errorf("expected the SDK matcher's fail-closed error, got: %v", err)
	}
}

// TestVerifyPeerNoAppIDRefused: a peer whose leaf carries no app-id (OID 3.6)
// cannot be matched to any entry.
func TestVerifyPeerNoAppIDRefused(t *testing.T) {
	d := depSetWith(rc.DependencyEntry{AppID: strings.Repeat("11", 16)})
	if err := d.VerifyPeer(rc.CertInfo{}, rc.TeeTypeTDX, false); err == nil {
		t.Fatal("a peer with no app-id must be refused when not grant-pinned")
	}
	// ...but a grant-pinned peer without an app-id is still allowed: an
	// external/user tool is pinned by its digest, not by a platform app id.
	if err := d.VerifyPeer(rc.CertInfo{}, rc.TeeTypeTDX, true); err != nil {
		t.Fatalf("grant-pinned peer without app-id must pass, got %v", err)
	}
}

// TestFoldChangesWithTheSet: the fold commits to WHICH tools were reachable,
// so it must move when the set does (it is stamped into every response).
func TestFoldChangesWithTheSet(t *testing.T) {
	a := depSetWith(rc.DependencyEntry{AppID: strings.Repeat("11", 16)})
	b := depSetWith(
		rc.DependencyEntry{AppID: strings.Repeat("11", 16)},
		rc.DependencyEntry{AppID: strings.Repeat("22", 16)},
	)
	if a.Fold() == "" || b.Fold() == "" {
		t.Fatal("a non-empty set must produce a fold")
	}
	if a.Fold() == b.Fold() {
		t.Fatal("adding a dependency must change the fold")
	}
}
