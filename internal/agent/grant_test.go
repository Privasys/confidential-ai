package agent

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// mintGrant hand-rolls an ES256 JWS (no JWT dependency, mirroring the
// verifier) so the test exercises the real signature path.
func mintGrant(t *testing.T, key *ecdsa.PrivateKey, kid string, claims map[string]any) string {
	t.Helper()
	hdr, _ := json.Marshal(map[string]string{"alg": "ES256", "typ": "JWT", "kid": kid})
	pl, _ := json.Marshal(claims)
	signingInput := b64(hdr) + "." + b64(pl)
	digest := sha256.Sum256([]byte(signingInput))
	r, s, err := ecdsa.Sign(rand.Reader, key, digest[:])
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	sig := make([]byte, 64)
	r.FillBytes(sig[:32])
	s.FillBytes(sig[32:])
	return signingInput + "." + base64.RawURLEncoding.EncodeToString(sig)
}

func b64(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }

func jwksServer(t *testing.T, key *ecdsa.PrivateKey, kid string) *httptest.Server {
	t.Helper()
	pub := key.PublicKey
	doc := map[string]any{"keys": []map[string]any{{
		"kty": "EC", "crv": "P-256", "kid": kid,
		"x": jwkCoord(pub.X), "y": jwkCoord(pub.Y),
	}}}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(doc)
	}))
}

func jwkCoord(i *big.Int) string {
	b := make([]byte, 32)
	i.FillBytes(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func TestGrantServers_RoundTrip(t *testing.T) {
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	srv := jwksServer(t, key, "k1")
	defer srv.Close()

	v := NewGrantVerifier(srv.URL, "inst-123")
	tok := mintGrant(t, key, "k1", map[string]any{
		"iss": "chat", "sub": "user-1", "aud": "inst-123",
		"exp": time.Now().Add(5 * time.Minute).Unix(),
		"tools": []map[string]any{{
			"name": "myrag", "transport": "mcp_sse",
			"base_url": "https://myrag.apps.privasys.org", "auth_mode": "exchange",
			"verified": true,
		}},
	})
	servers, err := v.GrantServers(context.Background(), tok)
	if err != nil {
		t.Fatalf("GrantServers: %v", err)
	}
	if len(servers) != 1 || servers[0].Name != "myrag" || servers[0].BaseURL != "https://myrag.apps.privasys.org" {
		t.Fatalf("unexpected servers: %+v", servers)
	}
}

func TestGrant_Rejections(t *testing.T) {
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	other, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	srv := jwksServer(t, key, "k1")
	defer srv.Close()
	v := NewGrantVerifier(srv.URL, "inst-123")

	t.Run("expired", func(t *testing.T) {
		tok := mintGrant(t, key, "k1", map[string]any{
			"aud": "inst-123", "exp": time.Now().Add(-time.Hour).Unix(),
		})
		if _, err := v.GrantServers(context.Background(), tok); err == nil {
			t.Fatal("expected expired grant to be rejected")
		}
	})

	t.Run("wrong audience", func(t *testing.T) {
		tok := mintGrant(t, key, "k1", map[string]any{
			"aud": "someone-else", "exp": time.Now().Add(time.Hour).Unix(),
		})
		if _, err := v.GrantServers(context.Background(), tok); err == nil {
			t.Fatal("expected audience mismatch to be rejected")
		}
	})

	t.Run("bad signature", func(t *testing.T) {
		tok := mintGrant(t, other, "k1", map[string]any{
			"aud": "inst-123", "exp": time.Now().Add(time.Hour).Unix(),
		})
		if _, err := v.GrantServers(context.Background(), tok); err == nil {
			t.Fatal("expected bad signature to be rejected")
		}
	})
}

func TestMergeServers(t *testing.T) {
	base := []Server{{Name: "admin1"}, {Name: "shared"}}
	extra := []Server{{Name: "shared"}, {Name: "user1"}}
	got := MergeServers(base, extra)
	if len(got) != 3 {
		t.Fatalf("want 3 merged servers, got %d: %+v", len(got), got)
	}
	// configured (base) wins on name collision: 'shared' appears once.
	count := 0
	for _, s := range got {
		if s.Name == "shared" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected 'shared' once, got %d", count)
	}
}
