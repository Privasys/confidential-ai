package handler

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// countingIDP is an issuer that counts key-set fetches and can make them slow.
func countingIDP(t *testing.T) (issuer string, mint func(kid string) string, fetches *atomic.Int64, slow *atomic.Bool) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), crand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	fetches, slow = &atomic.Int64{}, &atomic.Bool{}
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"jwks_uri": srv.URL + "/jwks"})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		fetches.Add(1)
		if slow.Load() {
			time.Sleep(2 * time.Second)
		}
		x, y := make([]byte, 32), make([]byte, 32)
		key.X.FillBytes(x)
		key.Y.FillBytes(y)
		json.NewEncoder(w).Encode(map[string]any{"keys": []map[string]string{{
			"kty": "EC", "crv": "P-256", "kid": "k1", "alg": "ES256", "use": "sig",
			"x": base64.RawURLEncoding.EncodeToString(x), "y": base64.RawURLEncoding.EncodeToString(y),
		}}})
	})
	mint = func(kid string) string {
		hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"ES256","kid":"` + kid + `","typ":"at+jwt"}`))
		pj, _ := json.Marshal(map[string]any{"iss": srv.URL, "sub": "u", "exp": time.Now().Add(time.Hour).Unix()})
		signing := hdr + "." + base64.RawURLEncoding.EncodeToString(pj)
		sum := sha256.Sum256([]byte(signing))
		r, s, _ := ecdsa.Sign(crand.Reader, key, sum[:])
		sig := make([]byte, 64)
		r.FillBytes(sig[:32])
		s.FillBytes(sig[32:])
		return signing + "." + base64.RawURLEncoding.EncodeToString(sig)
	}
	return srv.URL, mint, fetches, slow
}

// TestUnknownKidsFetchAtMostOnce proves a stream of tokens naming unknown
// keys costs at most one key-set fetch per interval.
func TestUnknownKidsFetchAtMostOnce(t *testing.T) {
	issuer, mint, fetches, _ := countingIDP(t)
	v := NewOIDCVerifier(issuer, "")
	if _, err := v.Verify(context.Background(), mint("k1")); err != nil {
		t.Fatalf("valid token: %v", err)
	}
	before := fetches.Load()
	for i := 0; i < 50; i++ {
		_, _ = v.Verify(context.Background(), mint("unknown-"+string(rune('a'+i%26))))
	}
	if got := fetches.Load() - before; got > 1 {
		t.Fatalf("%d key-set fetches for 50 unknown kids, want at most 1", got)
	}
}

// TestKnownKeyDoesNotWaitOnAFetch proves a token signed with a held key is
// verified while a slow fetch for an unknown key is in progress.
func TestKnownKeyDoesNotWaitOnAFetch(t *testing.T) {
	issuer, mint, _, slow := countingIDP(t)
	v := NewOIDCVerifier(issuer, "")
	good := mint("k1")
	if _, err := v.Verify(context.Background(), good); err != nil {
		t.Fatalf("valid token: %v", err)
	}
	v.mu.Lock()
	v.lastAttempt = time.Time{} // allow the next unknown kid to fetch
	v.mu.Unlock()
	slow.Store(true)
	go func() { _, _ = v.Verify(context.Background(), mint("unknown")) }()
	time.Sleep(100 * time.Millisecond) // the slow fetch is under way
	start := time.Now()
	if _, err := v.Verify(context.Background(), good); err != nil {
		t.Fatalf("valid token during a fetch: %v", err)
	}
	if took := time.Since(start); took > 500*time.Millisecond {
		t.Fatalf("a valid token waited %s on an unrelated fetch", took)
	}
}

// TestCancelledCallerStillRefreshes proves a caller that cancels its request
// cannot abort the key-set fetch others rely on.
func TestCancelledCallerStillRefreshes(t *testing.T) {
	issuer, mint, fetches, _ := countingIDP(t)
	v := NewOIDCVerifier(issuer, "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := v.Verify(ctx, mint("k1")); err != nil {
		t.Fatalf("a valid token with a cancelled context: %v", err)
	}
	if fetches.Load() != 1 {
		t.Fatalf("fetches = %d, want 1", fetches.Load())
	}
}

// TestHeldKeysExpireWhileTheIssuerIsDown proves a held key stops verifying
// oidcMaxStale after it was fetched, however long the issuer stays away.
func TestHeldKeysExpireWhileTheIssuerIsDown(t *testing.T) {
	issuer, mint, _, _ := countingIDP(t)
	v := NewOIDCVerifier(issuer, "")
	good := mint("k1")
	if _, err := v.Verify(context.Background(), good); err != nil {
		t.Fatalf("valid token: %v", err)
	}
	v.issuer = "http://127.0.0.1:1" // the issuer is now unreachable
	v.mu.Lock()
	v.fetchedAt = time.Now().Add(-oidcKeysTTL - time.Minute)
	v.lastAttempt = time.Time{}
	v.mu.Unlock()
	if _, err := v.signingKey(context.Background(), "k1", "ES256"); err != nil {
		t.Fatalf("a key within the maximum staleness was refused: %v", err)
	}
	v.mu.Lock()
	v.fetchedAt = time.Now().Add(-oidcMaxStale - time.Minute)
	v.lastAttempt = time.Time{}
	v.mu.Unlock()
	if _, err := v.signingKey(context.Background(), "k1", "ES256"); err == nil {
		t.Fatal("a key past the maximum staleness still verified")
	}
}

// TestTokenWithoutExpiryIsRefused proves a token must carry exp.
func TestTokenWithoutExpiryIsRefused(t *testing.T) {
	issuer, mint := jwksTestIDP(t)
	v := NewOIDCVerifier(issuer, "")
	if _, err := v.Verify(context.Background(), mint(map[string]any{"iss": issuer, "sub": "u"})); err == nil {
		t.Fatal("a token without exp was accepted")
	}
}
