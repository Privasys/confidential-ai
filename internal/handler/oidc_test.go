package handler

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"math/big"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/privasys/confidential-ai/internal/config"
)

// jwksTestIDP spins up a minimal OIDC issuer: discovery + JWKS for one
// ES256 key, plus a mint helper. Returns the issuer URL and minter.
func jwksTestIDP(t *testing.T) (issuer string, mint func(claims map[string]any) string) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), crand.Reader)
	if err != nil {
		t.Fatalf("gen key: %v", err)
	}
	b32 := func(i *big.Int) string {
		b := i.Bytes()
		if len(b) < 32 {
			pad := make([]byte, 32)
			copy(pad[32-len(b):], b)
			b = pad
		}
		return base64.RawURLEncoding.EncodeToString(b)
	}

	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"jwks_uri": srv.URL + "/jwks"})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"keys": []map[string]string{{
			"kty": "EC", "crv": "P-256", "kid": "k1", "alg": "ES256", "use": "sig",
			"x": b32(key.X), "y": b32(key.Y),
		}}})
	})

	mint = func(claims map[string]any) string {
		hdr := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"ES256","kid":"k1","typ":"at+jwt"}`))
		pj, _ := json.Marshal(claims)
		pl := base64.RawURLEncoding.EncodeToString(pj)
		signing := hdr + "." + pl
		sum := sha256.Sum256([]byte(signing))
		r, s, err := ecdsa.Sign(crand.Reader, key, sum[:])
		if err != nil {
			t.Fatalf("sign: %v", err)
		}
		sig := make([]byte, 64)
		r.FillBytes(sig[:32])
		s.FillBytes(sig[32:])
		return signing + "." + base64.RawURLEncoding.EncodeToString(sig)
	}
	return srv.URL, mint
}

func TestOIDCVerifier_ManagerRole(t *testing.T) {
	issuer, mint := jwksTestIDP(t)
	v := NewOIDCVerifier(issuer, "")
	tok := mint(map[string]any{
		"iss":   issuer,
		"sub":   "svc-account",
		"exp":   float64(time.Now().Add(time.Hour).Unix()),
		"roles": []string{"privasys-platform:manager"},
	})
	claims, err := v.Verify(context.Background(), tok)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if claims.Subject != "svc-account" {
		t.Errorf("sub = %q", claims.Subject)
	}
	if !claims.HasRole("privasys-platform:manager") {
		t.Errorf("manager role not extracted: %v", claims.Roles)
	}
}

func TestOIDCVerifier_RejectsWrongIssuerAndExpiry(t *testing.T) {
	issuer, mint := jwksTestIDP(t)
	v := NewOIDCVerifier(issuer, "")

	bad := mint(map[string]any{"iss": "https://evil", "exp": float64(time.Now().Add(time.Hour).Unix())})
	if _, err := v.Verify(context.Background(), bad); err == nil {
		t.Error("expected issuer mismatch rejection")
	}
	expired := mint(map[string]any{"iss": issuer, "exp": float64(time.Now().Add(-time.Hour).Unix())})
	if _, err := v.Verify(context.Background(), expired); err == nil {
		t.Error("expected expiry rejection")
	}
}

func TestRequireLoadAuth_OIDCManager(t *testing.T) {
	issuer, mint := jwksTestIDP(t)
	h := &Handler{
		cfg:          &config.Config{ManagerRole: "privasys-platform:manager", OIDCIssuer: issuer},
		oidcVerifier: NewOIDCVerifier(issuer, ""),
	}
	called := false
	gate := h.requireLoadAuth(func(http.ResponseWriter, *http.Request) { called = true })

	// Manager token → allowed.
	tok := mint(map[string]any{"iss": issuer, "sub": "svc", "exp": float64(time.Now().Add(time.Hour).Unix()), "roles": []string{"privasys-platform:manager"}})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/models/load", nil)
	req.Header.Set("Authorization", "Bearer "+tok)
	gate(rec, req)
	if !called || rec.Code != http.StatusOK {
		t.Fatalf("manager rejected: called=%v code=%d", called, rec.Code)
	}

	// Non-manager token → 403.
	called = false
	weak := mint(map[string]any{"iss": issuer, "sub": "u", "exp": float64(time.Now().Add(time.Hour).Unix()), "roles": []string{"privasys-platform:monitoring"}})
	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/v1/models/load", nil)
	req.Header.Set("Authorization", "Bearer "+weak)
	gate(rec, req)
	if called || rec.Code != http.StatusForbidden {
		t.Fatalf("non-manager not forbidden: called=%v code=%d", called, rec.Code)
	}

	// No token → 401.
	rec = httptest.NewRecorder()
	gate(rec, httptest.NewRequest("POST", "/v1/models/load", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("missing token: code=%d", rec.Code)
	}
}

func TestRequireLoadAuth_LegacyFallbackAndDevMode(t *testing.T) {
	// Legacy static token, OIDC disabled.
	h := &Handler{cfg: &config.Config{LoadToken: "s3cret"}}
	called := false
	gate := h.requireLoadAuth(func(http.ResponseWriter, *http.Request) { called = true })
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/models/load", nil)
	req.Header.Set("Authorization", "Bearer s3cret")
	gate(rec, req)
	if !called || rec.Code != http.StatusOK {
		t.Fatalf("legacy token rejected: called=%v code=%d", called, rec.Code)
	}
	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "/v1/models/load", nil)
	req.Header.Set("Authorization", "Bearer wrong")
	gate(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong legacy token: code=%d", rec.Code)
	}

	// Dev mode: nothing configured → open.
	h = &Handler{cfg: &config.Config{}}
	called = false
	gate = h.requireLoadAuth(func(http.ResponseWriter, *http.Request) { called = true })
	gate(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/models/load", nil))
	if !called {
		t.Fatal("dev mode should allow without auth")
	}
}

func TestResolveCaller(t *testing.T) {
	issuer, mint := jwksTestIDP(t)
	h := &Handler{cfg: &config.Config{OIDCIssuer: issuer}, oidcVerifier: NewOIDCVerifier(issuer, "")}
	tok := mint(map[string]any{"iss": issuer, "sub": "user-1", "exp": float64(time.Now().Add(time.Hour).Unix())})

	// X-App-Auth (proxied path).
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("X-App-Auth", tok)
	if sub, err := h.resolveCaller(r); err != nil || sub != "user-1" {
		t.Fatalf("X-App-Auth: sub=%q err=%v", sub, err)
	}
	// Authorization bearer (direct OpenAI client).
	r = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	r.Header.Set("Authorization", "Bearer "+tok)
	if sub, err := h.resolveCaller(r); err != nil || sub != "user-1" {
		t.Fatalf("Authorization: sub=%q err=%v", sub, err)
	}
	// Anonymous.
	if sub, err := h.resolveCaller(httptest.NewRequest("POST", "/", nil)); err != nil || sub != "" {
		t.Fatalf("anon: sub=%q err=%v", sub, err)
	}
	// Invalid credential present.
	r = httptest.NewRequest("POST", "/", nil)
	r.Header.Set("Authorization", "Bearer not.a.jwt")
	if sub, err := h.resolveCaller(r); err == nil || sub != "" {
		t.Fatalf("invalid: expected error, sub=%q err=%v", sub, err)
	}
}

func TestAuthorizeInference(t *testing.T) {
	issuer, mint := jwksTestIDP(t)
	tok := mint(map[string]any{"iss": issuer, "sub": "user-2", "exp": float64(time.Now().Add(time.Hour).Unix())})
	h := &Handler{cfg: &config.Config{OIDCIssuer: issuer}, oidcVerifier: NewOIDCVerifier(issuer, "")}

	// Flag off + anonymous → allowed, no caller.
	r2, ok := h.authorizeInference(httptest.NewRecorder(), httptest.NewRequest("POST", "/", nil))
	if !ok || callerFromContext(r2.Context()) != "" {
		t.Fatalf("flag off anon: ok=%v caller=%q", ok, callerFromContext(r2.Context()))
	}
	// Flag off + valid → allowed, caller attributed.
	r := httptest.NewRequest("POST", "/", nil)
	r.Header.Set("X-App-Auth", tok)
	r2, ok = h.authorizeInference(httptest.NewRecorder(), r)
	if !ok || callerFromContext(r2.Context()) != "user-2" {
		t.Fatalf("flag off valid: ok=%v caller=%q", ok, callerFromContext(r2.Context()))
	}
	// Flag on + anonymous → 401.
	h.cfg.InferenceAuthRequired = true
	rec := httptest.NewRecorder()
	if _, ok := h.authorizeInference(rec, httptest.NewRequest("POST", "/", nil)); ok || rec.Code != http.StatusUnauthorized {
		t.Fatalf("flag on anon: ok=%v code=%d", ok, rec.Code)
	}
	// Flag on + valid → allowed, caller attributed.
	r = httptest.NewRequest("POST", "/", nil)
	r.Header.Set("Authorization", "Bearer "+tok)
	r2, ok = h.authorizeInference(httptest.NewRecorder(), r)
	if !ok || callerFromContext(r2.Context()) != "user-2" {
		t.Fatalf("flag on valid: ok=%v caller=%q", ok, callerFromContext(r2.Context()))
	}
}

func TestRequireLoadAuth_OIDCConfiguredLegacyFallback(t *testing.T) {
	issuer, _ := jwksTestIDP(t)
	h := &Handler{
		cfg:          &config.Config{ManagerRole: "privasys-platform:manager", OIDCIssuer: issuer, LoadToken: "s3cret"},
		oidcVerifier: NewOIDCVerifier(issuer, ""),
	}
	called := false
	gate := h.requireLoadAuth(func(http.ResponseWriter, *http.Request) { called = true })
	// A non-JWT legacy token fails OIDC verify (malformed) then matches the
	// static fallback.
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/models/load", nil)
	req.Header.Set("Authorization", "Bearer s3cret")
	gate(rec, req)
	if !called || rec.Code != http.StatusOK {
		t.Fatalf("legacy fallback under OIDC rejected: called=%v code=%d", called, rec.Code)
	}
}
