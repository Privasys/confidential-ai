package agent

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"
)

// GrantVerifier verifies tool-grants minted by the chat back-end (ES256
// JWS) using ONLY the standard library — the enclave deliberately carries
// no JWT/JWKS dependency. A grant lists the MCP tool servers a user is
// authorised to use; the proxy unions them with the configured catalogue
// for a single request. Verifying the signature here is what lets the
// browser carry the grant without being trusted to name a server URL.
//
// Public keys are fetched from the chat back-end JWKS URL and cached by key
// id; an unknown kid triggers at most one refresh per refreshWindow.
type GrantVerifier struct {
	jwksURL  string
	audience string // required aud (this instance id); empty skips the check
	client   *http.Client

	mu        sync.RWMutex
	keys      map[string]*ecdsa.PublicKey
	lastFetch time.Time
}

const refreshWindow = 10 * time.Second

// NewGrantVerifier returns a verifier for grants signed by the chat
// back-end whose JWKS is served at jwksURL.
func NewGrantVerifier(jwksURL, audience string) *GrantVerifier {
	return &GrantVerifier{
		jwksURL:  jwksURL,
		audience: audience,
		client:   &http.Client{Timeout: 10 * time.Second},
		keys:     map[string]*ecdsa.PublicKey{},
	}
}

// GrantServers verifies the compact JWS and returns the authorised tool
// servers it carries. A valid grant with no tools yields (nil, nil).
func (v *GrantVerifier) GrantServers(ctx context.Context, token string) ([]Server, error) {
	claims, err := v.verify(ctx, token)
	if err != nil {
		return nil, err
	}
	out := make([]Server, 0, len(claims.Tools))
	for _, t := range claims.Tools {
		if t.Name == "" || t.BaseURL == "" {
			continue
		}
		out = append(out, Server{
			Name:                     t.Name,
			BaseURL:                  strings.TrimRight(t.BaseURL, "/"),
			Transport:                t.Transport,
			AuthMode:                 t.AuthMode,
			AuthAudience:             t.AuthAudience,
			AuthScopes:               t.AuthScopes,
			RequiresUserConfirmation: t.RequiresUserConfirmation,
		})
	}
	return out, nil
}

type grantTool struct {
	Name                     string   `json:"name"`
	Transport                string   `json:"transport"`
	BaseURL                  string   `json:"base_url"`
	AuthMode                 string   `json:"auth_mode"`
	AuthAudience             string   `json:"auth_audience"`
	AuthScopes               []string `json:"auth_scopes"`
	ExpectedDigest           string   `json:"expected_digest"`
	Verified                 bool     `json:"verified"`
	RequiresUserConfirmation bool     `json:"requires_user_confirmation"`
}

type grantClaims struct {
	Exp   int64       `json:"exp"`
	Aud   audClaim    `json:"aud"`
	Tools []grantTool `json:"tools"`
}

// audClaim accepts both the string and []string JSON encodings of `aud`.
type audClaim []string

func (a *audClaim) UnmarshalJSON(b []byte) error {
	var s string
	if json.Unmarshal(b, &s) == nil {
		*a = audClaim{s}
		return nil
	}
	var arr []string
	if err := json.Unmarshal(b, &arr); err != nil {
		return err
	}
	*a = arr
	return nil
}

func (a audClaim) contains(x string) bool {
	for _, v := range a {
		if v == x {
			return true
		}
	}
	return false
}

func (v *GrantVerifier) verify(ctx context.Context, token string) (*grantClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("grant: not a compact JWS")
	}
	var hdr struct {
		Alg string `json:"alg"`
		Kid string `json:"kid"`
	}
	hb, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return nil, fmt.Errorf("grant: bad header: %w", err)
	}
	if err := json.Unmarshal(hb, &hdr); err != nil {
		return nil, fmt.Errorf("grant: bad header json: %w", err)
	}
	if hdr.Alg != "ES256" {
		return nil, fmt.Errorf("grant: unexpected alg %q", hdr.Alg)
	}

	key, err := v.key(ctx, hdr.Kid)
	if err != nil {
		return nil, err
	}

	sig, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, fmt.Errorf("grant: bad signature: %w", err)
	}
	if len(sig) != 64 {
		return nil, fmt.Errorf("grant: ES256 signature must be 64 bytes")
	}
	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	r := new(big.Int).SetBytes(sig[:32])
	s := new(big.Int).SetBytes(sig[32:])
	if !ecdsa.Verify(key, digest[:], r, s) {
		return nil, fmt.Errorf("grant: signature verification failed")
	}

	pb, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("grant: bad payload: %w", err)
	}
	var claims grantClaims
	if err := json.Unmarshal(pb, &claims); err != nil {
		return nil, fmt.Errorf("grant: bad payload json: %w", err)
	}
	if claims.Exp == 0 || time.Now().After(time.Unix(claims.Exp, 0).Add(30*time.Second)) {
		return nil, fmt.Errorf("grant: expired")
	}
	if v.audience != "" && !claims.Aud.contains(v.audience) {
		return nil, fmt.Errorf("grant: audience mismatch")
	}
	return &claims, nil
}

// key returns the public key for kid, refreshing the JWKS at most once per
// refreshWindow when the kid is unknown.
func (v *GrantVerifier) key(ctx context.Context, kid string) (*ecdsa.PublicKey, error) {
	v.mu.RLock()
	k := v.keys[kid]
	v.mu.RUnlock()
	if k != nil {
		return k, nil
	}
	if err := v.refresh(ctx); err != nil {
		return nil, err
	}
	v.mu.RLock()
	k = v.keys[kid]
	v.mu.RUnlock()
	if k == nil {
		return nil, fmt.Errorf("grant: unknown key id %q", kid)
	}
	return k, nil
}

func (v *GrantVerifier) refresh(ctx context.Context) error {
	v.mu.Lock()
	if time.Since(v.lastFetch) < refreshWindow && len(v.keys) > 0 {
		v.mu.Unlock()
		return nil
	}
	v.mu.Unlock()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.jwksURL, nil)
	if err != nil {
		return err
	}
	resp, err := v.client.Do(req)
	if err != nil {
		return fmt.Errorf("grant: fetch jwks: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("grant: jwks status %d", resp.StatusCode)
	}
	var doc struct {
		Keys []struct {
			Kty string `json:"kty"`
			Crv string `json:"crv"`
			Kid string `json:"kid"`
			X   string `json:"x"`
			Y   string `json:"y"`
		} `json:"keys"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&doc); err != nil {
		return fmt.Errorf("grant: decode jwks: %w", err)
	}
	keys := map[string]*ecdsa.PublicKey{}
	for _, jwk := range doc.Keys {
		if jwk.Kty != "EC" || jwk.Crv != "P-256" {
			continue
		}
		xb, err1 := base64.RawURLEncoding.DecodeString(jwk.X)
		yb, err2 := base64.RawURLEncoding.DecodeString(jwk.Y)
		if err1 != nil || err2 != nil {
			continue
		}
		keys[jwk.Kid] = &ecdsa.PublicKey{
			Curve: elliptic.P256(),
			X:     new(big.Int).SetBytes(xb),
			Y:     new(big.Int).SetBytes(yb),
		}
	}
	if len(keys) == 0 {
		return fmt.Errorf("grant: jwks had no usable EC P-256 keys")
	}
	v.mu.Lock()
	v.keys = keys
	v.lastFetch = time.Now()
	v.mu.Unlock()
	return nil
}

// MergeServers returns base plus any servers in extra whose name is not
// already present in base (configured catalogue wins on name collision).
func MergeServers(base, extra []Server) []Server {
	seen := make(map[string]bool, len(base))
	for _, s := range base {
		seen[s.Name] = true
	}
	out := append([]Server(nil), base...)
	for _, s := range extra {
		if !seen[s.Name] {
			out = append(out, s)
			seen[s.Name] = true
		}
	}
	return out
}
