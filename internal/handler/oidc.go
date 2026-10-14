package handler

import (
	"context"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/big"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/privasys/confidential-ai/internal/billing"
)

// OIDCVerifier validates platform OIDC bearer tokens offline via JWKS
// discovery, so a privileged call is authorised without the enclave ever
// contacting the platform per request with the payload. It mirrors the
// enclave manager's own verifier (enclave-os-virtual internal/auth): the
// same issuer, `roles` claim, and signature algorithms (RS*, ES*).
//
// Verification is deliberately stdlib-only (like the tool-grant verifier)
// so the enclave carries no JWT dependency.
type OIDCVerifier struct {
	issuer   string
	audience string // required aud; empty skips the check

	client *http.Client

	mu        sync.RWMutex
	keys      map[string]*oidcJWK
	fetchedAt time.Time
}

// OIDCClaims is the subset of a verified token the handler needs.
type OIDCClaims struct {
	Subject string
	Roles   []string
	// SID is the session id the token is bound to (revocation handle). Empty
	// for tokens minted without a session.
	SID string
}

// callerCtxKey carries the verified inference caller's subject through the
// request context so the metering path (per-caller billing) can attribute usage.
type callerCtxKey struct{}

// callerFromContext returns the verified caller subject stashed by the inference
// handlers, or "" when the request was anonymous.
func callerFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(callerCtxKey{}).(string); ok {
		return v
	}
	return ""
}

// relaySubHeader is the wallet-authenticated subject the enclave-os
// session relay asserts toward the app on sealed-transport requests
// (sessionrelay.Middleware). The relay strips any inbound value on every
// path — terminate-leg and RA-TLS-direct alike — before setting its own,
// and the container only receives traffic through the relay, so a
// non-empty value here is always relay-set and carries an
// EncAuth-verified wallet identity. This is the platform's sealed-caller
// convention (chat-service and Drive authenticate the same way): the
// browser's bearer deliberately never crosses the gateway's terminate
// leg, so a sealed request carries NO token — without honouring this
// header, mandatory inference auth 401s every sealed chat turn.
const relaySubHeader = "X-Privasys-Sub"

// The header every stock Anthropic-style client puts its key in: the Anthropic
// SDKs, Claude Code, LangChain's Anthropic client, and dsh's own Messages
// adapter all send x-api-key and none of them send a bearer. Since this app
// serves /v1/messages it has to read the credential where those clients put
// it, or a caller with a perfectly good Privasys key is told to authenticate.
//
// It is the same credential in a different envelope: same verifier, same
// revocation set, same subject. It is NOT a new kind of credential.
//
// Note for callers on the app-to-app path: a credential presented here is
// used INSTEAD of the payer the platform vouched for in a spend token (see
// resolveCaller's ordering), so an app that lets a stray or junk x-api-key
// through now gets 401 where it used to be served on its verified payer.
// That is deliberate, a caller sending a bad key should fail loudly, but it
// makes stripping this header on an app's egress leg a requirement rather
// than hygiene. The attested harness does it in its egress proxy.
const apiKeyHeader = "X-Api-Key"

// Ingress mutual RA-TLS peer identity, asserted by the enclave manager after
// verifying the caller's attested client certificate against this app's
// allowed-callers policy (enclave-os-virtual internal/manager/peerverify.go).
// The manager strips these headers from every request that did not pass that
// verification, so their presence is trustworthy inside the enclave.
const (
	peerVerifiedHeader = "X-Privasys-Peer-Verified"
	peerAppIDHeader    = "X-Privasys-Peer-App-Id"
)

// Spend tokens (acting-subject plan v2). The enclave runtime verifies the
// caller app's spend token and per-request proof on ingress and asserts
// the PAYING USER here: the user who allowed that app to spend their
// credits. The headers sit in the X-Privasys-Peer-* namespace the runtime
// strips from every request it did not verify, so they are trustworthy on
// any runtime, and they need no peer verdict: a non-enclave caller with a
// spend token is as good a payer as an attested one. This is the final form
// of "the user pays"; the on-behalf-of header above is the transitional one
// and is removed once no caller sends it (see the source log lines).
const (
	payerHeader    = "X-Privasys-Peer-Payer"
	payerAppHeader = "X-Privasys-Peer-Payer-App"
	payerSIDHeader = "X-Privasys-Peer-Payer-Sid"
	// payerLedgerHeader is the spend token's ledger handle, forwarded by the
	// runtime: what the platform bills by, naming no one.
	payerLedgerHeader = "X-Privasys-Peer-Payer-Ledger"
)

// payerCtxKey carries the spender app and consent session behind a
// spend-token caller, for the metering path.
type payerCtxKey struct{}

// callerInfoFromContext returns the verified caller plus, for a spend-token
// caller, the app that spends for them and the consent session.
func callerInfoFromContext(ctx context.Context) billing.Caller {
	c := billing.Caller{Sub: callerFromContext(ctx)}
	if p, ok := ctx.Value(payerCtxKey{}).(billing.Caller); ok {
		c.App, c.SID = p.App, p.SID
	}
	return c
}

// resolveCaller extracts the end-user credential from X-App-Auth (the proxied
// path, forwarded by the management-service), the Authorization bearer (a
// direct OpenAI-SDK client) or x-api-key (a direct Anthropic-style client on
// /v1/messages), verifies it against the platform OIDC issuer, and returns its
// subject. Returns ("", nil) when no credential is present and ("", err) when a
// credential is present but invalid. A platform API key is just a long-lived
// signed token, so all three verify through the same path.
//
// Two DIFFERENT tokens in Authorization and x-api-key are refused rather than
// ranked. Both are client-presented credentials for the same direct-client
// role, so there is no principled winner, and picking one is how a caller ends
// up metered as somebody else.
//
// Sealed-transport requests carry no token at all; for those the
// relay-asserted subject (relaySubHeader) is the caller. An explicit token
// always takes precedence — a caller presenting an invalid or revoked token
// is rejected, never silently downgraded to the relay identity.
func (h *Handler) resolveCaller(r *http.Request) (string, error) {
	var bearer string
	if a := r.Header.Get("Authorization"); strings.HasPrefix(a, "Bearer ") {
		bearer = strings.TrimSpace(strings.TrimPrefix(a, "Bearer "))
	}
	apiKey := strings.TrimSpace(r.Header.Get(apiKeyHeader))
	if bearer != "" && apiKey != "" && bearer != apiKey {
		return "", errors.New("conflicting credentials")
	}
	// X-App-Auth stays ahead of both: on the proxied path it is the
	// credential the management-service resolved for this call, and a header
	// the client also happened to send does not displace it.
	tok := strings.TrimSpace(r.Header.Get("X-App-Auth"))
	// A disagreement HERE is reported, not refused. X-App-Auth and a client
	// header are different roles, so unlike the pair above there is a
	// principled winner and no need to fail. But nobody has shown whether the
	// proxied path ever forwards a client credential alongside the one it
	// resolved, so refusing could turn working traffic into 401s on a guess.
	// One line per distinct pair says whether it happens at all, and after
	// that the rule can be generalised on evidence instead (Bertrand,
	// 2026-09-26).
	if client := firstNonEmpty(bearer, apiKey); tok != "" && client != "" && tok != client {
		h.noteCredentialDisagreement(r, tok, client)
	}
	if tok == "" {
		tok = bearer
	}
	if tok == "" {
		tok = apiKey
	}
	if tok == "" {
		// Spend token: the runtime named the paying user (see payerHeader).
		if sub := strings.TrimSpace(r.Header.Get(payerHeader)); sub != "" {
			return sub, nil
		}
		if sub := strings.TrimSpace(r.Header.Get(relaySubHeader)); sub != "" {
			return sub, nil // sealed transport: wallet-vouched, relay-asserted
		}
		// Ingress mutual RA-TLS (app-to-app): the enclave manager verified the
		// caller's attested client certificate (quote, measurements, allowed-
		// callers policy, channel binder) and asserted its identity in these
		// headers — which it strips from every non-verified path, so they
		// cannot be spoofed from outside. The caller is the attested APP
		// itself; the "app:" prefix keeps the subject space disjoint from
		// pairwise user subs for metering and billing. An app-supplied
		// acting user (the former X-Privasys-On-Behalf-Of) is NOT honoured:
		// the app may say whom it serves, but only the platform (the spend
		// token payer above) or the person's own credential may say who
		// pays — with the any-attested caller policy an app-asserted name
		// would let any admitted app bill anyone.
		if r.Header.Get(peerVerifiedHeader) == "true" {
			if id := strings.TrimSpace(r.Header.Get(peerAppIDHeader)); id != "" {
				return "app:" + id, nil
			}
		}
		return "", nil // anonymous
	}
	if h.oidcVerifier == nil {
		return "", nil // no issuer configured; cannot verify, treat as anonymous
	}
	claims, err := h.oidcVerifier.Verify(r.Context(), tok)
	if err != nil {
		return "", err
	}
	// Reject a token whose session (API key) has been revoked. Checked against
	// the polled revoked-sid set — no per-request callout.
	if claims.SID != "" && h.revoked.Has(claims.SID) {
		return "", errors.New("credential revoked")
	}
	return claims.Subject, nil
}

// firstNonEmpty returns the first non-empty argument, or "".
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// subjectForLog names a token's subject for a log line, following the house
// convention of a truncated subject rather than the whole pseudonymous id. It
// never returns any part of the token itself.
func (h *Handler) subjectForLog(r *http.Request, tok string) string {
	if h.oidcVerifier == nil {
		return "unverified"
	}
	claims, err := h.oidcVerifier.Verify(r.Context(), tok)
	if err != nil {
		return "invalid"
	}
	return claims.Subject
}

// noteCredentialDisagreement reports, once per distinct pair and route, that a
// request carried BOTH the credential the proxied path resolved (X-App-Auth)
// and a client-presented one naming somebody else. The request is served on
// X-App-Auth regardless; this only answers whether the case occurs in
// production, so that refusing it later is a decision made on evidence.
//
// Deduplicated because a misconfigured caller would otherwise repeat this on
// every request, and the admin container-logs view is capped: a flood loses
// the beginning, which is exactly the part that says when it started. The
// distinct pairs are what carry the information; the repetitions do not.
func (h *Handler) noteCredentialDisagreement(r *http.Request, appAuth, client string) {
	appSub := h.subjectForLog(r, appAuth)
	clientSub := h.subjectForLog(r, client)
	if appSub == clientSub {
		return // one person, two envelopes: nothing to report
	}
	route := r.URL.Path
	key := appSub + "\x00" + clientSub + "\x00" + route

	h.credDisagreements.Lock()
	if h.credDisagreements.seen == nil {
		h.credDisagreements.seen = map[string]struct{}{}
	}
	_, already := h.credDisagreements.seen[key]
	// Bounded: past the cap we have long since learned that it happens, and
	// an unbounded map keyed by caller input is a memory leak.
	if !already && len(h.credDisagreements.seen) < 64 {
		h.credDisagreements.seen[key] = struct{}{}
	}
	h.credDisagreements.Unlock()
	if already {
		return
	}
	log.Printf("[auth] credential disagreement on %s: X-App-Auth names %.8s…, client header names %.8s… — serving X-App-Auth",
		route, appSub, clientSub)
}

// authorizeInference resolves the end-user on an inference request and rejects
// an anonymous/invalid/revoked caller with 401. Authentication is mandatory:
// privacy is guaranteed by the confidential-computing enclave (the prompt is
// never visible to us, provable via the enclave measurements and attestation),
// not by anonymity, so every caller is identified. On success it returns the
// request carrying the verified caller subject in context for the metering
// path.
func (h *Handler) authorizeInference(w http.ResponseWriter, r *http.Request) (*http.Request, bool) {
	sub, err := h.resolveCaller(r)
	if sub == "" {
		if err != nil {
			writeError(w, http.StatusUnauthorized, "invalid credential")
		} else {
			writeError(w, http.StatusUnauthorized, "authentication required")
		}
		return r, false
	}
	// Authenticated is not enough: the caller must be someone the platform
	// can debit. A subject with no billing account is refused here rather
	// than served on the deployment owner's tab (Bertrand, 2026-09-08). The
	// verdict is cached per caller; when the management-service cannot be
	// asked and nothing is cached, inference fails open (as the freeze probe
	// does) and the reporter drops the line server-side.
	if rep := h.billing.Load(); rep != nil {
		billable, err := rep.CallerBillable(r.Context(), sub)
		if err != nil {
			log.Printf("[billing] billability of caller %.8s… unknown, serving: %v", sub, err)
		} else if !billable {
			writeError(w, http.StatusPaymentRequired, "no billing account for this caller: open a Privasys platform account to use inference")
			return r, false
		}
	}
	ctx := context.WithValue(r.Context(), callerCtxKey{}, sub)
	if strings.TrimSpace(r.Header.Get(payerHeader)) == sub {
		ctx = context.WithValue(ctx, payerCtxKey{}, billing.Caller{
			Sub:    sub,
			App:    strings.ToLower(strings.TrimSpace(r.Header.Get(payerAppHeader))),
			SID:    strings.TrimSpace(r.Header.Get(payerSIDHeader)),
			Ledger: strings.TrimSpace(r.Header.Get(payerLedgerHeader)),
		})
	}
	r = r.WithContext(ctx)
	return r, true
}

// HasRole reports whether the token carries the given role.
func (c *OIDCClaims) HasRole(role string) bool {
	for _, r := range c.Roles {
		if r == role {
			return true
		}
	}
	return false
}

// NewOIDCVerifier returns a verifier for tokens issued by issuer. audience
// may be empty to skip the aud check.
func NewOIDCVerifier(issuer, audience string) *OIDCVerifier {
	return &OIDCVerifier{
		issuer:   strings.TrimRight(issuer, "/"),
		audience: audience,
		client:   &http.Client{Timeout: 10 * time.Second},
		keys:     map[string]*oidcJWK{},
	}
}

// Verify validates the token's signature, issuer, audience and expiry and
// returns its subject and roles. It never contacts the issuer with the
// request payload; only the (cached) JWKS is fetched.
func (v *OIDCVerifier) Verify(ctx context.Context, token string) (*OIDCClaims, error) {
	parts := strings.SplitN(token, ".", 3)
	if len(parts) != 3 {
		return nil, errors.New("oidc: malformed token")
	}

	headerJSON, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return nil, fmt.Errorf("oidc: header decode: %w", err)
	}
	var header struct {
		Alg string `json:"alg"`
		Kid string `json:"kid"`
	}
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("oidc: header parse: %w", err)
	}

	jwk, err := v.signingKey(ctx, header.Kid, header.Alg)
	if err != nil {
		return nil, fmt.Errorf("oidc: jwks lookup: %w", err)
	}

	signingInput := []byte(parts[0] + "." + parts[1])
	sig, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return nil, fmt.Errorf("oidc: sig decode: %w", err)
	}
	if err := jwkVerify(header.Alg, jwk, signingInput, sig); err != nil {
		return nil, fmt.Errorf("oidc: signature: %w", err)
	}

	claimsJSON, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("oidc: claims decode: %w", err)
	}
	var claims map[string]any
	if err := json.Unmarshal(claimsJSON, &claims); err != nil {
		return nil, fmt.Errorf("oidc: claims parse: %w", err)
	}

	if iss, _ := claims["iss"].(string); iss != v.issuer {
		return nil, fmt.Errorf("oidc: issuer %q != %q", iss, v.issuer)
	}
	if v.audience != "" && !oidcCheckAudience(claims, v.audience) {
		return nil, fmt.Errorf("oidc: audience missing %q", v.audience)
	}
	if exp, ok := claims["exp"].(float64); ok {
		if time.Now().Unix() > int64(exp) {
			return nil, errors.New("oidc: token expired")
		}
	}

	sub, _ := claims["sub"].(string)
	sid, _ := claims["sid"].(string)
	return &OIDCClaims{Subject: sub, Roles: oidcRoles(claims), SID: sid}, nil
}

// oidcRoles extracts a string list from the `roles` claim (array form) and
// Keycloak's realm_access.roles, matching the manager's checkRole paths.
func oidcRoles(claims map[string]any) []string {
	var out []string
	if arr, ok := claims["roles"].([]any); ok {
		for _, r := range arr {
			if s, ok := r.(string); ok {
				out = append(out, s)
			}
		}
	}
	if ra, ok := claims["realm_access"].(map[string]any); ok {
		if arr, ok := ra["roles"].([]any); ok {
			for _, r := range arr {
				if s, ok := r.(string); ok {
					out = append(out, s)
				}
			}
		}
	}
	return out
}

func oidcCheckAudience(claims map[string]any, expected string) bool {
	switch aud := claims["aud"].(type) {
	case string:
		return aud == expected
	case []any:
		for _, a := range aud {
			if s, ok := a.(string); ok && s == expected {
				return true
			}
		}
	}
	return false
}

// --- JWKS ---

type oidcJWK struct {
	Kty string `json:"kty"`
	Kid string `json:"kid"`
	Alg string `json:"alg"`
	Use string `json:"use"`
	N   string `json:"n"`
	E   string `json:"e"`
	Crv string `json:"crv"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

func (v *OIDCVerifier) signingKey(ctx context.Context, kid, alg string) (*oidcJWK, error) {
	v.mu.RLock()
	if len(v.keys) > 0 && time.Since(v.fetchedAt) < 5*time.Minute {
		if k, ok := v.keys[kid]; ok {
			v.mu.RUnlock()
			return k, nil
		}
	}
	v.mu.RUnlock()

	v.mu.Lock()
	defer v.mu.Unlock()
	if len(v.keys) > 0 && time.Since(v.fetchedAt) < 5*time.Minute {
		if k, ok := v.keys[kid]; ok {
			return k, nil
		}
	}

	jwksURL, err := v.discoverJWKS(ctx)
	if err != nil {
		return nil, err
	}
	keys, err := v.fetchJWKS(ctx, jwksURL)
	if err != nil {
		return nil, err
	}
	v.keys = keys
	v.fetchedAt = time.Now()

	if k, ok := keys[kid]; ok {
		return k, nil
	}
	if kid == "" {
		for _, k := range keys {
			if k.Alg == alg || (k.Use == "sig" && k.Alg == "") {
				return k, nil
			}
		}
	}
	return nil, fmt.Errorf("key %q not in JWKS", kid)
}

func (v *OIDCVerifier) discoverJWKS(ctx context.Context) (string, error) {
	url := v.issuer + "/.well-known/openid-configuration"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := v.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("discovery: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("discovery status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	var disc struct {
		JwksURI string `json:"jwks_uri"`
	}
	if err := json.Unmarshal(body, &disc); err != nil {
		return "", fmt.Errorf("discovery parse: %w", err)
	}
	if disc.JwksURI == "" {
		return "", errors.New("discovery: no jwks_uri")
	}
	return disc.JwksURI, nil
}

func (v *OIDCVerifier) fetchJWKS(ctx context.Context, uri string) (map[string]*oidcJWK, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, uri, nil)
	if err != nil {
		return nil, err
	}
	resp, err := v.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("jwks fetch: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("jwks status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	var doc struct {
		Keys []oidcJWK `json:"keys"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, fmt.Errorf("jwks parse: %w", err)
	}
	keys := make(map[string]*oidcJWK, len(doc.Keys))
	for i := range doc.Keys {
		k := doc.Keys[i]
		keys[k.Kid] = &k
	}
	if len(keys) == 0 {
		return nil, errors.New("jwks empty")
	}
	return keys, nil
}

// --- signature verification ---

func jwkVerify(alg string, key *oidcJWK, signingInput, sig []byte) error {
	switch {
	case strings.HasPrefix(alg, "RS"):
		return jwkVerifyRSA(alg, key, signingInput, sig)
	case strings.HasPrefix(alg, "ES"):
		return jwkVerifyEC(key, signingInput, sig)
	default:
		return fmt.Errorf("unsupported alg %q", alg)
	}
}

func jwkVerifyRSA(alg string, key *oidcJWK, signingInput, sig []byte) error {
	if key.Kty != "RSA" {
		return fmt.Errorf("expected RSA key, got %s", key.Kty)
	}
	nb, err := base64.RawURLEncoding.DecodeString(key.N)
	if err != nil {
		return err
	}
	eb, err := base64.RawURLEncoding.DecodeString(key.E)
	if err != nil {
		return err
	}
	e := 0
	for _, b := range eb {
		e = e<<8 + int(b)
	}
	pub := &rsa.PublicKey{N: new(big.Int).SetBytes(nb), E: e}
	var h crypto.Hash
	switch alg {
	case "RS256":
		h = crypto.SHA256
	case "RS384":
		h = crypto.SHA384
	case "RS512":
		h = crypto.SHA512
	default:
		return fmt.Errorf("unsupported RSA alg %q", alg)
	}
	hh := h.New()
	hh.Write(signingInput)
	return rsa.VerifyPKCS1v15(pub, h, hh.Sum(nil), sig)
}

func jwkVerifyEC(key *oidcJWK, signingInput, sig []byte) error {
	if key.Kty != "EC" {
		return fmt.Errorf("expected EC key, got %s", key.Kty)
	}
	xb, err := base64.RawURLEncoding.DecodeString(key.X)
	if err != nil {
		return err
	}
	yb, err := base64.RawURLEncoding.DecodeString(key.Y)
	if err != nil {
		return err
	}
	var curve elliptic.Curve
	var size int
	var hash func([]byte) []byte
	switch key.Crv {
	case "P-256":
		curve, size = elliptic.P256(), 32
		hash = func(b []byte) []byte { s := sha256.Sum256(b); return s[:] }
	case "P-384":
		curve, size = elliptic.P384(), 48
		hash = func(b []byte) []byte { s := sha512.Sum384(b); return s[:] }
	default:
		return fmt.Errorf("unsupported curve %q", key.Crv)
	}
	if len(sig) != size*2 {
		return fmt.Errorf("EC sig wrong length %d, want %d", len(sig), size*2)
	}
	pub := &ecdsa.PublicKey{Curve: curve, X: new(big.Int).SetBytes(xb), Y: new(big.Int).SetBytes(yb)}
	r := new(big.Int).SetBytes(sig[:size])
	s := new(big.Int).SetBytes(sig[size:])
	if !ecdsa.Verify(pub, hash(signingInput), r, s) {
		return errors.New("EC signature invalid")
	}
	return nil
}
