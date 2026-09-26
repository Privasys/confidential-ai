// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// messagesReply is the smallest well-formed non-streaming Messages response.
func messagesReply(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id": "msg_test", "type": "message", "role": "assistant",
		"content": []map[string]any{{"type": "text", "text": "hello"}},
		"usage":   map[string]any{"input_tokens": 11, "output_tokens": 3},
	})
}

func messagesRequest() *http.Request {
	body := `{"model":"qwen36-35b-a3b-fp8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

// A stock Anthropic-style client sends its key in x-api-key and no bearer at
// all. That has to authenticate, or the Messages route this app serves cannot
// be reached by the clients it exists for.
func TestMessagesAcceptsApiKeyHeader(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	req := messagesRequest()
	req.Header.Set("X-Api-Key", authInference(t, h))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("x-api-key should authenticate, got %d: %s", rec.Code, rec.Body.String())
	}
}

// Every real client spells the header x-api-key, lower case, while Go's
// canonical form is X-Api-Key. Inbound canonicalisation makes those the same
// header, so this goes over a real connection rather than a recorder: reading
// the credential by direct map index instead of Header.Get would pass a
// recorder test and 401 every actual client.
func TestApiKeyHeaderOverTheWireIsLowerCase(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	key := authInference(t, h)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	body := `{"model":"qwen36-35b-a3b-fp8","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`
	req, err := http.NewRequest("POST", srv.URL+"/v1/messages", strings.NewReader(body))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header["x-api-key"] = []string{key} // exactly as an Anthropic client writes it
	res, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(res.Body)
		t.Fatalf("lower-case x-api-key should authenticate, got %d: %s", res.StatusCode, b)
	}
}

// The same credential in both envelopes is one credential, not a conflict.
func TestSameTokenInBothHeadersIsAccepted(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	tok := authInference(t, h)
	req := messagesRequest()
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("X-Api-Key", tok)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("one token in two headers should authenticate, got %d: %s", rec.Code, rec.Body.String())
	}
}

// Two DIFFERENT valid tokens are refused rather than ranked: picking one is how
// a caller ends up metered as somebody else.
func TestConflictingCredentialsAreRefused(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	issuer, mint := jwksTestIDP(t)
	h.oidcVerifier = NewOIDCVerifier(issuer, "")
	mintFor := func(sub string) string {
		return mint(map[string]any{"iss": issuer, "sub": sub, "exp": float64(time.Now().Add(time.Hour).Unix())})
	}
	req := messagesRequest()
	req.Header.Set("Authorization", "Bearer "+mintFor("alice"))
	req.Header.Set("X-Api-Key", mintFor("bob"))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("two different credentials must be refused, got %d: %s", rec.Code, rec.Body.String())
	}
}

// A presented key is used INSTEAD of a platform-vouched payer, so a junk
// x-api-key alongside a verified spend-token payer fails loudly. That is what
// makes stripping the header on an app's egress leg a requirement; this test
// is the reason the comment on apiKeyHeader says so.
func TestJunkApiKeyBeatsAVerifiedPayerAndFails(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	authInference(t, h) // wire a real verifier; the key below is not from it
	req := messagesRequest()
	req.Header.Set(payerHeader, "pairwise-payer-subject")
	req.Header.Set("X-Api-Key", "not-a-token")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("a junk presented key must fail rather than fall back to the payer, got %d", rec.Code)
	}
}

// With no presented key at all, the platform-vouched payer still serves: the
// ordering only changes what happens when a key IS present.
func TestPayerStillServesWithoutAPresentedKey(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	authInference(t, h)
	req := messagesRequest()
	req.Header.Set(payerHeader, "pairwise-payer-subject")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Fatalf("a vouched payer with no presented key must still authenticate: %s", rec.Body.String())
	}
}

// X-App-Auth is the credential the proxied path resolved for this call; a
// header the client also happened to send does not displace it.
func TestAppAuthWinsOverAClientHeader(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	issuer, mint := jwksTestIDP(t)
	h.oidcVerifier = NewOIDCVerifier(issuer, "")
	good := mint(map[string]any{"iss": issuer, "sub": "resolved-user", "exp": float64(time.Now().Add(time.Hour).Unix())})
	req := messagesRequest()
	req.Header.Set("X-App-Auth", good)
	req.Header.Set("X-Api-Key", "not-a-token")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("X-App-Auth must still decide the caller, got %d: %s", rec.Code, rec.Body.String())
	}
}

// A revoked credential is revoked in either envelope: same verifier, same
// revocation set.
func TestApiKeyGoesThroughRevocation(t *testing.T) {
	mux, h, _ := messagesHandler(t, messagesReply)
	issuer, mint := jwksTestIDP(t)
	h.oidcVerifier = NewOIDCVerifier(issuer, "")
	tok := mint(map[string]any{
		"iss": issuer, "sub": "test-user", "sid": "revoked-session",
		"exp": float64(time.Now().Add(time.Hour).Unix()),
	})
	if h.revoked == nil {
		h.revoked = newRevokedSet("", time.Minute)
	}
	h.revoked.mu.Lock()
	h.revoked.set["revoked-session"] = struct{}{}
	h.revoked.mu.Unlock()
	req := messagesRequest()
	req.Header.Set("X-Api-Key", tok)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("a revoked credential must be refused in x-api-key too, got %d", rec.Code)
	}
}
