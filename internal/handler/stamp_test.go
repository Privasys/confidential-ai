// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// The manager installs exactly what it receives into the next leaf, so the
// request must carry the MODEL_DIGEST OID, the digest's raw bytes (not its
// hex text) and the container's own token.
func TestStampDigestRequest(t *testing.T) {
	const digest = "cd80fca1d85b0717"
	var got struct {
		OID      string `json:"oid"`
		ValueB64 string `json:"value_b64"`
	}
	var auth, path string
	mgr := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		json.NewDecoder(r.Body).Decode(&got)
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer mgr.Close()

	url := mgr.URL + "/api/v1/containers/cai/attestation-extensions"
	if err := stampDigest(context.Background(), mgr.Client(), url, "ctok", digest); err != nil {
		t.Fatal(err)
	}
	raw, _ := base64.StdEncoding.DecodeString(got.ValueB64)
	if got.OID != oidModelDigest || hex.EncodeToString(raw) != digest {
		t.Fatalf("body = %+v, want %s carrying the raw digest", got, oidModelDigest)
	}
	if auth != "Bearer ctok" || path != "/api/v1/containers/cai/attestation-extensions" {
		t.Fatalf("auth %q path %q", auth, path)
	}

	refused := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusForbidden)
	}))
	defer refused.Close()
	if err := stampDigest(context.Background(), refused.Client(), refused.URL, "ctok", digest); err == nil {
		t.Fatal("a refusal must be reported so the stamper retries")
	}
}
