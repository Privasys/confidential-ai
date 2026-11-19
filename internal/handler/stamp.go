// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// Publishing the chat model's digest into the serving certificate.
//
// The RA-TLS module also pulls /.well-known/attestation-extensions, but only
// when it mints a leaf, and a leaf is minted at the first handshake after
// the container starts (before any model is loaded) and then served for up
// to 24 hours. So a pull alone leaves the certificate without the model
// digest for most of a day. The manager's push endpoint is the designed
// path: it rewrites the extension file and reloads the RA-TLS module, so the
// next leaf carries the value. It is called whenever the digest changes.

// oidModelDigest is MODEL_DIGEST, in the app-defined 5.4 sub-arc.
const oidModelDigest = "1.3.6.1.4.1.65230.5.4.5"

// digestStamperConfigured reports whether the push path is available, in
// which case /.well-known/attestation-extensions must not declare the same
// OID (see attestationExtensions).
func digestStamperConfigured() bool {
	return os.Getenv("PRIVASYS_MANAGER_URL") != "" &&
		os.Getenv("PRIVASYS_CONTAINER_NAME") != "" &&
		os.Getenv("PRIVASYS_CONTAINER_TOKEN") != ""
}

// StartDigestStamper publishes the chat model's digest to the enclave
// manager each time it changes, until ctx ends. A no-op outside an enclave
// (no manager URL or container credentials in the environment).
func (h *Handler) StartDigestStamper(ctx context.Context) {
	managerURL := strings.TrimRight(os.Getenv("PRIVASYS_MANAGER_URL"), "/")
	container := os.Getenv("PRIVASYS_CONTAINER_NAME")
	token := os.Getenv("PRIVASYS_CONTAINER_TOKEN")
	if managerURL == "" || container == "" || token == "" || h.fleet == nil {
		return
	}
	// Proxy: nil, because the manager is on the bridge gateway and must
	// never be reached through an egress proxy set in the environment.
	client := &http.Client{Timeout: 15 * time.Second, Transport: &http.Transport{Proxy: nil}}
	url := fmt.Sprintf("%s/api/v1/containers/%s/attestation-extensions", managerURL, container)
	go func() {
		published := ""
		tick := time.NewTicker(10 * time.Second)
		defer tick.Stop()
		for {
			if d := h.fleet.Generate().ModelDigest(); d != "" && d != published {
				if err := stampDigest(ctx, client, url, token, d); err != nil {
					log.Printf("[attest] model digest not published yet (retrying): %v", err)
				} else {
					published = d
					log.Printf("[attest] model digest published at %s (%.16s…); the next leaf carries it", oidModelDigest, d)
				}
			}
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
			}
		}
	}()
}

// stampDigest POSTs one digest (hex) as the raw bytes of the extension.
func stampDigest(ctx context.Context, client *http.Client, url, token, digest string) error {
	raw, err := hex.DecodeString(digest)
	if err != nil {
		return fmt.Errorf("digest %q is not hex: %w", digest, err)
	}
	body, _ := json.Marshal(map[string]string{
		"oid":       oidModelDigest,
		"value_b64": base64.StdEncoding.EncodeToString(raw),
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return fmt.Errorf("manager answered %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return nil
}
