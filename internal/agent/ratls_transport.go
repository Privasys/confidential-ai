// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package agent

import (
	"crypto/rand"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	rc "enclave-os-mini/clients/go/ratls"
)

// RATLSTransport is an http.RoundTripper that carries each request over a
// freshly attested RA-TLS connection to the target enclave, instead of a
// gateway-terminated TLS leg.
//
// Why: the enclave gateways refuse plaintext app traffic on the terminated
// leg (403 sealed-transport-required) so user data can never transit the
// gateway in the clear — which is exactly what the agent loop's MCP calls
// are (tool arguments and results). RA-TLS terminates at the peer enclave
// itself (the gateway splices it through untouched, same hostname), passes
// that gate, AND attests the tool enclave before any data is sent — the
// enclave-to-enclave transport the platform's governance model expects.
//
// Per request: dial with a fresh ClientHello challenge, verify the peer's
// quote binds it (challenge-response report data), then send the request
// verbatim. Connections are not pooled; a tool call's dominant cost is the
// tool itself. Non-HTTPS URLs (local dev against plain-HTTP MCP servers)
// fall through to the standard transport.
//
// Building WITHOUT the Privasys Go fork (-tags ratls) leaves the challenge
// unsupported: Connect fails at runtime and the catalogue logs the error —
// production images are built with the fork (see Dockerfile).
type RATLSTransport struct {
	// Timeout bounds connect + attestation verification per request.
	Timeout time.Duration
	// Plain serves non-HTTPS requests (local dev). Defaults to
	// http.DefaultTransport.
	Plain http.RoundTripper
	// ExpectedDigests optionally pins the workload the peer must be
	// running, keyed by lowercase hostname: after the attestation
	// verifies, the peer leaf's workload code hash (OID
	// 1.3.6.1.4.1.65230.3.2) must equal the pinned bare-hex digest or the
	// request is refused. This is what makes a granted tool's
	// expected_digest an enforced promise, not just UI copy: a tool
	// enclave that was redeployed with different code since the user
	// admitted it fails closed. Hosts without an entry are not pinned.
	ExpectedDigests map[string]string
}

// NewRATLSTransport returns a RoundTripper with sane defaults.
func NewRATLSTransport() *RATLSTransport {
	return &RATLSTransport{Timeout: 15 * time.Second}
}

func (t *RATLSTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "https" {
		p := t.Plain
		if p == nil {
			p = http.DefaultTransport
		}
		return p.RoundTrip(req)
	}

	host := req.URL.Hostname()
	port := 443
	if ps := req.URL.Port(); ps != "" {
		if v, err := strconv.Atoi(ps); err == nil {
			port = v
		}
	}

	// Fresh nonce per request: the peer must bind its quote to THIS
	// handshake (challenge-response report data), so a replayed cert
	// or intercepted session cannot pass verification.
	nonce := make([]byte, 32)
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("ratls: nonce: %w", err)
	}

	timeout := t.Timeout
	if timeout <= 0 {
		timeout = 15 * time.Second
	}
	cli, err := rc.Connect(host, port, &rc.Options{
		ServerName: host,
		Timeout:    timeout,
		Challenge:  nonce,
	})
	if err != nil {
		return nil, fmt.Errorf("ratls: connect %s: %w", host, err)
	}

	// Verify the enclave BEFORE sending any tool data.
	info := cli.InspectCert()
	oid := ""
	if info.Quote != nil {
		oid = info.Quote.OID
	}
	policy := &rc.VerificationPolicy{
		TEE:        teeFromOID(oid),
		ReportData: rc.ReportDataChallengeResponse,
		Nonce:      nonce,
	}
	if _, verr := cli.VerifyCertificate(policy); verr != nil {
		cli.Close()
		return nil, fmt.Errorf("ratls: %s attestation failed — refusing to send tool data: %w", host, verr)
	}

	// Per-host workload pinning: the attested leaf must carry the exact
	// workload code hash the caller admitted (OID 3.2).
	if want := t.ExpectedDigests[strings.ToLower(host)]; want != "" {
		got := ""
		for _, ext := range info.CustomOids {
			if ext.OID == rc.OidWorkloadCodeHash {
				got = strings.ToLower(fmt.Sprintf("%x", ext.Value))
				break
			}
		}
		if !strings.EqualFold(got, want) {
			cli.Close()
			return nil, fmt.Errorf(
				"ratls: %s workload digest mismatch — enclave runs %s but the tool was admitted at %s; refusing to send tool data (the app changed since it was added)",
				host, orUnset(got), want)
		}
	}

	var body []byte
	if req.Body != nil {
		body, err = io.ReadAll(req.Body)
		req.Body.Close()
		if err != nil {
			cli.Close()
			return nil, fmt.Errorf("ratls: read request body: %w", err)
		}
	}
	bearer := strings.TrimPrefix(req.Header.Get("Authorization"), "Bearer ")

	resp, err := cli.HTTPDo(req.Method, req.URL.RequestURI(), host, body, bearer)
	if err != nil {
		cli.Close()
		return nil, fmt.Errorf("ratls: %s %s: %w", req.Method, host, err)
	}
	// The connection lives exactly as long as the response body.
	resp.Body = &connBody{ReadCloser: resp.Body, cli: cli}
	return resp, nil
}

// connBody ties the RA-TLS connection's lifetime to the response body.
type connBody struct {
	io.ReadCloser
	cli *rc.Client
}

func (b *connBody) Close() error {
	err := b.ReadCloser.Close()
	b.cli.Close()
	return err
}

// orUnset renders an empty digest readably in refusal messages.
func orUnset(v string) string {
	if v == "" {
		return "(no workload digest)"
	}
	return v
}

// teeFromOID maps a quote-extension OID to the TEE type the verification
// policy expects (mirrors the CLI's ratls package).
func teeFromOID(oid string) rc.TeeType {
	switch oid {
	case rc.OidTDXQuote:
		return rc.TeeTypeTDX
	case rc.OidSEVSNPReport:
		return rc.TeeTypeSEVSNP
	case rc.OidNVIDIAGPUEvidence:
		return rc.TeeTypeNVIDIAGPU
	default:
		return rc.TeeTypeSGX
	}
}
