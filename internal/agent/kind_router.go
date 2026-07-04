// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package agent

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// KindRouter routes MCP traffic per target host according to the tool's
// kind:
//
//   - ENCLAVE tools (the grant carries their attested workload digest) go
//     over the attested RA-TLS transport — the gateway's terminated leg
//     refuses plaintext app traffic, and the peer enclave is verified
//     before any tool data is sent.
//   - EXTERNAL tools (no digest — off-platform MCP servers the user added
//     behind an explicit acknowledgement) go over standard WebPKI TLS with
//     an SSRF-guarded dialer: the URL is user input reaching out from
//     inside the enclave, so it must never be able to address loopback,
//     RFC1918/ULA, link-local (cloud metadata!), or CGNAT ranges.
//
// Hosts not registered as external default to the enclave transport, so a
// mis-tagged server fails closed (attestation error) rather than leaking
// tool data over an unattested channel.
type KindRouter struct {
	enclave  http.RoundTripper
	external http.RoundTripper
	// externalHosts is lowercase hostname (no port) → true.
	externalHosts map[string]bool
}

// NewKindRouter builds a router. enclave must be the attested RA-TLS
// RoundTripper (or a dev fallback); externalHosts lists the hostnames of
// the request's external tools.
func NewKindRouter(enclave http.RoundTripper, externalHosts []string) *KindRouter {
	set := make(map[string]bool, len(externalHosts))
	for _, h := range externalHosts {
		if h = strings.ToLower(strings.TrimSpace(h)); h != "" {
			set[h] = true
		}
	}
	if enclave == nil {
		enclave = http.DefaultTransport
	}
	return &KindRouter{
		enclave:       enclave,
		external:      newExternalTransport(),
		externalHosts: set,
	}
}

func (r *KindRouter) RoundTrip(req *http.Request) (*http.Response, error) {
	if r.externalHosts[strings.ToLower(req.URL.Hostname())] {
		if req.URL.Scheme != "https" {
			return nil, fmt.Errorf("external tool %s: only https is allowed", req.URL.Hostname())
		}
		return r.external.RoundTrip(req)
	}
	return r.enclave.RoundTrip(req)
}

// ExternalHostsOf collects the hostnames of servers WITHOUT an attested
// workload digest — i.e. external tools — for NewKindRouter. Servers with
// unparseable BaseURLs are skipped (their requests will fail on the
// enclave path, which is the fail-closed default).
func ExternalHostsOf(servers []Server) []string {
	var hosts []string
	for _, s := range servers {
		if s.ExpectedDigest != "" {
			continue
		}
		if u, err := url.Parse(s.BaseURL); err == nil && u.Hostname() != "" {
			hosts = append(hosts, u.Hostname())
		}
	}
	return hosts
}

// PinnedEnclaveTransport returns `base` upgraded with per-host workload
// digest pinning for every server that carries an ExpectedDigest: after
// the peer attests, its leaf's workload code hash (OID 3.2) must equal
// the digest the user admitted, or the request fails closed. When `base`
// is not the RA-TLS transport (local dev fallback) it is returned
// unchanged — pinning without attestation would be theatre.
func PinnedEnclaveTransport(base http.RoundTripper, servers []Server) http.RoundTripper {
	rt, ok := base.(*RATLSTransport)
	if !ok || rt == nil {
		return base
	}
	digests := map[string]string{}
	for _, s := range servers {
		if s.ExpectedDigest == "" {
			continue
		}
		if u, err := url.Parse(s.BaseURL); err == nil && u.Hostname() != "" {
			digests[strings.ToLower(u.Hostname())] = strings.ToLower(s.ExpectedDigest)
		}
	}
	if len(digests) == 0 {
		return base
	}
	pinned := *rt // shallow copy: per-request dialing, no shared conn state
	// Merge over any digests the base already pins (per-request map).
	merged := make(map[string]string, len(rt.ExpectedDigests)+len(digests))
	for k, v := range rt.ExpectedDigests {
		merged[k] = v
	}
	for k, v := range digests {
		merged[k] = v
	}
	pinned.ExpectedDigests = merged
	return &pinned
}

// newExternalTransport returns a WebPKI TLS transport whose dialer
// resolves the target and refuses non-public addresses, dialling the
// vetted IP directly so a DNS rebind between check and connect cannot
// bypass the guard.
func newExternalTransport() http.RoundTripper {
	return &http.Transport{
		Proxy:                 nil,
		DialContext:           ssrfGuardedDial,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          10,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 2 * time.Second,
	}
}

func ssrfGuardedDial(ctx context.Context, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, fmt.Errorf("external tool dial %q: %w", addr, err)
	}
	ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("external tool resolve %q: %w", host, err)
	}
	d := net.Dialer{Timeout: 10 * time.Second}
	var lastErr error
	for _, ip := range ips {
		if !isPublicIP(ip.IP) {
			lastErr = fmt.Errorf("external tool %q resolves to non-public address %s — refused", host, ip.IP)
			continue
		}
		conn, derr := d.DialContext(ctx, network, net.JoinHostPort(ip.IP.String(), port))
		if derr == nil {
			return conn, nil
		}
		lastErr = derr
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("external tool %q: no addresses", host)
	}
	return nil, lastErr
}

// isPublicIP rejects every address family an external tool must never
// reach from inside the enclave: loopback, RFC1918 + ULA (IsPrivate),
// link-local (incl. 169.254.169.254 cloud metadata), multicast,
// unspecified, and CGNAT 100.64.0.0/10.
func isPublicIP(ip net.IP) bool {
	if ip == nil ||
		ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsInterfaceLocalMulticast() ||
		ip.IsMulticast() ||
		ip.IsUnspecified() {
		return false
	}
	if v4 := ip.To4(); v4 != nil {
		// CGNAT 100.64.0.0/10 (not covered by IsPrivate).
		if v4[0] == 100 && v4[1]&0xc0 == 64 {
			return false
		}
	}
	return true
}
