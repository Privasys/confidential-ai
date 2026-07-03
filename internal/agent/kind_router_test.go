// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package agent

import (
	"net"
	"net/http"
	"net/url"
	"testing"
)

type markerRT struct{ name string }

func (m *markerRT) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, &url.Error{Op: m.name, URL: "", Err: nil}
}

func TestKindRouterRouting(t *testing.T) {
	enclave := &markerRT{name: "enclave"}
	r := NewKindRouter(enclave, []string{"External.Example.COM"})

	// External host → external transport (https enforced first).
	req, _ := http.NewRequest("GET", "http://external.example.com/mcp", nil)
	if _, err := r.RoundTrip(req); err == nil || err.Error() == "enclave" {
		t.Fatalf("external over http must be refused, got %v", err)
	}

	// Unknown host → enclave transport (fail-closed default).
	req2, _ := http.NewRequest("GET", "https://tool.apps-test.privasys.org/mcp", nil)
	_, err := r.RoundTrip(req2)
	uerr, ok := err.(*url.Error)
	if !ok || uerr.Op != "enclave" {
		t.Fatalf("unknown host should route to the enclave transport, got %v", err)
	}
}

func TestExternalHostsOf(t *testing.T) {
	servers := []Server{
		{Name: "enclave_tool", BaseURL: "https://a.apps.privasys.org", ExpectedDigest: "ab12"},
		{Name: "external_tool", BaseURL: "https://mcp.example.com:8443/base"},
		{Name: "broken", BaseURL: "://nope"},
	}
	hosts := ExternalHostsOf(servers)
	if len(hosts) != 1 || hosts[0] != "mcp.example.com" {
		t.Fatalf("hosts = %v, want [mcp.example.com]", hosts)
	}
}

func TestIsPublicIP(t *testing.T) {
	blocked := []string{
		"127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1",
		"169.254.169.254", // cloud metadata
		"100.64.0.1", "100.127.255.254", // CGNAT
		"0.0.0.0", "224.0.0.1",
		"::1", "fe80::1", "fc00::1", "ff02::1", "::",
	}
	for _, s := range blocked {
		if isPublicIP(net.ParseIP(s)) {
			t.Errorf("%s must be blocked", s)
		}
	}
	allowed := []string{"93.184.216.34", "1.1.1.1", "100.63.255.255", "100.128.0.1", "2606:4700::1111"}
	for _, s := range allowed {
		if !isPublicIP(net.ParseIP(s)) {
			t.Errorf("%s must be allowed", s)
		}
	}
}
