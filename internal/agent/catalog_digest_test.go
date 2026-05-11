package agent

import "testing"

func TestServersDigest_IsStableAndOrderIndependent(t *testing.T) {
	a := NewCatalog([]Server{
		{Name: "lp", BaseURL: "https://lp", Transport: TransportMCPSSE, AuthMode: AuthModeExchange, AuthAudience: "lp"},
		{Name: "rag", BaseURL: "https://rag", BearerForward: true},
	}, nil, 0)
	b := NewCatalog([]Server{
		{Name: "rag", BaseURL: "https://rag", BearerForward: true},
		{Name: "lp", BaseURL: "https://lp", Transport: TransportMCPSSE, AuthMode: AuthModeExchange, AuthAudience: "lp"},
	}, nil, 0)
	if a.ServersDigest() == "" {
		t.Fatal("digest empty")
	}
	if a.ServersDigest() != b.ServersDigest() {
		t.Fatalf("digest not order-independent: %s != %s", a.ServersDigest(), b.ServersDigest())
	}

	c := NewCatalog([]Server{
		{Name: "rag", BaseURL: "https://rag-v2", BearerForward: true},
	}, nil, 0)
	if a.ServersDigest() == c.ServersDigest() {
		t.Fatal("digest should change when servers change")
	}
}
