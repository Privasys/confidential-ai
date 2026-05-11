package agent

import "testing"

func TestParseServerSpec(t *testing.T) {
	good := map[string]struct {
		in    string
		count int
		check func(*testing.T, []Server)
	}{
		"empty":  {"", 0, nil},
		"single": {"rag=http://rag:8443", 1, func(t *testing.T, ss []Server) {
			if ss[0].Name != "rag" || ss[0].BaseURL != "http://rag:8443" || ss[0].BearerForward {
				t.Fatalf("%+v", ss[0])
			}
		}},
		"with_bearer": {"rag=https://r/?bearer=1", 1, func(t *testing.T, ss []Server) {
			if !ss[0].BearerForward {
				t.Fatal("bearer should be true")
			}
			if ss[0].BaseURL != "https://r" {
				t.Fatalf("trailing slash not stripped: %q", ss[0].BaseURL)
			}
		}},
		"two": {"rag=http://r:1, lp = http://l:2", 2, func(t *testing.T, ss []Server) {
			if ss[1].Name != "lp" || ss[1].BaseURL != "http://l:2" {
				t.Fatalf("%+v", ss[1])
			}
		}},
		"sse_exchange": {"lp=https://lightpanda.example?transport=mcp_sse&auth=exchange&aud=lightpanda&scopes=browse", 1, func(t *testing.T, ss []Server) {
			s := ss[0]
			if s.Transport != TransportMCPSSE {
				t.Fatalf("transport: %q", s.Transport)
			}
			if s.AuthMode != AuthModeExchange || s.AuthAudience != "lightpanda" {
				t.Fatalf("auth: %+v", s)
			}
			if len(s.AuthScopes) != 1 || s.AuthScopes[0] != "browse" {
				t.Fatalf("scopes: %+v", s.AuthScopes)
			}
		}},
		"confirm": {"w=http://x?confirm=1", 1, func(t *testing.T, ss []Server) {
			if !ss[0].RequiresUserConfirmation {
				t.Fatal("confirm not set")
			}
		}},
	}
	for name, tc := range good {
		t.Run(name, func(t *testing.T) {
			ss, err := ParseServerSpec(tc.in)
			if err != nil {
				t.Fatal(err)
			}
			if len(ss) != tc.count {
				t.Fatalf("count: %d", len(ss))
			}
			if tc.check != nil {
				tc.check(t, ss)
			}
		})
	}

	bad := []string{
		"=http://x",
		"rag",
		"rag=notaurl",
		"rag=http://x,rag=http://y",
		"bad-name=http://x",
		"lp=http://x?transport=bogus",
		"lp=http://x?auth=bogus",
		"lp=http://x?auth=exchange",
	}
	for _, in := range bad {
		t.Run("bad/"+in, func(t *testing.T) {
			if _, err := ParseServerSpec(in); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
