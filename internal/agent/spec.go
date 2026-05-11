package agent

import (
	"fmt"
	"net/url"
	"strings"
)

// ParseServerSpec parses the MCP_SERVERS env-var format into Server values.
//
// Format:
//
//	name1=https://url1[?bearer=1&transport=mcp_sse&auth=exchange&aud=...&scopes=a+b],name2=...
//
// Supported query-string keys (all optional):
//
//	bearer=1                     -> BearerForward=true (legacy)
//	transport=privasys_http|mcp_sse  (default: privasys_http)
//	auth=forward|exchange|static|none
//	aud=<audience>               -> AuthAudience (required when auth=exchange)
//	scopes=a+b+c                 -> AuthScopes (space-separated, URL-encoded)
//	confirm=1                    -> RequiresUserConfirmation=true
//
// Trailing slashes on the URL are stripped. Names must match
// [a-zA-Z0-9_]+ so they survive concatenation with the tool name into a
// vLLM function name.
func ParseServerSpec(spec string) ([]Server, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, nil
	}
	parts := strings.Split(spec, ",")
	out := make([]Server, 0, len(parts))
	seen := map[string]bool{}
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		eq := strings.IndexByte(p, '=')
		if eq <= 0 {
			return nil, fmt.Errorf("entry %q: expected name=url", p)
		}
		name := strings.TrimSpace(p[:eq])
		raw := strings.TrimSpace(p[eq+1:])
		if !validServerName(name) {
			return nil, fmt.Errorf("entry %q: name must match [a-zA-Z0-9_]+", p)
		}
		if seen[name] {
			return nil, fmt.Errorf("entry %q: duplicate server name", p)
		}
		seen[name] = true
		u, err := url.Parse(raw)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return nil, fmt.Errorf("entry %q: invalid URL", p)
		}
		s := Server{Name: name}
		q := u.Query()
		if q.Get("bearer") == "1" {
			s.BearerForward = true
			q.Del("bearer")
		}
		if t := q.Get("transport"); t != "" {
			switch t {
			case TransportPrivasysHTTP, TransportMCPSSE:
				s.Transport = t
			default:
				return nil, fmt.Errorf("entry %q: invalid transport %q", p, t)
			}
			q.Del("transport")
		}
		if a := q.Get("auth"); a != "" {
			switch a {
			case AuthModeForward, AuthModeExchange, AuthModeStatic, AuthModeNone:
				s.AuthMode = a
			default:
				return nil, fmt.Errorf("entry %q: invalid auth %q", p, a)
			}
			q.Del("auth")
		}
		if aud := q.Get("aud"); aud != "" {
			s.AuthAudience = aud
			q.Del("aud")
		}
		if scopes := q.Get("scopes"); scopes != "" {
			s.AuthScopes = strings.Fields(scopes)
			q.Del("scopes")
		}
		if q.Get("confirm") == "1" {
			s.RequiresUserConfirmation = true
			q.Del("confirm")
		}
		if s.AuthMode == AuthModeExchange && s.AuthAudience == "" {
			return nil, fmt.Errorf("entry %q: auth=exchange requires aud=", p)
		}
		u.RawQuery = q.Encode()
		s.BaseURL = strings.TrimRight(u.String(), "/")
		out = append(out, s)
	}
	return out, nil
}

func validServerName(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '_':
		default:
			return false
		}
	}
	return true
}
