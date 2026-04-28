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
//	name1=https://url1[?bearer=1],name2=https://url2,...
//
// `bearer=1` flips BearerForward on for that server. Trailing slashes on
// the URL are stripped. Names must match [a-zA-Z0-9_]+ so they survive
// concatenation with the tool name into a vLLM function name.
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
		bearer := false
		q := u.Query()
		if q.Get("bearer") == "1" {
			bearer = true
			q.Del("bearer")
			u.RawQuery = q.Encode()
		}
		base := strings.TrimRight(u.String(), "/")
		out = append(out, Server{Name: name, BaseURL: base, BearerForward: bearer})
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
