package handler

import (
	"net/http"
	"strings"
)

// CORS wraps next so cross-origin browser requests from Origins listed in
// allowOrigins receive the appropriate Access-Control-* response headers.
// Preflight (OPTIONS) requests are short-circuited with 204 No Content.
//
// allowOrigins is a comma-separated list of full Origin values
// (scheme://host[:port]), e.g. "https://chat.privasys.org,https://chat-test.privasys.org".
// An empty list disables CORS entirely.
func CORS(allowOrigins string, next http.Handler) http.Handler {
	allow := map[string]struct{}{}
	for _, o := range strings.Split(allowOrigins, ",") {
		o = strings.TrimSpace(o)
		if o != "" {
			allow[o] = struct{}{}
		}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" {
			if _, ok := allow[origin]; ok {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Vary", "Origin")
				w.Header().Set("Access-Control-Allow-Credentials", "true")
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				// Echo requested headers for preflight; otherwise allow the
				// common ones the chat front-end sends.
				if rh := r.Header.Get("Access-Control-Request-Headers"); rh != "" {
					w.Header().Set("Access-Control-Allow-Headers", rh)
				} else {
					w.Header().Set("Access-Control-Allow-Headers", "Accept, Authorization, Content-Type")
				}
				w.Header().Set("Access-Control-Max-Age", "600")
			}
		}
		if r.Method == http.MethodOptions && origin != "" {
			// Preflight short-circuit. If the origin is not allowed, the
			// missing ACAO header tells the browser to block; we still
			// respond 204 to avoid noisy errors in the enclave logs.
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
