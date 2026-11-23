"""Serves the test logs. GET or POST: / lists them, /<name> returns one, and
/tail/<name> its last 64 KiB (`privasys apps call` always POSTs).

POST /hf-token stores the request body as the Hugging Face token for gated
test models, in /data/secrets (outside the logs directory, never served).
"""

import http.server
import os
import sys

LOGS, PORT = sys.argv[1], int(sys.argv[2])
HF_TOKEN_FILE = "/data/secrets/hf-token"


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].strip("/")
        if path in ("health", "healthz", "readiness"):
            # The manager health-checks the container; a 404 here gets it
            # killed as unhealthy.
            self._send(200, b"ok")
            return
        tail = path.startswith("tail/")
        name = os.path.basename(path[5:] if tail else path)
        if not name:
            self._send(200, "\n".join(sorted(os.listdir(LOGS))).encode())
            return
        full = os.path.join(LOGS, name)
        if not os.path.isfile(full):
            self._send(404, b"not found")
            return
        with open(full, "rb") as f:
            if tail:
                f.seek(max(0, os.path.getsize(full) - 65536))
            self._send(200, f.read())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path.split("?", 1)[0].strip("/") == "hf-token":
            token = body.decode(errors="replace").strip().strip('"')
            if not token.startswith("hf_"):
                self._send(400, b"expected a Hugging Face token (hf_...)")
                return
            os.makedirs(os.path.dirname(HF_TOKEN_FILE), mode=0o700, exist_ok=True)
            fd = os.open(HF_TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(token)
            self._send(200, b"stored")
            return
        self.do_GET()

    def log_message(self, *_args) -> None:
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
