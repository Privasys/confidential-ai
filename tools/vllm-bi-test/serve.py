"""Serves the test logs. GET or POST: / lists them, /<name> returns one, and
/tail/<name> its last 64 KiB (`privasys apps call` always POSTs)."""

import http.server
import os
import sys

LOGS, PORT = sys.argv[1], int(sys.argv[2])


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].strip("/")
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
        if length:
            self.rfile.read(length)
        self.do_GET()

    def log_message(self, *_args) -> None:
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
