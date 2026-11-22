# Copyright (c) Privasys. All rights reserved.
# Licensed under the GNU Affero General Public License v3.0.
"""Loopback watermark detector for the instance key.

Started by the proxy (internal/models/watermark.go) once a watermarked chat
model is serving. POST /detect {"text": ...} tokenizes the text with the
model's own tokenizer, exactly as generation did (no special tokens), and
scores it with vLLM's Gumbel detector and the instance key, read from the
key file so it never appears on a command line. CPU only.

The detector settings mirror the engine's: vLLM's WatermarkConfig defaults
(philox PRF, context width 4) and the detector's own default of skipping
repeated contexts.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm.tokenizers import cached_get_tokenizer
from vllm.v1.watermarking import GumbelWatermarkDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--p-value-threshold", type=float, default=0.01)
    args = parser.parse_args()

    with open(args.key_file, encoding="ascii") as f:
        key = int(f.read().strip())
    tokenizer = cached_get_tokenizer(args.tokenizer)
    detector = GumbelWatermarkDetector(key=key, p_value_threshold=args.p_value_threshold)
    del key

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                text = json.loads(self.rfile.read(length))["text"]
                ids = tokenizer.encode(text, add_special_tokens=False)
                d = detector.detect(ids)
                out = {
                    "is_watermarked": d.is_watermarked,
                    "p_value": d.p_value,
                    "score": d.score,
                    "num_scored_tokens": d.num_scored_tokens,
                }
                body = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:  # never echo the input back
                self.send_error(400, type(e).__name__)

        def log_message(self, *_args) -> None:
            pass  # no request logging: texts are customer data

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
