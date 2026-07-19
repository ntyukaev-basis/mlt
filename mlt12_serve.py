#!/usr/bin/env python3
"""MLT-12 serving script — wine model behind a KServe-compatible HTTP API.

Runs inside an InferenceService predictor container: the platform mounts
the registered model version (RO, zero-copy) and this script answers
``POST /v1/models/<name>:predict`` with ``{"predictions": [...]}`` —
the OIP v1 shape the capture sidecar and quality monitor understand.

The model directory is an MLflow sklearn artifact (``MLmodel`` +
pickled estimator); features arrive as one dict per instance, keyed by
the training column names.
"""
import argparse
import http.server
import json
import os
import socketserver

import mlflow.sklearn
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/model"),
                   help="Mounted MLflow sklearn model directory.")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9090")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = mlflow.sklearn.load_model(args.model_dir)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — stdlib naming
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            frame = pd.DataFrame(body.get("instances", []))
            preds = model.predict(frame)
            out = json.dumps({"predictions": [int(x) for x in preds]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self) -> None:  # noqa: N802 — health/readiness probes
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args) -> None:
            pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    print(f"mlt12_serve: wine model server on :{args.port} (model={args.model_dir})",
          flush=True)
    socketserver.ThreadingTCPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
