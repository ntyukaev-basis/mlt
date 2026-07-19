#!/usr/bin/env python3
"""MLT-12 traffic generator — prod-like requests to a served model.

Reads feature rows from a mounted dataset CSV and POSTs them to a KServe
predict endpoint, imitating production inference traffic so the capture
sidecar fills a window the quality monitor then evaluates.

``--shift`` displaces every feature by a constant to simulate data drift:
0 → clean traffic (from the training distribution, monitor stays HEALTHY);
a large shift (e.g. 5) pushes every feature out of the trained range so
the drift detector fires DEGRADED and continuous training retrains.

The predict URL is derived from the InferenceService name
(``--serve-name`` → ``http://<name>-predictor/v1/models/<name>:predict``)
or given verbatim (``--target-url``) — nothing about the serve is
hard-coded into the workload template.
"""
import argparse
import glob
import json
import os
import urllib.request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        default=os.environ.get("DATASET_DIR", "/data/wine"),
        help="Mount holding the feature CSV (first *.csv found is used).",
    )
    p.add_argument(
        "--serve-name",
        default=os.environ.get("SERVE_NAME", "mlt12-wine-serve"),
        help="InferenceService name → "
        "http://<name>-predictor/v1/models/<name>:predict.",
    )
    p.add_argument(
        "--target-url",
        default=os.environ.get("TARGET_URL"),
        help="Full predict URL; overrides --serve-name when set.",
    )
    p.add_argument(
        "--shift",
        type=float,
        default=float(os.environ.get("SHIFT", "0")),
        help="Constant added to every feature (0=clean, >0=drift).",
    )
    p.add_argument(
        "--count",
        type=int,
        default=int(os.environ.get("COUNT", "150")),
        help="How many rows to send.",
    )
    p.add_argument("--target-col", default="quality",
                   help="Label column to drop before sending features.")
    return p.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    csvs = sorted(
        glob.glob(os.path.join(args.dataset_dir, "**", "*.csv"), recursive=True)
    )
    if not csvs:
        raise SystemExit(f"traffic: no CSV under {args.dataset_dir!r}")
    df = pd.read_csv(csvs[0], sep=None, engine="python")
    if args.target_col in df.columns:
        df = df.drop(columns=[args.target_col])
    df = df.head(args.count)

    url = args.target_url or (
        f"http://{args.serve_name}-predictor"
        f"/v1/models/{args.serve_name}:predict"
    )
    ok = 0
    for _, row in df.iterrows():
        payload = {
            "instances": [
                {k: round(float(v) + args.shift, 4) for k, v in row.items()}
            ]
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                ok += resp.status == 200
        except Exception as exc:  # noqa: BLE001 — report, keep sending
            print("traffic: request failed:", exc, flush=True)
    print(
        f"traffic: sent={len(df)} ok={ok} shift={args.shift} url={url}",
        flush=True,
    )


if __name__ == "__main__":
    main()
