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
hard-coded into the workload template. Prefer ``--target-url``: the
derived form assumes the InferenceService name and the name the runtime
publishes the model under are the same, which is only true by convention.

Column headers are normalised to the form platform training records in
the model signature (see ``normalise_column``); ``--raw-columns`` sends
them verbatim.
"""
import argparse
import glob
import json
import os
import urllib.request


def normalise_column(name: str) -> str:
    """Column name as the training pipeline records it in the signature.

    MLflow enforces its input schema BY NAME, and platform training jobs
    normalise headers before fitting — so a model trained on
    ``wine-quality.csv`` expects ``fixed_acidity`` / ``ph`` while the CSV
    itself still says ``fixed acidity`` / ``pH``. Sending the raw headers
    makes every request fail schema validation with HTTP 400, which reads
    like a broken endpoint rather than a naming mismatch.
    """
    return "_".join(str(name).strip().lower().split())


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
    p.add_argument(
        "--raw-columns",
        action="store_true",
        default=os.environ.get("RAW_COLUMNS", "").lower() in ("1", "true"),
        help="Send CSV headers verbatim instead of normalising them to the "
             "form platform training records in the model signature. For a "
             "model trained outside the platform on the raw headers.",
    )
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
    if not args.raw_columns:
        df.columns = [normalise_column(c) for c in df.columns]
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
