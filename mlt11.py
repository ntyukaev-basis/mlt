#!/usr/bin/env python3
"""MLT-11 retrain script — Continuous Training target.

Launched by the platform (ct-system) from the retrain template whenever a
new version of the bound dataset is finalized. The whole dataset is
mounted — one folder per version — and the version that fired the
trigger is identified by the env the platform injects.

Resolving WHICH folder matters more than it looks. A version created by
UPLOAD lives in a folder named after its version id, but a version
PRODUCED by a workload (the capture→labels ETL that closes the CT loop)
lives in whatever folder that workload created — e.g.
``labeled-202607171230``. Assuming folder == version id therefore works
until the first produced version and then fails with "version folder not
found", which is exactly what happened on the DE stand. So the folder is
taken from AIR_DATASET_STORAGE_URI (the platform states the true
location), and the version id is only a fallback.

Metrics/params/weights are captured by zero-code autolog — no mlflow
calls here.
"""
import argparse
import glob
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True,
                   help="Mount point of the whole dataset (a folder per version).")
    p.add_argument("--version-id", default=os.environ.get("AIR_DATASET_VERSION_ID"),
                   help="Version id of the dataset version to train on "
                        "(default: $AIR_DATASET_VERSION_ID).")
    p.add_argument("--storage-uri", default=os.environ.get("AIR_DATASET_STORAGE_URI"),
                   help="Platform-stated location of that version; its last "
                        "path segment is the folder name "
                        "(default: $AIR_DATASET_STORAGE_URI).")
    p.add_argument("--target-col", default="quality")
    return p.parse_args()


def pick_version_dir(
    dataset_dir: str, version_id: str | None, storage_uri: str | None = None
) -> str:
    """Locate the mounted folder of the version that fired the trigger.

    Order matters: the storage URI is what the platform actually recorded
    for the version, so it is authoritative for BOTH upload-created and
    workload-produced versions. The version id is tried next (it names
    the folder only for uploads), and newest-by-mtime is the last resort.
    """
    candidates = []
    if storage_uri:
        # s3://bucket/…/<dataset>/<folder>/ → <folder>
        folder = storage_uri.rstrip("/").rsplit("/", 1)[-1]
        if folder:
            candidates.append(folder)
    if version_id:
        candidates.append(version_id)

    for name in candidates:
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            return path

    dirs = [d for d in glob.glob(os.path.join(dataset_dir, "*", "")) if os.path.isdir(d)]
    if not dirs:
        raise SystemExit(f"retrain: no version folders under {dataset_dir!r}")
    if candidates:
        print(
            "retrain: none of "
            f"{candidates} found under {dataset_dir!r} — falling back to the "
            "newest folder by mtime"
        )
    return max(dirs, key=os.path.getmtime)


def main() -> None:
    args = parse_args()
    vdir = pick_version_dir(args.dataset_dir, args.version_id)
    csvs = sorted(glob.glob(os.path.join(vdir, "**", "*.csv"), recursive=True))
    if not csvs:
        raise SystemExit(f"retrain: no CSV in version dir {vdir!r}")
    src = csvs[0]
    print(f"retrain: version dir={vdir} csv={src}")

    df = pd.read_csv(src, sep=None, engine="python")
    X = df.drop(columns=[args.target_col])
    y = df[args.target_col] >= df[args.target_col].median()
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
    clf.fit(Xtr, ytr)
    print("retrain: test accuracy:", clf.score(Xte, yte))


if __name__ == "__main__":
    main()
