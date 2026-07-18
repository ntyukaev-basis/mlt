#!/usr/bin/env python3
"""MLT-11 retrain script — Continuous Training target.

Launched by the platform (ct-system) from the retrain template whenever a
new version of the bound dataset is finalized. The whole dataset is
mounted (a folder per version, named by version id); the version that
fired the trigger arrives as AIR_DATASET_VERSION_ID — pass it via
--version-id, or omit to fall back to that env, or to the newest folder
by mtime when neither is set. Metrics/params/weights are captured by
zero-code autolog — no mlflow calls here.
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
                   help="Dataset version folder to train on "
                        "(default: $AIR_DATASET_VERSION_ID, else newest by mtime).")
    p.add_argument("--target-col", default="quality")
    return p.parse_args()


def pick_version_dir(dataset_dir: str, version_id: str | None) -> str:
    if version_id:
        path = os.path.join(dataset_dir, version_id)
        if os.path.isdir(path):
            return path
        raise SystemExit(f"retrain: version folder {path!r} not found")
    dirs = [d for d in glob.glob(os.path.join(dataset_dir, "*", "")) if os.path.isdir(d)]
    if not dirs:
        raise SystemExit(f"retrain: no version folders under {dataset_dir!r}")
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
