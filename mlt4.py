#!/usr/bin/env python3
"""MLT-04 training — sklearn on the MLT-03 dataset, with zero-code autolog.

THE POINT OF THIS FILE IS WHAT IS NOT IN IT: there is not a single ``import
mlflow`` and not one ``mlflow.log_*`` call. Hyperparameters, the training
metric and the model artifact are captured because the platform injects an
``air-autolog-bootstrap`` sitecustomize that calls ``mlflow.autolog()`` before
this module runs. Adding logging calls here would defeat the requirement the
case exists to prove (13.2, «автоматическая фиксация»).

That also means the libraries have to be IN the image (``air/datascience-cpu``
ships sklearn + mlflow): the bootstrap imports mlflow at interpreter start,
before any runtime ``pip install`` could have happened.

Config is argparse, not os.environ — the platform's env is passed INTO the
arguments at the manifest level. Keeps the script runnable by hand:

    python mlt4.py --data-dir ./out --n-estimators 10
"""
import argparse
import glob
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir", required=True,
        help="Dataset mount point. The platform states it in "
             "AIR_DATASET_<SLUG>_PATH; pass that through in the manifest.",
    )
    p.add_argument("--file-name", default="wine.csv",
                   help="CSV to read inside the dataset.")
    p.add_argument("--target-col", default="quality",
                   help="Column to derive the label from.")
    p.add_argument("--threshold", type=float, default=6.0,
                   help="Label is <target-col> >= threshold (good wine).")
    p.add_argument("--test-size", type=float, default=0.25,
                   help="Hold-out share for the accuracy print.")
    # Constructor hyperparameters — deliberately passed, never logged: autolog
    # records whatever the estimator was built with.
    p.add_argument("--n-estimators", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def resolve_csv(data_dir: str, file_name: str) -> str:
    """Find the CSV whether a WHOLE dataset or a single version is mounted.

    A whole dataset mounts as one folder per version, so the file sits at
    ``<mount>/<version>/<name>`` and the newest folder wins. A pinned version
    mounts its own contents directly, putting the file at ``<mount>/<name>``.
    Supporting both keeps the script usable from MLT-04 (whole dataset) and
    from a retrain that pins one version, without a second code path.
    """
    direct = os.path.join(data_dir, file_name)
    if os.path.isfile(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(data_dir, "*", file_name)))
    if not hits:
        raise SystemExit(
            f"no {file_name} under {data_dir} (looked at <dir>/{file_name} "
            f"and <dir>/*/{file_name}) — has MLT-03 published a version yet?"
        )
    return hits[-1]


def main() -> None:
    args = parse_args()

    path = resolve_csv(args.data_dir, args.file_name)
    print(f"training on: {path}")
    df = pd.read_csv(path)

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col] >= args.threshold
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train)

    # Printed, not logged: the run's metrics come from autolog.
    print(f"test accuracy: {clf.score(X_test, y_test)}")


if __name__ == "__main__":
    main()
