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
    p.add_argument(
        "--storage-uri", default=os.environ.get("AIR_DATASET_STORAGE_URI"),
        help="Platform-stated location of the tracked version; its last path "
             "segment is the folder to train on "
             "(default: $AIR_DATASET_STORAGE_URI).",
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


def resolve_csv(data_dir: str, file_name: str, storage_uri: str | None) -> str:
    """Find the CSV whether a WHOLE dataset or a single version is mounted.

    A pinned version mounts its own contents, putting the file at
    ``<mount>/<name>``. A whole dataset mounts one folder PER VERSION, and
    picking the right folder is where this gets sharp.

    Folder names are NOT uniform: a version created by upload is named after
    its version id, one produced by a workload carries whatever name that
    workload chose (``20260721-035614``, ``labeled-202607171240``). So sorting
    the names and taking the last one does not mean "newest" — it means
    "whichever string happens to sort highest", which on the DE stand quietly
    selected a months-old uploaded version with a different schema and blew up
    on a missing column. Same trap as in mlt11.py.

    So: the platform states the location of the version being tracked in
    AIR_DATASET_STORAGE_URI — trust that, and use the last path segment as the
    folder. Only when it is absent fall back to the newest folder BY
    MODIFICATION TIME, which at least means what it says.
    """
    direct = os.path.join(data_dir, file_name)
    if os.path.isfile(direct):
        return direct

    if storage_uri:
        folder = storage_uri.rstrip("/").rsplit("/", 1)[-1]
        pinned = os.path.join(data_dir, folder, file_name)
        if os.path.isfile(pinned):
            return pinned
        print(f"warning: {folder}/{file_name} not found, falling back to newest")

    hits = glob.glob(os.path.join(data_dir, "*", file_name))
    if not hits:
        raise SystemExit(
            f"no {file_name} under {data_dir} (looked at <dir>/{file_name} "
            f"and <dir>/*/{file_name}) — has MLT-03 published a version yet?"
        )
    return max(hits, key=os.path.getmtime)


def main() -> None:
    args = parse_args()

    path = resolve_csv(args.data_dir, args.file_name, args.storage_uri)
    print(f"training on: {path}")
    df = pd.read_csv(path)

    # Versions of one dataset can differ in schema (an uploaded CSV and a
    # produced one need not agree). Say so plainly instead of letting pandas
    # raise a KeyError from three frames down.
    if args.target_col not in df.columns:
        raise SystemExit(
            f"no '{args.target_col}' column in {path}; columns are: "
            f"{list(df.columns)}. Wrong dataset version? Pin the intended one "
            f"via experiment_tracking.dataset_version_id, or pass --target-col."
        )

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
