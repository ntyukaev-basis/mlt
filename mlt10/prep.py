#!/usr/bin/env python3
"""MLT-10 pipeline — prep step. Read a dataset CSV, write stratified train/test splits."""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split


def find_csv(input_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"prep: no CSV found under {input_dir!r}")
    return paths[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=os.environ.get("AIR_DATASET_PATH", "/data"),
                   help="Directory holding the dataset CSV (default: %(default)s).")
    p.add_argument("--train-out", required=True, help="Output path for the train split CSV.")
    p.add_argument("--test-out", required=True, help="Output path for the test split CSV.")
    p.add_argument("--target-col", default=None,
                   help="Target column; auto-detected (target/label/quality/...) if omitted.")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = find_csv(args.input_dir)
    print(f"prep: reading {src}")
    df = pd.read_csv(src, sep=None, engine="python")
    print(f"prep: columns={list(df.columns)} shape={df.shape}")

    tcol = args.target_col
    if tcol is None:
        cand = [c for c in df.columns
                if c.strip().lower() in ("target", "label", "y", "quality", "class", "output")]
        tcol = cand[0] if cand else df.columns[-1]

    y = df[tcol]
    X = df.drop(columns=[tcol]).select_dtypes(include=[np.number]).fillna(0)
    if y.nunique() > 2:
        thr = float(y.median())
        y = (y > thr).astype(int)
        print(f"prep: binarized target {tcol!r} at median {thr}")

    strat = y if y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=strat)
    tr = X_tr.copy(); tr["target"] = y_tr.values
    te = X_te.copy(); te["target"] = y_te.values

    for path in (args.train_out, args.test_out):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tr.to_csv(args.train_out, index=False)
    te.to_csv(args.test_out, index=False)
    print(f"prep: wrote train={tr.shape} -> {args.train_out}, test={te.shape} -> {args.test_out} "
          f"(target_col={tcol})")


if __name__ == "__main__":
    main()