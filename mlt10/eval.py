#!/usr/bin/env python3
"""MLT-10 pipeline — eval step. Log test_accuracy into the shared run, write accuracy gate file."""
import argparse, os
import joblib, mlflow, pandas as pd
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", required=True, help="Test split CSV (must have a 'target' column).")
    p.add_argument("--model", required=True, help="Pickled model produced by the train step.")
    p.add_argument("--accuracy-out", required=True, help="Output path to write the accuracy value.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.test)
    y = df["target"]; X = df.drop(columns=["target"])

    pipe = joblib.load(args.model)
    acc = float(accuracy_score(y, pipe.predict(X)))

    with mlflow.start_run():  # attaches to $MLFLOW_RUN_ID
        mlflow.log_metric("test_accuracy", acc)

    os.makedirs(os.path.dirname(os.path.abspath(args.accuracy_out)), exist_ok=True)
    with open(args.accuracy_out, "w") as fh:
        fh.write(f"{acc:.4f}")
    print(f"eval: test_accuracy={acc:.4f} -> {args.accuracy_out}")


if __name__ == "__main__":
    main()