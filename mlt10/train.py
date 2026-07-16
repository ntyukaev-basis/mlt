#!/usr/bin/env python3
"""MLT-10 pipeline — train step. autolog into the shared run ($MLFLOW_RUN_ID from env)."""
import argparse, os
import joblib, mlflow, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, help="Train split CSV (must have a 'target' column).")
    p.add_argument("--model-out", required=True, help="Output path for the pickled model.")
    p.add_argument("--max-iter", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.train)
    y = df["target"]; X = df.drop(columns=["target"])

    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:  # attaches to $MLFLOW_RUN_ID; experiment from env
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=args.max_iter))]).fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"train: run_id={run.info.run_id} model->{args.model_out}")


if __name__ == "__main__":
    main()