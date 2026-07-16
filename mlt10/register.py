#!/usr/bin/env python3
"""MLT-10 pipeline — register step. Register runs:/$MLFLOW_RUN_ID/model (run id from env)."""
import argparse, os
import mlflow


def _resolve(value, file_value):
    return open(file_value).read().strip() if file_value else value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True, help="Registered model name.")
    p.add_argument("--artifact-path", default="model")
    p.add_argument("--accuracy")
    p.add_argument("--accuracy-file")
    p.add_argument("--min-accuracy", type=float, default=None,
                   help="Skip registration unless accuracy is strictly above this threshold.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise SystemExit("register: MLFLOW_RUN_ID env is required")

    acc_str = _resolve(args.accuracy, args.accuracy_file)
    if args.min_accuracy is not None:
        if acc_str is None:
            raise SystemExit("register: --min-accuracy requires --accuracy/--accuracy-file")
        if float(acc_str) <= args.min_accuracy:
            print(f"register: accuracy {acc_str} <= threshold {args.min_accuracy} — skipping")
            return

    uri = f"runs:/{run_id}/{args.artifact_path}"
    print(f"register: acc={acc_str} registering {uri} as {args.model_name!r}")
    res = mlflow.register_model(uri, args.model_name)
    print(f"register: {args.model_name} version {res.version}")


if __name__ == "__main__":
    main()