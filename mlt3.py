#!/usr/bin/env python3
"""MLT-03 preprocessing — raw rows in, a new platform dataset version out.

The data-engineer's job: clean the input and publish a reproducible version
of a platform dataset. The point of the case is HOW that version is created:
by writing a NEW FOLDER into the mounted dataset. There is no SDK here, no
token, no HTTP call — the container only writes files. A background CP task
(``dataset_version_sync``) notices the folder once its contents stay stable
between two passes and registers it as a version, so publishing does not
depend on the workload finishing. That is why the same script works from a
batch Job and from a Jupyter terminal alike.

Input is optional on purpose: with a raw mount present the real CSV is read,
without one 500 wine-quality rows are synthesised (~8% missing cells) so the
case runs on a stand with no external bucket.

Config is argparse, not os.environ — the platform's env is passed INTO the
arguments at the manifest level. Keeps the script runnable by hand:

    python mlt3.py --out-base ./out --rows 50
"""
import argparse
import csv
import glob
import os
import random
import time

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol",
]
COLS = FEATURES + ["quality"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-base", required=True,
        help="Mount point of the dataset to publish into. The platform states "
             "it in AIR_DATASET_<SLUG>_PATH; pass that through in the manifest.",
    )
    p.add_argument(
        "--raw-dir", default="/mnt/raw",
        help="Where to look for input *.csv. Missing or empty → synthesise.",
    )
    p.add_argument(
        "--version-name", default=None,
        help="Name of the folder to create (default: UTC timestamp). A NEW "
             "folder is what makes a NEW version, so reruns must not collide.",
    )
    p.add_argument("--rows", type=int, default=500,
                   help="Rows to synthesise when there is no real input.")
    p.add_argument("--missing-rate", type=float, default=0.08,
                   help="Share of synthesised rows given one empty cell.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the synthetic input, so a rerun is comparable.")
    p.add_argument("--delimiter", default=";",
                   help="Delimiter of the input CSV (wine-quality ships ';').")
    return p.parse_args()


def read_raw(raw_dir: str, delimiter: str) -> list[dict] | None:
    """Return rows from the first CSV in ``raw_dir``, or None if there is none."""
    hits = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not hits:
        return None
    print(f"input: {hits[0]}")
    with open(hits[0], newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def synthesise(rows: int, missing_rate: float, seed: int) -> list[dict]:
    """Stand-in input so the case needs no external bucket."""
    print(f"input: synthesised ({rows} rows, ~{missing_rate:.0%} missing)")
    random.seed(seed)
    out = []
    for _ in range(rows):
        row = {c: round(random.uniform(0.1, 14.0), 3) for c in FEATURES}
        row["quality"] = random.randint(3, 8)
        if random.random() < missing_rate:
            row[random.choice(FEATURES)] = ""
        out.append(row)
    return out


def clean(rows: list[dict]) -> list[dict]:
    """Drop rows with any empty cell — the whole point of the preprocessing."""
    return [
        r for r in rows
        if all(str(r.get(c, "")).strip() != "" for c in COLS)
    ]


def main() -> None:
    args = parse_args()

    rows = read_raw(args.raw_dir, args.delimiter)
    if rows is None:
        rows = synthesise(args.rows, args.missing_rate, args.seed)

    before = len(rows)
    cleaned = clean(rows)
    after = len(cleaned)
    print(f"rows before={before} after={after}")
    if not after:
        raise SystemExit("nothing left after cleaning — refusing to publish an empty version")

    # A NEW folder is a NEW version. Written in one pass so the sync task sees
    # stable contents on its next visit rather than a half-written directory.
    name = args.version_name or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir = os.path.join(args.out_base, name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "wine.csv")
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for r in cleaned:
            writer.writerow({c: r[c] for c in COLS})

    print(f"wrote {out_file} rows={after}")


if __name__ == "__main__":
    main()
