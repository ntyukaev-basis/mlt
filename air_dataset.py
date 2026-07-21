"""Locating the right file inside a mounted platform dataset.

Shared because getting this wrong is subtle and has already bitten twice. A
dataset can be mounted two ways, and the whole-dataset case is where it hurts:

* a PINNED version mounts its own contents  → ``<mount>/<name>``
* a WHOLE dataset mounts one folder PER VERSION → ``<mount>/<version>/<name>``

Folder names in the second case are NOT uniform. A version created by upload is
named after its version id (``de44fbb4-…``), one produced by a workload carries
whatever name that workload chose (``20260721-035614``, ``labeled-202607171240``).
So ``sorted(folders)[-1]`` does not mean "newest" — it means "whichever string
sorts highest", which on the DE stand silently picked a months-old uploaded
version with a different schema, and training died on a KeyError deep inside
pandas.

The platform states the location of the version it is tracking in
AIR_DATASET_STORAGE_URI. Trust that; fall back to newest BY MODIFICATION TIME,
which at least means what it says.
"""
import glob
import os

__all__ = ["resolve_csv", "require_columns"]


def resolve_csv(
    data_dir: str,
    file_name: str = "wine.csv",
    storage_uri: str | None = None,
) -> str:
    """Return the path to ``file_name`` inside a mounted dataset.

    ``storage_uri`` is the platform's AIR_DATASET_STORAGE_URI; its last path
    segment names the folder of the tracked version.
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
            f"and <dir>/*/{file_name}) — has the dataset a published version?"
        )
    return max(hits, key=os.path.getmtime)


def require_columns(df: object, columns: list[str], path: str) -> None:
    """Fail with the actual schema instead of a KeyError three frames down.

    Versions of one dataset need not agree on columns: an uploaded CSV and one
    produced by preprocessing are different things.
    """
    present = list(getattr(df, "columns", []))
    missing = [c for c in columns if c not in present]
    if missing:
        raise SystemExit(
            f"{path} has no column(s) {missing}; columns are: {present}. "
            "Wrong dataset version? Pin the intended one via "
            "experiment_tracking.dataset_version_id."
        )
