#!/usr/bin/env python3
"""
scripts/delete_saved_time_data_for_sim.py

Delete per-simulation time-series / bulky intermediate data after analysis finishes.

Default target:
  <sim_dir>/save_data

Usage:
  python3 scripts/delete_saved_time_data_for_sim.py /path/to/sim_dir
  python3 scripts/delete_saved_time_data_for_sim.py /path/to/sim_dir --subdir save_data --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def delete_saved_time_data_for_sim(
    sim_dir: str | os.PathLike,
    subdir: str = "save_data",
    *,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Deletes <sim_dir>/<subdir> if it exists.

    Returns:
        True if deletion happened, False if nothing was deleted (missing / skipped).
    """
    sim_dir_path = Path(sim_dir).expanduser().resolve()
    target = (sim_dir_path / subdir).resolve()

    # ---- safety checks (avoid catastrophes) ----
    if not sim_dir_path.exists():
        if verbose:
            print(f"[cleanup] sim_dir does not exist: {sim_dir_path}", flush=True)
        return False

    # Ensure target is actually inside sim_dir
    try:
        target.relative_to(sim_dir_path)
    except ValueError:
        raise RuntimeError(f"[cleanup] Refusing to delete outside sim_dir: {target}")

    # Refuse to delete if target is symlink (can be surprising)
    if target.is_symlink():
        raise RuntimeError(f"[cleanup] Refusing to delete symlink: {target}")

    if not target.exists():
        if verbose:
            print(f"[cleanup] Nothing to delete (missing): {target}", flush=True)
        return False

    if not target.is_dir():
        raise RuntimeError(f"[cleanup] Expected a directory, got: {target}")

    if dry_run:
        if verbose:
            print(f"[cleanup] DRY RUN would delete: {target}", flush=True)
        return False

    # Do the deletion
    shutil.rmtree(target)
    if verbose:
        print(f"[cleanup] Deleted: {target}", flush=True)
    return True


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir", help="Path to a simulation directory")
    parser.add_argument("--subdir", default="save_data", help="Subdirectory to delete (default: save_data)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    deleted = delete_saved_time_data_for_sim.py(
        args.sim_dir,
        subdir=args.subdir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    return 0 if (deleted or args.dry_run) else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
