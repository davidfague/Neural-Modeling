#!/usr/bin/env python3
"""
scripts/delete_saved_time_data_for_sim.py

Delete per-simulation time-series / bulky intermediate data after analysis finishes.

Default target:
  <sim_dir>/save_data

Usage:
  python3 scripts/delete_saved_time_data_for_sim.py /path/to/sim_dir
  python3 scripts/delete_saved_time_data_for_sim.py /path/to/sim_dir --subdir save_data --dry-run
  python3 scripts/delete_saved_time_data_for_sim.py /path/to/sim_dir --subdir raw_data --keep-files v.h5
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def delete_saved_time_data_for_sim(
    sim_dir: str | os.PathLike,
    subdir: str = "raw_data",
    *,
    keep_files: list[str] | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Deletes <sim_dir>/<subdir> if it exists, optionally keeping specified files.

    Args:
        sim_dir: Path to the simulation directory
        subdir: Name of subdirectory to clean/delete
        keep_files: If provided, keep these files in saved_at_step_* folders and delete everything else.
                   If None, delete the entire subdirectory.
        dry_run: If True, only show what would be deleted
        verbose: Print detailed output

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

    # If keep_files specified, selectively delete in saved_at_step_* folders
    if keep_files:
        keep_files_set = set(keep_files)
        saved_folders = sorted(target.glob("saved_at_step_*"))
        saved_folders = [f for f in saved_folders if f.is_dir()]
        
        if not saved_folders:
            if verbose:
                print(f"[cleanup] No saved_at_step_* folders found in {target}", flush=True)
            return False
        
        if verbose:
            print(f"[cleanup] Found {len(saved_folders)} saved_at_step_* folders", flush=True)
            print(f"[cleanup] Will keep: {', '.join(keep_files)}", flush=True)
        
        if dry_run:
            if verbose:
                print(f"[cleanup] DRY RUN - would clean {len(saved_folders)} folders", flush=True)
            return False
        
        total_files_deleted = 0
        total_dirs_deleted = 0
        
        for folder in saved_folders:
            if verbose:
                print(f"[cleanup] Processing: {folder.name}", flush=True)
            
            for item in folder.iterdir():
                if item.name in keep_files_set:
                    if verbose:
                        print(f"[cleanup]   Keeping: {item.name}", flush=True)
                    continue
                
                if item.is_file():
                    item.unlink()
                    total_files_deleted += 1
                    if verbose:
                        print(f"[cleanup]   Deleted file: {item.name}", flush=True)
                elif item.is_dir():
                    shutil.rmtree(item)
                    total_dirs_deleted += 1
                    if verbose:
                        print(f"[cleanup]   Deleted directory: {item.name}/", flush=True)
        
        if verbose:
            print(f"[cleanup] Cleaned {len(saved_folders)} folders: deleted {total_files_deleted} files and {total_dirs_deleted} directories", flush=True)
        return True
    
    else:
        # Delete entire directory (original behavior)
        if dry_run:
            if verbose:
                print(f"[cleanup] DRY RUN would delete: {target}", flush=True)
            return False
        
        shutil.rmtree(target)
        if verbose:
            print(f"[cleanup] Deleted: {target}", flush=True)
        return True


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_dir", help="Path to a simulation directory")
    parser.add_argument("--subdir", default="save_data", help="Subdirectory to delete (default: save_data)")
    parser.add_argument("--keep-files", nargs="+", help="Files to keep in saved_at_step_* folders (e.g., v.h5)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    deleted = delete_saved_time_data_for_sim(
        args.sim_dir,
        subdir=args.subdir,
        keep_files=args.keep_files,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    return 0 if (deleted or args.dry_run) else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
