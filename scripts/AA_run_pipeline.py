#!/usr/bin/env python3
# scripts/AA_run_pipeline.py
#
# 1) Run pre-sim pipeline via scripts/AA_pre_sim.
# 2) Pull sims_dir from AA_pre_sim.simulator.
# 3) Run sim and analysis using scripts/AA_sim_then_post_parallel.py on sims_dir.

import os
import sys
import AA_pre_sim
from multiprocessing import set_start_method

def main() -> None:
    # Ensure sibling scripts and Modules import cleanly
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
    for p in (REPO_ROOT, THIS_DIR, os.path.join(REPO_ROOT, "Modules")):
        if p not in sys.path:
            sys.path.append(p)

    # --- Step 1: PRE-SIM (executes AA_pre_sim's top-level code) ---
    simulator = AA_pre_sim.run_pre_sim()
    sims_dir = simulator.sims_dir

    print(f"\n[AA_run_pipeline] Using sims_dir: {sims_dir}\n", flush=True)

    # Build list of simulation subfolders
    sim_dirs = [
        os.path.join(sims_dir, d)
        for d in sorted(os.listdir(sims_dir))
        if os.path.isdir(os.path.join(sims_dir, d))
    ]
    if not sim_dirs:
        print(f"[AA_run_pipeline] No simulation subfolders found in: {sims_dir}")
        return

    # --- Step 3: SIM + POST ---
    # IMPORTANT: import the correct module name here
    import AA_sim_then_post_parallel as simpost  # or rename file to AA_sim_then_post.py

    # Pick your parallelism level
    N_PROCESSES = min(6, len(sim_dirs))

    print(f"[AA_run_pipeline] Running {len(sim_dirs)} simulations from: {sims_dir}")
    print(f"[AA_run_pipeline] Using {N_PROCESSES} processes...\n", flush=True)

    from multiprocessing import Pool
    # Simpler/safer than map_async: map blocks until done and surfaces exceptions immediately
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(simpost.run_one, sim_dirs)

    print("\n[AA_run_pipeline] Finished pre-sim, simulations, and post-processing.\n")

if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()