#!/usr/bin/env python3
# scripts/AA_run_pipeline.py
#
# 1) Run pre-sim pipeline via scripts/AA_pre_sim.
# 2) Pull sims_dir from AA_pre_sim.simulator.
# 3) Run sim and post-analysis in parallel, pairing each sim_dir's simulation
#    and analysis into one sequential process.

import os
import sys
import pre_sim as pre_sim
from multiprocessing import set_start_method

def run_sim_and_analysis(sim_dir: str):
    """
    Run simulation followed by post-analysis for a single sim_dir.
    This ensures analysis starts immediately after simulation completes.
    
    Args:
        sim_dir: Path to simulation directory
    """
    from Modules.logger import Logger
    import scripts.sim as sim
    import scripts.post_sim_analysis as post_sim
    
    logger = Logger(sim_dir)
    logger.start_timer("sim_and_analysis_paired")
    
    # Run simulation
    sim.run_simulation(sim_dir)
    
    # Run post-analysis immediately after
    post_sim.run_post_analysis(sim_dir)
    
    logger.log_runtime("run_pipeline", "sim_and_analysis_paired", timer_name="sim_and_analysis_paired")
    print(f"[run_pipeline] Completed sim+analysis for: {sim_dir}", flush=True)

def main() -> None:
    from Modules.logger import Logger
    
    # Ensure sibling scripts and Modules import cleanly
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
    for p in (REPO_ROOT, THIS_DIR, os.path.join(REPO_ROOT, "Modules")):
        if p not in sys.path:
            sys.path.append(p)

    # Temporary logger for pipeline timing (don't have simulation folders yet)
    temp_logger = Logger()
    temp_logger.start_timer("total_pipeline")

    # run AA_pre_sim.py
    temp_logger.start_timer("pre_sim_stage")
    simulator = pre_sim.run_pre_sim()
    sims_dir = simulator.sims_dir
    temp_logger.log_runtime("AA_run_pipeline", "pre_sim_stage", timer_name="pre_sim_stage")

    print(f"\n[AA_run_pipeline] Using sims_dir: {sims_dir}\n", flush=True)
    
    # Create proper logger with sims_dir (now that we have the simulation folders)
    logger = Logger(sims_dir)
    logger._timers = temp_logger._timers

    # Build list of simulation subfolders
    sim_dirs = [
        os.path.join(sims_dir, d)
        for d in sorted(os.listdir(sims_dir))
        if os.path.isdir(os.path.join(sims_dir, d))
    ]
    if not sim_dirs:
        print(f"[AA_run_pipeline] No simulation subfolders found in: {sims_dir}")
        return

    # For parallelization - each process runs one sim + its analysis
    N_PROCESSES = min(9, len(sim_dirs))

    print(f"[AA_run_pipeline] Running {len(sim_dirs)} simulations from: {sims_dir}")
    print(f"[AA_run_pipeline] Using {N_PROCESSES} processes (paired sim+analysis)...\n", flush=True)

    logger.start_timer("sim_and_post_stage")
    from multiprocessing import Pool
    with Pool(processes=N_PROCESSES) as pool:
        pool.map(run_sim_and_analysis, sim_dirs)

    logger.log_runtime("AA_run_pipeline", "sim_and_post_stage", timer_name="sim_and_post_stage")
    logger.log_runtime("AA_run_pipeline", "total_pipeline", timer_name="total_pipeline")

    print("\n[AA_run_pipeline] Finished pre-sim, simulations, and post-processing.\n")

if __name__ == "__main__":
    main()