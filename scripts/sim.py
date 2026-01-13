'''
sim.py
Run a single NEURON simulation for a given sim_dir.
This is the simulation-only portion, extracted from sim_then_post_parallel.py
'''

import os
import sys

# Repo-relative imports
sys.path.append('..')
sys.path.append('../Modules')

from Modules.sim.sim_funcs import run_single_sim
from Modules.logger import Logger


def run_simulation(sim_dir: str):
    """
    Execute a single NEURON simulation.
    
    Args:
        sim_dir: Path to simulation directory containing parameters.pickle and synapses.csv
    """
    print(f"\n[scripts/sim.py] === Running simulation: {sim_dir} ===", flush=True)

    logger = Logger(sim_dir)
    logger.start_timer("simulation")

    # Run simulation
    run_single_sim(sim_dir, logger=logger)
    
    logger.log_runtime("sim", "simulation", timer_name="simulation")
    print(f"[sim.py] Finished simulation: {sim_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NEURON simulation for one sim_dir")
    parser.add_argument("-d", "--dir", required=True, help="Path to simulation directory")
    args = parser.parse_args()
    
    run_simulation(args.dir)
