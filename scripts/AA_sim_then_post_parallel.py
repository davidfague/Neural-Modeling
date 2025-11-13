'''
AA_sim_then_post_parallel.py
simulate all simulations in a specified sims_dir (ex. ~/simulations/sims_dir)
then run analysis scripts: ["../scripts/find_events_ben.py", "../Modules/event_histograms.py", "../scripts/plot_voltages.py", "../scripts/plot_sta.py", "../scripts/drew_analysis.py"]
'''

import os
import sys
import subprocess

# Repo-relative imports (same as your scripts)
sys.path.append('..')
sys.path.append('../Modules')

from Modules.synapses_file import PreSimSynapseGenerator
from Modules.simulation_slurm import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.logger import Logger
import Modules.analysis as analysis


def run_one(sim_dir: str):
    print(f"\n[scripts/AA_sim_then_post_parallel.py] === {sim_dir} ===", flush=True)

    # --- Load params & build cell ---
    parameters = analysis.DataReader.load_parameters(sim_dir)
    pssg = PreSimSynapseGenerator(sim_dir)
    cell = pssg.build_synapses_onto_cell_obj()

    # --- Run simulation ---
    sim = Simulation(getattr(SkeletonCell, parameters.skeleton_cell_type), create_dir=False)
    sim.path = os.path.split(sim_dir)[0]   # parent sims_dir
    sim.logger = Logger(sim_dir)
    sim.run_single_simulation(parameters=parameters, cell=cell)

    # --- Post sim  analysis ---
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "AA_post_sim_analysis.py"),
                    "-d", sim_dir],
                   check=False)
    
    print(f"[AA_sim_then_post_parallel.py] Finished Processing: {sim_dir}", flush=True)



if __name__ == "__main__":
    # ====== EDIT THIS TO YOUR PARENT SIMS FOLDER ======
    sims_dir = "/home/drfrbc/Neural-Modeling/simulations/2025-10-27-19-14-2.5x_dec_tuft_inh_from_2.5x___3.0x_inc_nexus_inh_from_3x___2.5x_tuft_inh_from_2.5x___3.5x_inc_tuft_exc_from_3.0x"
    # ==================================================

    # Max number of parallel processes
    N_PROCESSES = 6

    sim_dirs = [
        os.path.join(sims_dir, d)
        for d in sorted(os.listdir(sims_dir))
        if os.path.isdir(os.path.join(sims_dir, d))
    ]

    if not sim_dirs:
        print(f"[scripts/AA_sim_then_post_parallel.py] No simulation subfolders found in: {sims_dir}")
        sys.exit(0)

    if len(sim_dirs) < N_PROCESSES:
        N_PROCESSES = len(sim_dirs)

    print(f"[scripts/AA_sim_then_post_parallel.py] Running {len(sim_dirs)} simulations from: {sims_dir}")
    print(f"[scripts/AA_sim_then_post_parallel.py] Using {N_PROCESSES} processes...\n")

    from multiprocessing import Pool
    pool = Pool(processes=N_PROCESSES)
    # try:
        # map_async + close + join (your pattern)
    pool.map_async(run_one, sim_dirs)
    pool.close()
    pool.join()
    # finally:
    #     # In case of early exceptions
    #     try:
    #         pool.terminate()
    #     except Exception:
    #         pass

    print(f"\n[AA_sim_then_post_parallel.py] Finished Processing all simulations in {sims_dir}.", flush=True)
