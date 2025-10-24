# AA_sim_then_post_parallel.py
# Minimal pipeline: for each sim_dir -> run simulation, then post-process.
# Parallelized using multiprocessing.Pool with map_async/close/join (like your originals).

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
    print(f"\n=== {sim_dir} ===", flush=True)

    # --- Load params & build cell ---
    parameters = analysis.DataReader.load_parameters(sim_dir)
    pssg = PreSimSynapseGenerator(sim_dir)
    cell = pssg.build_synapses_onto_cell_obj()

    # --- Run simulation ---
    sim = Simulation(getattr(SkeletonCell, parameters.skeleton_cell_type), create_dir=False)
    sim.path = os.path.split(sim_dir)[0]   # parent sims_dir
    sim.logger = Logger(sim_dir)
    sim.run_single_simulation(parameters=parameters, cell=cell)

    # --- Post-processing (same calls you already use) ---

    # Quick soma voltage PNG (optional; keep minimal & non-fatal on error)
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        v = analysis.DataReader.read_data(sim_dir, 'v')  # shape [nseg, nt]
        t = np.arange(0, parameters.h_tstop + parameters.h_dt, parameters.h_dt)
        os.makedirs(os.path.join(sim_dir, "figs"), exist_ok=True)
        plt.figure()
        plt.plot(t[:v.shape[1]], v[0, :len(t)])
        plt.xlabel('Time (ms)'); plt.ylabel('Voltage (mV)')
        plt.title('Soma voltage (seg 0)'); plt.tight_layout()
        plt.savefig(os.path.join(sim_dir, "figs", "soma_voltage.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Dendritic event detection + histograms if flags allow
    if getattr(parameters, "record_all_channels", False) and getattr(parameters, "record_all_synapses", False):
        subprocess.run(["python", "../scripts/find_events_ben.py", "-d", sim_dir], check=False)
        subprocess.run(["python", "../Modules/event_histograms.py", "-d", sim_dir], check=False)

    # Voltage plots (new)
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "plot_voltages.py"),
                    "-d", sim_dir],
                   check=False)
    # STA
    subprocess.run(["python", "../scripts/plot_sta.py", "-d", sim_dir, "-s"], check=False)

    # Drew's analysis
    subprocess.run([sys.executable,
                os.path.join(os.path.dirname(__file__), "drew_analysis.py"),
                "-d", sim_dir],
               check=False)
    
    print(f"[AA_sim_then_post_parallel.py] Finished Processing: {sim_dir}", flush=True)



if __name__ == "__main__":
    # ====== EDIT THIS TO YOUR PARENT SIMS FOLDER ======
    sims_dir = "/home/drfrbc/Neural-Modeling/simulations/2025-10-24-14-42-1.0xNexusInhFrom2x_0.25xNexusExc_1.3xTuftExc"
    # ==================================================

    # Max number of parallel processes
    N_PROCESSES = 6

    sim_dirs = [
        os.path.join(sims_dir, d)
        for d in sorted(os.listdir(sims_dir))
        if os.path.isdir(os.path.join(sims_dir, d))
    ]

    if not sim_dirs:
        print(f"No simulation subfolders found in: {sims_dir}")
        sys.exit(0)

    if len(sim_dirs) < N_PROCESSES:
        N_PROCESSES = len(sim_dirs)

    print(f"Running {len(sim_dirs)} simulations from: {sims_dir}")
    print(f"Using {N_PROCESSES} processes...\n")

    from multiprocessing import Pool
    pool = Pool(processes=N_PROCESSES)
    try:
        # map_async + close + join (your pattern)
        pool.map_async(run_one, sim_dirs)
        pool.close()
        pool.join()
    finally:
        # In case of early exceptions
        try:
            pool.terminate()
        except Exception:
            pass

    print(f"\n[AA_sim_then_post_parallel.py] Finished Processing all simulations in {sims_dir}.", flush=True)
