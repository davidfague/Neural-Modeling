# ============================================
# AA_post_sim_analysis.py
# Runs post-simulation analysis on one sim_dir
# example usage: python AA_post_sim_analysis.py -d /path/to/sim_dir
# ============================================

import os
import sys
import subprocess

sys.path.append('..')
import Modules.analysis as analysis

def run_post_analysis(sim_dir: str):
    print(f"\n[AA_post_analysis.py] --- Running post-analysis on {sim_dir} ---", flush=True)

    # --- Quick soma voltage plot ---
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        parameters = analysis.DataReader.load_parameters(sim_dir)
        v = analysis.DataReader.read_data(sim_dir, 'v')  # shape [nseg, nt]
        t = np.arange(0, parameters.h_tstop + parameters.h_dt, parameters.h_dt)
        os.makedirs(os.path.join(sim_dir, "figs"), exist_ok=True)
        plt.figure()
        plt.plot(t[:v.shape[1]], v[0, :len(t)])
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('Soma voltage (seg 0)')
        plt.tight_layout()
        plt.savefig(os.path.join(sim_dir, "figs", "soma_voltage.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[AA_post_analysis.py] Warning: Soma voltage plot failed: {e}")

    # --- Run subprocess analyses ---
    parameters = analysis.DataReader.load_parameters(sim_dir)

    # Dendritic event detection + histograms
    if getattr(parameters, "record_all_channels", False) and getattr(parameters, "record_all_synapses", False):
        subprocess.run(["python", "../scripts/find_events_ben.py", "-d", sim_dir], check=False)
        subprocess.run(["python", "../Modules/event_histograms.py", "-d", sim_dir], check=False)

    # Voltage plots
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "plot_voltages.py"),
                    "-d", sim_dir],
                   check=False)

    # Drew’s analysis
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "drew_analysis.py"),
                    "-d", sim_dir],
                   check=False)

    # # STA
    # subprocess.run(["python", "../scripts/plot_sta.py", "-d", sim_dir, "-s"], check=False)

    print(f"[AA_post_analysis.py] Finished post-analysis for {sim_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run post-simulation analyses for one sim_dir")
    parser.add_argument("-d", "--dir", required=True, help="Path to simulation directory")
    args = parser.parse_args()
    run_post_analysis(args.dir)
