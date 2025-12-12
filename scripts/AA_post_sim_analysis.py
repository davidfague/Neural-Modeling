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
from Modules.logger import Logger

def run_post_analysis(sim_dir: str):
    print(f"\n[AA_post_analysis.py] --- Running post-analysis on {sim_dir} ---", flush=True)

    logger = Logger(sim_dir)
    logger.start_timer("total_post_analysis")

    # --- Quick soma voltage plot ---
    logger.start_timer("soma_voltage_plot")
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
        logger.log_runtime("AA_post_sim_analysis", "soma_voltage_plot", timer_name="soma_voltage_plot")
    except Exception as e:
        print(f"[AA_post_analysis.py] Warning: Soma voltage plot failed: {e}")

    # --- Run subprocess analyses ---
    parameters = analysis.DataReader.load_parameters(sim_dir)

    # Dendritic event detection + histograms
    if getattr(parameters, "record_all_channels", False) and getattr(parameters, "record_all_synapses", False):
        logger.start_timer("find_events")
        subprocess.run(["python", "../scripts/find_events_ben.py", "-d", sim_dir], check=False)
        logger.log_runtime("AA_post_sim_analysis", "find_events", timer_name="find_events")
        
        logger.start_timer("event_histograms")
        subprocess.run(["python", "../Modules/event_histograms.py", "-d", sim_dir], check=False)
        logger.log_runtime("AA_post_sim_analysis", "event_histograms", timer_name="event_histograms")

    # Voltage plots
    logger.start_timer("plot_voltages")
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "plot_voltages.py"),
                    "-d", sim_dir],
                   check=False)
    logger.log_runtime("AA_post_sim_analysis", "plot_voltages", timer_name="plot_voltages")

    # Drew's analysis
    logger.start_timer("drew_analysis")
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "drew_analysis.py"),
                    "-d", sim_dir],
                   check=False)
    logger.log_runtime("AA_post_sim_analysis", "drew_analysis", timer_name="drew_analysis")

    # # STA
    # subprocess.run(["python", "../scripts/plot_sta.py", "-d", sim_dir, "-s"], check=False)

    # --- Clustering analysis (generate summary report) ---
    logger.start_timer("analyze_clustering")
    try:
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__), "analyze_clustering.py"),
                        "-d", sim_dir],
                       check=False)
        logger.log_runtime("AA_post_sim_analysis", "analyze_clustering", timer_name="analyze_clustering")
    except Exception as e:
        print(f"[AA_post_analysis.py] Warning: clustering analysis script failed: {e}")

    # --- Spike-synchrony analysis (save figures + CSV) ---
    # DISABLED for CA tuning - takes 2+ hours per sim (68% of total runtime)
    # To re-enable: uncomment lines below
    # logger.start_timer("spike_synchrony")
    # try:
    #     subprocess.run([sys.executable,
    #                     os.path.join(os.path.dirname(__file__), "run_spike_synchrony.py"),
    #                     "-d", sim_dir],
    #                    check=False)
    #     logger.log_runtime("AA_post_sim_analysis", "spike_synchrony", timer_name="spike_synchrony")
    # except Exception as e:
    #     print(f"[AA_post_analysis.py] Warning: spike synchrony script failed: {e}")

    logger.log_runtime("AA_post_sim_analysis", "total_post_analysis", timer_name="total_post_analysis")
    print(f"[AA_post_analysis.py] Finished post-analysis for {sim_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run post-simulation analyses for one sim_dir")
    parser.add_argument("-d", "--dir", required=True, help="Path to simulation directory")
    args = parser.parse_args()
    run_post_analysis(args.dir)
