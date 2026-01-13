# ============================================
# post_sim_analysis.py
# Runs post-simulation analysis on one sim_dir
# example usage: python post_sim_analysis.py -d /path/to/sim_dir
# ============================================

import os
import sys
import subprocess

sys.path.append('..')
import Modules.post_sim.analysis as analysis
from Modules.logger import Logger
from Modules.post_sim.post_sim_funcs import plot_soma_voltage

def run_post_analysis(sim_dir: str):
    print(f"\n[post_sim_analysis.py] --- Running post-analysis on {sim_dir} ---", flush=True)

    logger = Logger(sim_dir)
    logger.start_timer("total_post_analysis")

    # --- Quick soma voltage plot ---
    logger.start_timer("soma_voltage_plot")
    try:
        plot_soma_voltage(sim_dir)
        logger.log_runtime("post_sim_analysis", "soma_voltage_plot", timer_name="soma_voltage_plot")
    except Exception as e:
        print(f"[post_sim_analysis.py] Warning: Soma voltage plot failed: {e}")

    # --- Run subprocess analyses ---
    parameters = analysis.DataReader.load_parameters(sim_dir)

    # Dendritic event detection + histograms
    if getattr(parameters, "record_all_channels", False) and getattr(parameters, "record_all_synapses", False):
        logger.start_timer("find_events")
        subprocess.run(["python", "../Modules/dendritic_spikes/find_events_ben.py", "-d", sim_dir], check=False)
        logger.log_runtime("post_sim_analysis", "find_events", timer_name="find_events")
        
        logger.start_timer("event_histograms")
        subprocess.run(["python", "../Modules/dendritic_spikes/event_histograms.py", "-d", sim_dir], check=False)
        logger.log_runtime("post_sim_analysis", "event_histograms", timer_name="event_histograms")

    # Voltage plots (including spike-centered plots for NMDA/Na/Ca)
    logger.start_timer("plot_voltages")
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "../Modules/post_sim/voltages/plot_voltages.py"),
                    "-d", sim_dir,
                    "--no-events"],  # Skip event detection since it's already run above
                   check=False)
    logger.log_runtime("post_sim_analysis", "plot_voltages", timer_name="plot_voltages")

    # Drew's analysis
    logger.start_timer("drew_analysis")
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(__file__), "../Modules/dendritic_spikes/drew_analysis.py"),
                    "-d", sim_dir],
                   check=False)
    logger.log_runtime("post_sim_analysis", "drew_analysis", timer_name="drew_analysis")

    # # STA
    # subprocess.run(["python", "../scripts/plot_sta.py", "-d", sim_dir, "-s"], check=False)

    # --- Clustering analysis (generate summary report) ---
    logger.start_timer("analyze_clustering")
    try:
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__), "../Modules/clustering/analyze_clustering.py"),
                        "-d", sim_dir],
                       check=False)
        logger.log_runtime("post_sim_analysis", "analyze_clustering", timer_name="analyze_clustering")
    except Exception as e:
        print(f"[post_sim_analysis.py] Warning: clustering analysis script failed: {e}")

    # --- Spike-synchrony analysis (save figures + CSV) ---
    # DISABLED for CA tuning - takes 2+ hours per sim (68% of total runtime)
    # To re-enable: uncomment lines below
    # logger.start_timer("spike_synchrony")
    # try:
    #     subprocess.run([sys.executable,
    #                     os.path.join(os.path.dirname(__file__), "run_spike_synchrony.py"),
    #                     "-d", sim_dir],
    #                    check=False)
    #     logger.log_runtime("post_sim_analysis", "spike_synchrony", timer_name="spike_synchrony")
    # except Exception as e:
    #     print(f"[post_sim_analysis.py] Warning: spike synchrony script failed: {e}")

    logger.log_runtime("post_sim_analysis", "total_post_analysis", timer_name="total_post_analysis")
    print(f"[post_sim_analysis.py] Finished post-analysis for {sim_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run post-simulation analyses for one sim_dir")
    parser.add_argument("-d", "--dir", required=True, help="Path to simulation directory")
    args = parser.parse_args()
    run_post_analysis(args.dir)
