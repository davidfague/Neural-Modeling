#!/usr/bin/env python3
# plot_voltages.py
#
# Usage (like plot_sta.py):
#   python plot_voltages.py -d <sim_directory> [--no-events] [--window-ms 1000]
#                           [--basal-count 6] [--show] [--seed 0]
#
# Notes:
# - By default, this will (re)run find_events_ben.py to generate ca/nmda/na .csvs.
#   Add --no-events to skip that step.
# - Plots are saved under <sim_directory>/voltages/ :
#     - soma_voltage_init.png
#     - soma_voltage.png
#     - apic_segs.png
#     - apical_voltages.png
#     - basal_segs.png
#     - basal_voltages.png
#
# - xlimits follow your original code style:
#     xlimits = [h_tstop*h_dt - (window_ms/h_dt), h_tstop*h_dt]
#   (kept as-is to match your Modules.plot_voltage expectations.)

import os
import sys
import argparse
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Repo-relative imports (match other scripts)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
for p in (REPO_ROOT, THIS_DIR, os.path.join(REPO_ROOT, "Modules")):
    if p not in sys.path:
        sys.path.append(p)

# Now import your modules
from Modules.analysis import load_sim, DataReader
from Modules.plot_morphology import plot_segments, plot
from Modules.plot_voltage import plot_voltage

OVERWRITE = False

def run_find_events(sim_dir: str) -> None:
    """Run find_events_ben.py to generate ca.csv / nmda.csv / na.csv."""
    script = os.path.join(THIS_DIR, "find_events_ben.py")
    if not os.path.exists(script):
        # Fallback: maybe it’s in scripts/ sibling (when executed from elsewhere)
        alt = os.path.join(REPO_ROOT, "scripts", "find_events_ben.py")
        script = alt if os.path.exists(alt) else script

    if not os.path.exists(script):
        warnings.warn(f"[scripts/plot_voltages.py] Could not find find_events_ben.py at {script}. Skipping event generation.")
        return

    try:
        subprocess.run([sys.executable, script, "-d", sim_dir], check=False)
    except Exception as e:
        warnings.warn(f"[scripts/plot_voltages.py] find_events_ben.py failed: {e}")


def safe_read_event_csvs(sim_dir: str) -> dict:
    """Try to read ca.csv, nmda.csv, na.csv if present."""
    files = {
        "ca":   os.path.join(sim_dir, "ca.csv"),
        "nmda": os.path.join(sim_dir, "nmda.csv"),
        "na":   os.path.join(sim_dir, "na.csv"),
    }
    dfs = {}
    for k, f in files.items():
        if os.path.exists(f):
            try:
                dfs[k] = pd.read_csv(f)
            except Exception as e:
                warnings.warn(f"[scripts/plot_voltages.py] Failed to read {f}: {e}")
    return dfs


def first_soma_segment(seg_df: pd.DataFrame) -> int | None:
    soma = seg_df[seg_df["Type"] == "soma"]
    if soma.empty:
        return None
    return int(soma.segmentID.iloc[0])


def choose_basal_segments(seg_df: pd.DataFrame, n: int, seed: int | None) -> list[int]:
    dend = seg_df[seg_df["Type"] == "dend"]
    ids = dend.segmentID.to_numpy(dtype=int)
    if ids.size == 0:
        return []
    rng = np.random.default_rng(seed)
    if n >= ids.size:
        return list(ids)
    return list(rng.choice(ids, size=n, replace=False))


def filter_existing(ids: list[int], avail: set[int]) -> list[int]:
    present = [i for i in ids if i in avail]
    missing = sorted(set(ids) - set(present))
    if missing:
        warnings.warn(f"[scripts/plot_voltages.py] Skipping missing segment IDs: {missing}")
    return present

def plot_mean_voltage(seg_data, sim_data, sim_directory):
    seg_data['mean_v'] = sim_data['v'].mean(axis=0)
    from Modules import plot_morphology
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plot_morphology.plot(seg_data,  seg_data['mean_v'], ax, clims = [min(seg_data['mean_v']), max(seg_data['mean_v'])], radius_scale=1.5)
    fig = plot(seg_data,  seg_data['mean_v'], ax, clims = [-80, -10], radius_scale=1.5)
    ax.clabel('mean voltage')
    plt.savefig(os.path.join(sim_directory, 'voltages', 'mean_v_morphology'))


def main():
    ap = argparse.ArgumentParser(description="Plot soma and dendritic voltages for a single simulation directory.")
    ap.add_argument("-d", "--dir", required=True, help="Path to a simulation directory (contains parameters.pkl, data, etc.)")
    ap.add_argument("--no-events", action="store_true", help="Skip running find_events_ben.py")
    ap.add_argument("--window-ms", type=float, default=1000.0, help="Window length (ms) near the end of the sim for detailed plots (default: 1000)")
    ap.add_argument("--basal-count", type=int, default=6, help="Number of random basal (Type='dend') segments to plot (default: 6)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for selecting basal segments (default: 0)")
    ap.add_argument("--show", action="store_true", help="Show matplotlib windows (in addition to saving)")
    ap.add_argument("--dpi", type=int, default=300, help="Saved figure DPI (default: 300)")
    args = ap.parse_args()

    sim_dir = os.path.abspath(args.dir)
    if not os.path.isdir(sim_dir):
        raise SystemExit(f"[scripts/plot_voltages.py] Not a directory: {sim_dir}")

    out_dir = os.path.join(sim_dir, "voltages")
    os.makedirs(out_dir, exist_ok=True)

    # Optionally generate dendritic event CSVs
    if not args.no_events:
        run_find_events(sim_dir)

    # Load data
    parameters, sim_data, seg_data, elec_dist = load_sim(sim_dir)

    # Read event CSVs if present
    dendritic_dfs = safe_read_event_csvs(sim_dir)
    dendritic_dfs = dendritic_dfs if dendritic_dfs else None

    # ------------- Soma quick plot (init-style) -------------
    try:
        # Build time_points similar to your original snippet
        n_steps = int(parameters.h_tstop / parameters.h_dt)
        time_points = np.arange(0, n_steps)
        # If extremely long, emulate your example slice (safe bound)
        if time_points.size > 20000:
            time_points = time_points[:19999]

        plt.figure(figsize=(20, 6))
        plt.plot(sim_data["v"][time_points, 0], "k-")
        plt.ylim([-90, 10])
        plt.title(f"SOMA Voltage (segment 0)")
        plt.xlabel(f"Timesteps (dt={parameters.h_dt} ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "soma_voltage_init.png"), dpi=args.dpi, bbox_inches="tight")
        if args.show:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        warnings.warn(f"[scripts/plot_voltages.py] Soma quick plot failed: {e}")

    # Compute x-limits following your original convention (kept exactly):
    #   xlimits = [h_tstop*h_dt - (window_ms/h_dt), h_tstop*h_dt]
    # This matches your previous Modules.plot_voltage usage.
    try:
        xlimits = [
            parameters.h_tstop / parameters.h_dt - (args.window_ms / parameters.h_dt),
            parameters.h_tstop / parameters.h_dt,
        ]

    except Exception:
        # Fallback if attrs are weird; just skip xlims
        xlimits = None

    # ------------- Soma plot via plot_voltage -------------
    soma_idx = first_soma_segment(seg_data)
    if soma_idx is not None:
        try:
            plot_voltage(
                sim_data,
                [soma_idx],
                ["k"],
                xlims=xlimits,
                title_suffix="(Soma)",
                save_file=os.path.join(out_dir, "soma_voltage"),
                show=args.show,
            )
        except Exception as e:
            warnings.warn(f"[scripts/plot_voltages.py] plot_voltage (soma) failed: {e}")
    else:
        warnings.warn("[scripts/plot_voltages.py] No soma segment found; skipping soma plot.")

    # ------------- Apical plot (predefined IDs, filtered to existing) -------------
    # Curate set with colors and label suffixes:
    if not hasattr(parameters, "plot_voltages_apic_segment_dict") or OVERWRITE:  # fall back on default
        parameters.plot_voltages_apic_segment_dict = {
            1647: {"color": "black",  "description": "[most Ca spikes]"},
            1554: {"color": "green",  "description": "[should be used in nexus elec_distance calc]"},
            1547: {"color": "orange", "description": "[used in nexus elec_distance calc]"},
            1842: {"color": "red",    "description": "[right tuft dendrite (halfway)]"},
            1210: {"color": "lime",   "description": "[oblique (middle, near nexus)]"},
            1080: {"color": "green",  "description": "[apical trunk (very near soma)]"},
            1900: {"color": "black",  "description": ""},
            2000: {"color": "tab:blue","description": ""},
            1760: {"color": "g",      "description": ""},
            1340: {"color": "m",      "description": ""},
        }
    apic_df = seg_data[seg_data["Type"] == "apic"]
    apic_ids_available = set(map(int, apic_df.segmentID.to_numpy(dtype=int)))
    apic_ids = filter_existing(list(parameters.plot_voltages_apic_segment_dict.keys()), apic_ids_available)

    if apic_ids:
        apic_colors = [parameters.plot_voltages_apic_segment_dict[i]["color"] for i in apic_ids]
        apic_suffixes = [parameters.plot_voltages_apic_segment_dict[i]["description"] for i in apic_ids]
        # segment locations figure
        try:
            plot_segments(
                apic_df,
                apic_ids,
                apic_colors,
                title_suffix="(Apical)",
                save_file=os.path.join(out_dir, "apic_segs"),
                show=args.show,
                label_special_ids=True,
            )
        except Exception as e:
            warnings.warn(f"[scripts/plot_voltages.py] plot_segments (apic) failed: {e}")
        # voltage traces figure
        try:
            plot_voltage(
                sim_data,
                apic_ids,
                apic_colors,
                xlims=xlimits,
                title_suffix="(Apical)",
                dendritic_dfs=dendritic_dfs,
                additional_title_suffixes=apic_suffixes,
                save_file=os.path.join(out_dir, "apical_voltages"),
                show=args.show,
            )
        except Exception as e:
            warnings.warn(f"[scripts/plot_voltages.py] plot_voltage (apic) failed: {e}")
    else:
        warnings.warn("[scripts/plot_voltages.py] No requested apical segment IDs were found; skipping apical plots.")

    # ------------- Basal/dendritic plot (random sample) -------------
    dend_df = seg_data[seg_data["Type"] == "dend"]
    dend_ids = choose_basal_segments(seg_data, args.basal_count, args.seed)
    if dend_ids:
        # Reuse the apic colors list style or just cycle
        color_cycle = ["r", "g", "b", "m", "y", "k"]
        dend_colors = [color_cycle[i % len(color_cycle)] for i in range(len(dend_ids))]
        # segment locations figure
        try:
            plot_segments(
                dend_df,
                dend_ids,
                dend_colors,
                title_suffix="(Dendritic)",
                save_file=os.path.join(out_dir, "basal_segs"),
                show=args.show,
                label_special_ids=True,
            )
        except Exception as e:
            warnings.warn(f"[scripts/plot_voltages.py] plot_segments (dend) failed: {e}")
        # voltage traces figure
        try:
            plot_voltage(
                sim_data,
                dend_ids,
                dend_colors,
                xlims=xlimits,
                title_suffix="(Dendritic)",
                dendritic_dfs=dendritic_dfs,
                save_file=os.path.join(out_dir, "basal_voltages"),
                show=args.show,
            )
        except Exception as e:
            warnings.warn(f"[scripts/plot_voltages.py] plot_voltage (dend) failed: {e}")
    else:
        warnings.warn("[scripts/plot_voltages.py] No basal/dendritic segments available to plot.")

    print(f"[scripts/plot_voltages.py] Done. Figures saved under: {out_dir}")


    plot_mean_voltage(seg_data, sim_data, sim_dir)

if __name__ == "__main__":
    main()
