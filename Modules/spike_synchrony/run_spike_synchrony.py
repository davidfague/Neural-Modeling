#!/usr/bin/env python3
"""Run spike-synchrony analysis on one or more simulation directories.

Saves figures and a CSV summary into <sim_dir>/spike_synchrony/.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

# Ensure Modules is importable when run from scripts/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
MODULES_DIR = os.path.join(REPO_ROOT, "Modules")
for p in (REPO_ROOT, MODULES_DIR):
    if p not in sys.path:
        sys.path.append(p)

import Modules.spike_synchrony.spike_synchrony as ss  # type: ignore


def ensure_outdir(sim_dir: str) -> str:
    outdir = os.path.join(sim_dir, "spike_synchrony")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def run_for_dirs(sim_dirs: Iterable[str], outdir_base: str | None = None) -> pd.DataFrame:
    """Run COMPLETE full-resolution spike synchrony analysis."""
    sim_dirs = list(sim_dirs)
    print(f"[scripts/run_spike_synchrony] Loading synapses from {len(sim_dirs)} simulation(s)...")
    
    # Load parameters from first sim directory to get h_tstop
    param_path = os.path.join(sim_dirs[0], "parameters.pickle")
    parameters = None
    if os.path.isfile(param_path):
        with open(param_path, "rb") as f:
            parameters = pickle.load(f)
        print(f"[scripts/run_spike_synchrony] Loaded parameters from {sim_dirs[0]}")
    
    # Load all synapses - FULL RESOLUTION (no subsampling)
    syn_df = ss.load_synapses(sim_dirs)
    print(f"[scripts/run_spike_synchrony] Loaded {len(syn_df)} spike trains total")
    
    syn_df, (t_start, t_end) = ss.prepare_spike_trains(syn_df, t_start=0.0, min_spikes=1, parameters=parameters)
    print(f"[scripts/run_spike_synchrony] Prepared spike trains in time window [{t_start}, {t_end}]")
    
    syn_df = ss.assign_positions_within_sim(syn_df)
    print(f"[scripts/run_spike_synchrony] Building SPIKE-synchronization matrices (this may take time for large datasets)...")
    sync_mats = ss.build_sync_matrices_per_sim(syn_df)

    # Compute COMPLETE aggregated stats across all 6 grouping levels
    print("[scripts/run_spike_synchrony] Computing synchrony statistics at all hierarchical levels...")
    all_stats = ss.compute_all_stats(syn_df, sync_mats, drop_fg_negative=True)
    print(f"[scripts/run_spike_synchrony] Generated {len(all_stats)} summary statistics")

    # Determine output directory
    if len(sim_dirs) == 1:
        outdir = ensure_outdir(sim_dirs[0])
    else:
        outdir = ensure_outdir(outdir_base or sim_dirs[0])

    # Save sync matrix heatmap for each sim
    print("[scripts/run_spike_synchrony] Saving sync matrix heatmaps...")
    for sim_name in sync_mats:
        fig, ax = plt.subplots(figsize=(8, 8))
        ss.plot_sync_matrix(sync_mats, sim_name, ax=ax)
        p = os.path.join(outdir, f"sync_matrix_{sim_name}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[scripts/run_spike_synchrony]   Saved: {p}")

    # Create COMPREHENSIVE 6-panel visualization
    print("[scripts/run_spike_synchrony] Creating comprehensive 6-panel analysis figure...")
    fig = ss.plot_comprehensive_analysis(all_stats, figsize=(18, 10))
    outp = os.path.join(outdir, "sync_comprehensive_analysis.png")
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[scripts/run_spike_synchrony]   Saved: {outp}")

    # Create per-input_source figures
    print("[scripts/run_spike_synchrony] Creating per-input_source analysis figures...")
    per_source_figs = ss.plot_per_input_source_analysis(all_stats, figsize=(12, 8))
    for src_name, src_fig in per_source_figs.items():
        # Clean source name for filename
        safe_name = src_name.replace("/", "_").replace(" ", "_")
        src_outp = os.path.join(outdir, f"sync_per_source_{safe_name}.png")
        src_fig.savefig(src_outp, dpi=150, bbox_inches="tight")
        plt.close(src_fig)
        print(f"[scripts/run_spike_synchrony]   Saved: {src_outp}")

    # Save CSV summary
    csv_out = os.path.join(outdir, "spike_sync_summary.csv")
    all_stats.to_csv(csv_out, index=False)
    print(f"[scripts/run_spike_synchrony]   Saved: {csv_out}")
    print(f"[scripts/run_spike_synchrony] \nAnalysis complete! Output directory: {outdir}")

    return all_stats


def main() -> None:
    p = argparse.ArgumentParser(description="Run spike synchrony analysis on sim directories")
    p.add_argument("-d", "--dir", required=True, help="Path to simulation directory (contains synapses.csv)")
    args = p.parse_args()

    sim_dir = args.dir
    if not os.path.isdir(sim_dir):
        raise SystemExit(f"Sim directory not found: {sim_dir}")

    # run for single simulation directory
    run_for_dirs([sim_dir])


if __name__ == "__main__":
    main()
