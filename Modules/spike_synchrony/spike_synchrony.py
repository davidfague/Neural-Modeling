"""Module for computing and plotting spike-train synchrony using PySpike.

Provides functions to load synapse CSVs, prepare PySpike SpikeTrain objects,
compute SPIKE-synchronization matrices per simulation, summarize stats, and
plot results.

Designed to be imported from notebooks.
"""
from __future__ import annotations

import os
import pickle
from itertools import combinations
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd
import pyspike as spk
import matplotlib.pyplot as plt
import seaborn as sns


def to_spike_array(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        return np.array([], dtype=float)
    return np.fromstring(s, sep=" ", dtype=float)


def load_synapses(sim_dirs: Iterable[str]) -> pd.DataFrame:
    syn_list = []
    for sim_dir in sim_dirs:
        csv_path = os.path.join(sim_dir, "synapses.csv")
        if not os.path.isfile(csv_path):
            print(f"WARNING: synapses.csv not found in {sim_dir}, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["sim_name"] = os.path.basename(sim_dir.rstrip("/"))
        syn_list.append(df)
    if not syn_list:
        raise RuntimeError("No synapses.csv files were loaded!")
    syn_df = pd.concat(syn_list, ignore_index=True)
    return syn_df


def prepare_spike_trains(df: pd.DataFrame, t_start: float = 0.0, t_end: float | None = None, min_spikes: int = 1, parameters=None) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    df = df.copy()
    df["spike_times"] = df["spike_train"].apply(to_spike_array)
    if t_end is None:
        if parameters is not None and hasattr(parameters, 'h_tstop'):
            t_end = float(parameters.h_tstop)
        else:
            max_t = max((st.max() if st.size > 0 else 0.0) for st in df["spike_times"])
            t_end = float(max_t + 1.0)
    df = df[df["spike_times"].apply(len) >= min_spikes].copy()
    df["spike_obj"] = df["spike_times"].apply(lambda st: spk.SpikeTrain(st, edges=(t_start, t_end)))
    return df, (t_start, t_end)


def assign_positions_within_sim(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_in_sim"] = -1
    for sim, g in df.groupby("sim_name"):
        idx = g.index.to_numpy()
        df.loc[idx, "pos_in_sim"] = np.arange(len(idx), dtype=int)
    return df


def build_sync_matrices_per_sim(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    sync_mats: Dict[str, np.ndarray] = {}
    for sim, g in df.groupby("sim_name"):
        g_sorted = g.sort_values("pos_in_sim")
        trains = g_sorted["spike_obj"].tolist()
        mat = spk.spike_sync_matrix(trains)
        sync_mats[sim] = mat
    return sync_mats


def stats_from_submatrix(mat: np.ndarray, idx: np.ndarray):
    idx = np.asarray(idx, dtype=int)
    if idx.size < 2:
        return np.nan, np.nan, 0
    sub = mat[np.ix_(idx, idx)]
    iu = np.triu_indices(idx.size, k=1)
    vals = sub[iu]
    return float(vals.mean()), float(vals.std(ddof=0)), vals.size


def between_stats_from_indices(mat: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray):
    idx_a = np.asarray(idx_a, dtype=int)
    idx_b = np.asarray(idx_b, dtype=int)
    if idx_a.size == 0 or idx_b.size == 0:
        return np.nan, np.nan, 0
    block = mat[np.ix_(idx_a, idx_b)]
    vals = block.ravel()
    return float(vals.mean()), float(vals.std(ddof=0)), vals.size


def compute_all_stats(syn_df: pd.DataFrame, sync_mats: Dict[str, np.ndarray], drop_fg_negative: bool = True) -> pd.DataFrame:
    df = syn_df.copy()
    if drop_fg_negative and "functional_group" in df.columns:
        df_fg = df[df["functional_group"] >= 0].copy()
    else:
        df_fg = df.copy()

    rows = []
    # within input_source per sim
    for (sim, src), sub in df.groupby(["sim_name", "input_source"]):
        mat = sync_mats[sim]
        idx = sub["pos_in_sim"].to_numpy()
        m, s, n_pairs = stats_from_submatrix(mat, idx)
        rows.append({
            "level": "within_input_source_per_sim",
            "sim_name": sim,
            "input_source": src,
            "functional_group": None,
            "presynaptic_cell": None,
            "n_trains": len(sub),
            "n_pairs": n_pairs,
            "mean_sync": m,
            "std_sync": s,
        })

    # within input_source x fg per sim
    if "functional_group" in df_fg.columns:
        for (sim, src, fg), sub in df_fg.groupby(["sim_name", "input_source", "functional_group"]):
            mat = sync_mats[sim]
            idx = sub["pos_in_sim"].to_numpy()
            m, s, n_pairs = stats_from_submatrix(mat, idx)
            rows.append({
                "level": "within_input_source_fg_per_sim",
                "sim_name": sim,
                "input_source": src,
                "functional_group": fg,
                "presynaptic_cell": None,
                "n_trains": len(sub),
                "n_pairs": n_pairs,
                "mean_sync": m,
                "std_sync": s,
            })

        # within input_source x fg x presynaptic_cell
        if "presynaptic_cell" in df_fg.columns:
            for (sim, src, fg, pc), sub in df_fg.groupby(["sim_name", "input_source", "functional_group", "presynaptic_cell"]):
                mat = sync_mats[sim]
                idx = sub["pos_in_sim"].to_numpy()
                m, s, n_pairs = stats_from_submatrix(mat, idx)
                rows.append({
                    "level": "within_input_source_fg_pc_per_sim",
                    "sim_name": sim,
                    "input_source": src,
                    "functional_group": fg,
                    "presynaptic_cell": pc,
                    "n_trains": len(sub),
                    "n_pairs": n_pairs,
                    "mean_sync": m,
                    "std_sync": s,
                })

    # between input_sources per sim
    for sim, sim_df in df.groupby("sim_name"):
        groups = {src: sub["pos_in_sim"].to_numpy() for src, sub in sim_df.groupby("input_source")}
        mat = sync_mats[sim]
        for (src_a, idx_a), (src_b, idx_b) in combinations(groups.items(), 2):
            m, s, n_pairs = between_stats_from_indices(mat, idx_a, idx_b)
            rows.append({
                "level": "between_input_sources_per_sim",
                "sim_name": sim,
                "input_source_a": src_a,
                "input_source_b": src_b,
                "n_pairs": n_pairs,
                "mean_sync": m,
                "std_sync": s,
            })

    # between functional_groups within each input_source & sim
    if "functional_group" in df_fg.columns:
        for (sim, src), src_df in df_fg.groupby(["sim_name", "input_source"]):
            groups = {fg: sub["pos_in_sim"].to_numpy() for fg, sub in src_df.groupby("functional_group")}
            mat = sync_mats[sim]
            for (fg_a, idx_a), (fg_b, idx_b) in combinations(groups.items(), 2):
                m, s, n_pairs = between_stats_from_indices(mat, idx_a, idx_b)
                rows.append({
                    "level": "between_fgs_within_source_per_sim",
                    "sim_name": sim,
                    "input_source": src,
                    "functional_group_a": fg_a,
                    "functional_group_b": fg_b,
                    "n_pairs": n_pairs,
                    "mean_sync": m,
                    "std_sync": s,
                })

    # between presynaptic_cells within fg & input_source & sim
    if "functional_group" in df_fg.columns and "presynaptic_cell" in df_fg.columns:
        for (sim, src, fg), fg_df in df_fg.groupby(["sim_name", "input_source", "functional_group"]):
            pc_groups = {pc: sub["pos_in_sim"].to_numpy() for pc, sub in fg_df.groupby("presynaptic_cell")}
            mat = sync_mats[sim]
            for (pc_a, idx_a), (pc_b, idx_b) in combinations(pc_groups.items(), 2):
                m, s, n_pairs = between_stats_from_indices(mat, idx_a, idx_b)
                rows.append({
                    "level": "between_pcs_within_fg_per_sim",
                    "sim_name": sim,
                    "input_source": src,
                    "functional_group": fg,
                    "presynaptic_cell_a": pc_a,
                    "presynaptic_cell_b": pc_b,
                    "n_pairs": n_pairs,
                    "mean_sync": m,
                    "std_sync": s,
                })

    all_stats = pd.DataFrame(rows)
    return all_stats


def plot_sync_matrix(sync_mats: Dict[str, np.ndarray], sim_name: str, ax=None, cmap: str = "viridis") -> None:
    if sim_name not in sync_mats:
        raise KeyError(sim_name)
    mat = sync_mats[sim_name]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(mat, ax=ax, cmap=cmap)
    ax.set_title(f"SPIKE-synchronization matrix: {sim_name}")
    ax.set_xlabel("train index")
    ax.set_ylabel("train index")


def plot_mean_sync_distribution(all_stats: pd.DataFrame, ax=None, level_filter: str | None = None) -> None:
    df = all_stats.copy()
    if level_filter is not None:
        df = df[df["level"] == level_filter]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["mean_sync"].dropna(), bins=40, kde=True, ax=ax)
    ax.set_xlabel("Mean SPIKE-synchronization")
    ax.set_title("Distribution of mean synchrony")


def plot_comprehensive_analysis(all_stats: pd.DataFrame, figsize=(18, 10)) -> plt.Figure:
    """Create comprehensive 6-panel figure showing all grouping levels.
    
    Panels:
    - Row 1 (Within-group): input_source, input_source x fg, input_source x fg x pc
    - Row 2 (Between-group): input_sources, fgs within source, pcs within fg
    
    Uses consistent [0, 1] y-axis scale across all panels.
    """
    levels = [
        ("within_input_source_per_sim", "Within input_source"),
        ("within_input_source_fg_per_sim", "Within input_source × FG"),
        ("within_input_source_fg_pc_per_sim", "Within input_source × FG × PC"),
        ("between_input_sources_per_sim", "Between input_sources"),
        ("between_fgs_within_source_per_sim", "Between FGs (same source)"),
        ("between_pcs_within_fg_per_sim", "Between PCs (same FG)"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for ax, (level_key, title) in zip(axes, levels):
        df_level = all_stats[all_stats["level"] == level_key].copy()
        if len(df_level) > 0:
            try:
                sns.violinplot(data=df_level, y="mean_sync", ax=ax, inner="box", color="skyblue")
                ax.set_ylabel("Mean SPIKE-sync")
                ax.set_title(title)
                ax.set_xlabel("")
                ax.set_ylim([0, 1])  # Consistent scale: SPIKE-sync is in [0, 1]
                # Add count
                ax.text(0.02, 0.98, f"n={len(df_level)}", transform=ax.transAxes, 
                       va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            except Exception as e:
                ax.text(0.5, 0.5, f"No data\n{level_key}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
        else:
            ax.text(0.5, 0.5, f"No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    
    fig.tight_layout()
    return fig


def plot_per_input_source_analysis(all_stats: pd.DataFrame, figsize=(12, 8)) -> Dict[str, plt.Figure]:
    """Create per-input_source figures showing within/between groupings for each source.
    
    For each input_source, creates a 2x2 figure:
    - Top-left: Within input_source (overall)
    - Top-right: Within FGs (for this source)
    - Bottom-left: Within FG×PC (for this source) 
    - Bottom-right: Between FGs (for this source)
    
    Returns dict mapping input_source name to figure.
    """
    figures = {}
    
    # Get all unique input sources from the data
    within_src_df = all_stats[all_stats["level"] == "within_input_source_per_sim"].copy()
    if "input_source" not in within_src_df.columns or len(within_src_df) == 0:
        return figures
    
    input_sources = within_src_df["input_source"].unique()
    
    for src in input_sources:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Spike Synchrony Analysis: {src}", fontsize=14, fontweight="bold")
        
        # Panel 1: Within this input_source (overall)
        ax = axes[0, 0]
        df_within = all_stats[
            (all_stats["level"] == "within_input_source_per_sim") &
            (all_stats["input_source"] == src)
        ].copy()
        if len(df_within) > 0:
            sns.violinplot(data=df_within, y="mean_sync", ax=ax, inner="box", color="lightcoral")
            ax.set_title(f"Within {src}")
            ax.set_ylabel("Mean SPIKE-sync")
            ax.set_ylim([0, 1])
            ax.set_xlabel("")
            ax.text(0.02, 0.98, f"n={len(df_within)}", transform=ax.transAxes,
                   va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"Within {src}")
        
        # Panel 2: Within FGs for this input_source
        ax = axes[0, 1]
        df_within_fg = all_stats[
            (all_stats["level"] == "within_input_source_fg_per_sim") &
            (all_stats["input_source"] == src)
        ].copy()
        if len(df_within_fg) > 0:
            sns.violinplot(data=df_within_fg, y="mean_sync", ax=ax, inner="box", color="lightgreen")
            ax.set_title(f"Within FGs ({src})")
            ax.set_ylabel("Mean SPIKE-sync")
            ax.set_ylim([0, 1])
            ax.set_xlabel("")
            ax.text(0.02, 0.98, f"n={len(df_within_fg)}", transform=ax.transAxes,
                   va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "Only 1 FG", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"Within FGs ({src})")
        
        # Panel 3: Within FG×PC for this input_source
        ax = axes[1, 0]
        df_within_pc = all_stats[
            (all_stats["level"] == "within_input_source_fg_pc_per_sim") &
            (all_stats["input_source"] == src)
        ].copy()
        if len(df_within_pc) > 0:
            sns.violinplot(data=df_within_pc, y="mean_sync", ax=ax, inner="box", color="lightyellow")
            ax.set_title(f"Within FG×PC ({src})")
            ax.set_ylabel("Mean SPIKE-sync")
            ax.set_ylim([0, 1])
            ax.set_xlabel("")
            ax.text(0.02, 0.98, f"n={len(df_within_pc)}", transform=ax.transAxes,
                   va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "No FG×PC data", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"Within FG×PC ({src})")
        
        # Panel 4: Between FGs for this input_source
        ax = axes[1, 1]
        df_between_fg = all_stats[
            (all_stats["level"] == "between_fgs_within_source_per_sim") &
            (all_stats["input_source"] == src)
        ].copy()
        if len(df_between_fg) > 0:
            sns.violinplot(data=df_between_fg, y="mean_sync", ax=ax, inner="box", color="lightblue")
            ax.set_title(f"Between FGs ({src})")
            ax.set_ylabel("Mean SPIKE-sync")
            ax.set_ylim([0, 1])
            ax.set_xlabel("")
            ax.text(0.02, 0.98, f"n={len(df_between_fg)}", transform=ax.transAxes,
                   va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "Only 1 FG", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(f"Between FGs ({src})")
        
        fig.tight_layout()
        figures[src] = fig
    
    return figures


__all__ = [
    "load_synapses",
    "prepare_spike_trains",
    "assign_positions_within_sim",
    "build_sync_matrices_per_sim",
    "compute_all_stats",
    "plot_sync_matrix",
    "plot_mean_sync_distribution",
    "plot_comprehensive_analysis",
    "plot_per_input_source_analysis",
]
