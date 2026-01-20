import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from scipy import stats
import os
from matplotlib.collections import LineCollection
import colorsys
from matplotlib.patches import Patch
import pickle
from Modules.cell_model.plot_morphology import plot_clusters
from matplotlib import colors as mplcolors
import matplotlib.lines as mlines
import re
from matplotlib.colors import ListedColormap
from Modules.logger import Logger

def as_mpl_rgba(c) -> Tuple[float, float, float, float]:
    """
    Convert various color specs to a valid RGBA tuple in [0,1], with alpha=1 if missing.
    - Strings (e.g., '#aabbcc', 'tab:blue') -> mpl conversion
    - Tuples/lists length 3/4 with 0–1 or 0–255 (auto-detect) -> normalized, clipped
    - Anything invalid -> returns default black (0,0,0,1)
    """
    try:
        # Directly handle Matplotlib-recognized strings
        if isinstance(c, str):
            return mplcolors.to_rgba(c)

        arr = np.asarray(c, dtype=float).flatten()
        if arr.size == 0:
            return (0.0, 0.0, 0.0, 1.0)

        # If only 3 components, append alpha=1
        if arr.size == 3:
            arr = np.r_[arr, 1.0]
        elif arr.size > 4:
            arr = arr[:4]

        # If looks like 0–255, normalize to 0–1
        if np.nanmax(arr[:3]) > 1.0:
            arr[:3] = arr[:3] / 255.0

        # Replace NaNs/Infs and clip
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr[:3] = np.clip(arr[:3], 0.0, 1.0)
        arr[3] = float(np.clip(arr[3], 0.0, 1.0))
        return tuple(map(float, arr))
    except Exception:
        return (0.0, 0.0, 0.0, 1.0)


class SynapseAnalyzer:
    def __init__(self, sim_dir: str):
        """Initialize the analyzer with the simulation directory."""
        self.sim_dir = sim_dir
        self.synapses = pd.read_csv(os.path.join(sim_dir, "synapses.csv"))
        # Convert spike train strings to numpy arrays
        self.synapses["spike_train"] = self.synapses["spike_train"].apply(
            lambda s: np.fromstring(s.strip("[]"), sep=" ")
        )
        self.logger = Logger(sim_dir)
        
    def add_segment_data(self):
        segments = pd.read_csv(os.path.join(self.sim_dir, "segment_data.csv"))
        synapses_with_seg_info = self.synapses.merge(
        segments, 
        on='seg_id', 
        how='left',               # carry along all synapses even if a seg_id is missing
        suffixes=('','_seg')      # e.g. if both have a 'length' column
        )
        self.synapses = synapses_with_seg_info

    def plot_spike_raster(
        self,
        synapses: pd.DataFrame | None = None,
        time_window: tuple[float, float] | None = None,
        synapse_types: list[str] | None = None,
        functional_groups: list[int] | None = None,
        figsize: tuple[int, int] = (12, 8),
        save_path: str | None = None,
        title: str | None = "Spike Raster Plot",
        show: bool = False,
        *,
        color_by: str = "input_source",
        legend_loc: str = "upper right",
        legend_cols: int = 1,
        legend_matches_top: bool = True,
        shuffle_colors: bool = True,
        color_seed: int | None = 7,
        cmap_name: str = "tab20",
        draw_group_separators: bool = True,
        separator_kwargs: dict | None = None,
        marker_size: float = 5,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import numpy as np
        import pandas as pd

        df_all = synapses if synapses is not None else self.synapses
        if df_all is None or df_all.empty:
            raise ValueError("No synapses available to plot.")
        if color_by not in df_all.columns:
            raise KeyError(f"Column '{color_by}' not found in synapses DataFrame.")

        # ----- fast filtering (avoid Series ops when possible)
        mask = np.ones(len(df_all), dtype=bool)
        if synapse_types is not None:
            # compile regex once
            pat = "|".join(map(re.escape, synapse_types))
            mask &= df_all["name"].astype("string", copy=False).str.contains(pat, regex=True, na=False).to_numpy(dtype=bool)
        if functional_groups is not None:
            mask &= df_all["functional_group"].isin(functional_groups).to_numpy(dtype=bool)

        df = df_all.loc[mask, [color_by, "spike_train"]].copy()
        if df.empty:
            raise ValueError("No synapses left after filtering.")

        # ----- group ordering (same semantics as your version)
        def _input_source_key(s: str) -> tuple[int, str]:
            s = (s or "").lower()
            blocks = ["tuft", "nexus", "oblique", "trunk", "perisomatic", "distal_basal"]
            for rank, b in enumerate(blocks):
                if s.startswith(b):
                    return (rank, s)
            return (len(blocks), s)

        # ----- group ordering (same semantics as before)
        def _input_source_key(s: str) -> tuple[int, str]:
            s = (s or "").lower()
            blocks = ["tuft", "nexus", "oblique", "trunk", "perisomatic", "distal_basal"]
            for rank, b in enumerate(blocks):
                if s.startswith(b):
                    return (rank, s)
            return (len(blocks), s)

        if color_by == "input_source":
            df["_sort_key"] = (
                df[color_by]
                .fillna("UNKNOWN")
                .astype(str)
                .map(_input_source_key)
            )
            df.sort_values(by=["_sort_key", color_by], kind="stable", inplace=True)
            df.drop(columns="_sort_key", inplace=True)
        else:
            df.sort_values(by=color_by, kind="stable", inplace=True)

        # ----- stable ordered groups + color mapping
        groups = df[color_by].fillna("UNKNOWN").astype(str).to_numpy()
        ordered_groups = pd.unique(groups).tolist()

        cmap = plt.get_cmap(cmap_name)
        base_colors = [cmap(i % cmap.N) for i in range(len(ordered_groups))]
        if shuffle_colors and len(base_colors) > 1:
            rng = np.random.default_rng(color_seed)
            base_colors = [base_colors[i] for i in rng.permutation(len(base_colors))]
        # factorize for fast lookup
        grp_to_idx = {g: i for i, g in enumerate(ordered_groups)}
        group_codes = np.fromiter((grp_to_idx[g] for g in groups), count=len(groups), dtype=int)

        # ----- parse spike trains fast
        # Expect spike_train as ndarray/list/str; normalize to ndarray[int] without per-row plotting
        def _to_array(x):
            if isinstance(x, np.ndarray):
                return x.astype(int, copy=False)
            if isinstance(x, list):
                return np.asarray(x, dtype=int)
            if isinstance(x, (int, np.integer)):
                return np.array([int(x)], dtype=int)
            if isinstance(x, float):
                return np.empty(0, dtype=int) if np.isnan(x) else np.array([int(x)], dtype=int)
            if isinstance(x, str):
                s = x.strip().strip("[]")
                return np.fromstring(s, sep=" ", dtype=int) if s else np.empty(0, dtype=int)
            return np.empty(0, dtype=int)

        spike_lists = [ _to_array(x) for x in df["spike_train"].to_numpy() ]

        # optional time-window crop (vectorized per row)
        if time_window is not None:
            t0, t1 = time_window
            spike_lists = [arr[(arr >= t0) & (arr <= t1)] if arr.size else arr for arr in spike_lists]

        # ----- build one big scatter payload (no iterrows)
        # y indices: 0..n-1 (later inverted visually if requested)
        nrows = len(df)
        # counts per row → repeat row_idx for each spike in that row
        counts = np.fromiter((len(a) for a in spike_lists), count=nrows, dtype=int)
        if counts.sum() == 0:
            # nothing to plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title(title or "Spike Raster Plot")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel(f"Synapse (sorted by {color_by})")
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)
            return

        y_idx = np.repeat(np.arange(nrows, dtype=int), counts)
        x_all = np.concatenate([a for a in spike_lists if a.size])
        # per-point colors via row's group code mapped to base_colors
        # Map each row to an integer group code (already computed as `group_codes`)
        # Expand to per-point codes using the expanded row indices y_idx
        codes_all = group_codes[y_idx]  # shape: (total_points,)

        fig, ax = plt.subplots(figsize=figsize)

        # single fast scatter
        # Use integer codes with a ListedColormap: no huge object arrays, no repeat()
        cmap_obj = ListedColormap(base_colors)
        sc = ax.scatter(
            x_all, y_idx,
            s=marker_size,
            c=codes_all,                 # integers per point
            cmap=cmap_obj,
            vmin=0, vmax=len(base_colors)-1,
            marker='.',
            linewidths=0,
        )
        ax.set_ylim(-1, nrows)
        if time_window is not None:
            ax.set_xlim(*time_window)
        if legend_matches_top:
            ax.invert_yaxis()

        # ----- separators (compute group change indices with vector ops)
        if draw_group_separators and len(ordered_groups) > 1:
            g = groups  # already numpy array
            change = np.flatnonzero(g[1:] != g[:-1]) + 1  # row indices where group starts
            skw = {"linestyle": (0, (4, 4)), "linewidth": 0.6, "alpha": 0.4, "color": "k"}
            if separator_kwargs:
                skw.update(separator_kwargs)
            for end in change:
                ax.axhline(end - 0.5, **skw)

        # ----- legend (matches visual order)
        handles = [
            mlines.Line2D([], [], color=base_colors[i], marker='.', linestyle='None', markersize=6, label=g)
            for i, g in enumerate(ordered_groups)
        ]
        if not legend_matches_top:
            handles = list(reversed(handles))
        if handles:
            ax.legend(handles=handles, loc=legend_loc, ncol=legend_cols, frameon=False, title=color_by)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Synapse (sorted by {color_by})")
        ax.set_title(title or "Spike Raster Plot")

        # helpful for vector outputs with many points
        for c in ax.collections:
            c.set_rasterized(True)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_spike_raster_fgpc_legend(
        synapses: pd.DataFrame,
        time_window: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        yticklabel_stride: int = 30,
        show_y_labels: bool = True,
        legend_loc: str = 'upper right',
        show: bool = False
    ):
        """
        Spike raster with FG-based base colors and PC-based lightness shading.
        Accepts a DataFrame with columns: 'functional_group', 'presynaptic_cell', 'spike_train'.
        """

        def get_shaded_color(base_rgb, pc_idx, n_pcs):
            """
            Return a lighter/darker shade (in 0–1 RGB) for pc_idx out of n_pcs based on base_rgb.
            We vary lightness in HLS between 0.40 and 0.80.
            """
            base = np.asarray(base_rgb, dtype=float)
            if base.max() > 1.0:  # tolerate 0–255
                base = base / 255.0
            base = np.clip(base[:3], 0.0, 1.0)
            h, l, s = colorsys.rgb_to_hls(*base)
            l_new = 0.4 + 0.4 * (pc_idx / max(n_pcs - 1, 1))
            rgb = colorsys.hls_to_rgb(h, l_new, s)
            return tuple(np.clip(rgb, 0.0, 1.0))

        # Sort and reset index for stable plotting order
        synapses_sorted = synapses.sort_values(['functional_group', 'presynaptic_cell']).reset_index(drop=True)

        # Unique FGs (sorted for stable colors)
        unique_fgs = sorted(synapses_sorted['functional_group'].dropna().unique().tolist())
        cmap = plt.cm.get_cmap('tab20', max(len(unique_fgs), 1))
        # Base colors per FG (tab20 returns 0–1 RGBA; take RGB)
        fg_base_colors = {fg: cmap(i % cmap.N)[:3] for i, fg in enumerate(unique_fgs)}

        # PCs per FG for consistent shading
        fg_to_pcs = {
            fg: sorted(synapses_sorted.loc[synapses_sorted['functional_group'] == fg, 'presynaptic_cell'].dropna().unique().tolist())
            for fg in unique_fgs
        }

        # Precompute FG/PC → color
        fg_pc_color: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
        for fg, pcs in fg_to_pcs.items():
            n_pcs = len(pcs) if len(pcs) > 0 else 1
            base = fg_base_colors.get(fg, (0.0, 0.0, 0.0))
            for i, pc in enumerate(pcs):
                c = get_shaded_color(base, i, n_pcs)
                fg_pc_color[(fg, pc)] = as_mpl_rgba(c)

        # Build segments for a vertical-tick raster (LineCollection)
        segments = []
        colors = []
        legend_labels: Dict[str, Tuple[float, float, float, float]] = {}

        n_rows = len(synapses_sorted)
        if n_rows == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Spike Raster Plot (FG color, PC shade, legend) — no data")
            ax.set_xticks([])
            ax.set_yticks([])
            if show:
                plt.show()
            else:
                plt.close()
            return

        # Precompute y-ticks (either a clean fixed number or stride)
        if show_y_labels:
            # Use a clean number of ticks for readability
            n_ticks = 15
            yticks = np.linspace(1, n_rows, n_ticks, dtype=int)
            yticklabels = [
                f"FG{int(synapses_sorted.iloc[i-1]['functional_group'])}_PC{int(synapses_sorted.iloc[i-1]['presynaptic_cell'])}"
                for i in yticks
            ]
        else:
            yticks, yticklabels = [], []

        for idx, row in synapses_sorted.iterrows():
            spikes = row.get('spike_train', None)

            # Robust spike parsing
            if isinstance(spikes, str):
                spikes = np.fromstring(spikes.replace('[', '').replace(']', ''), sep=' ')
            elif isinstance(spikes, (float, int)) or spikes is None or (isinstance(spikes, np.ndarray) and spikes.ndim == 0):
                continue
            else:
                spikes = np.array(spikes, dtype=float).flatten()

            if spikes.size == 0:
                continue

            if time_window is not None:
                t0, t1 = time_window
                spikes = spikes[(spikes >= t0) & (spikes <= t1)]
                if spikes.size == 0:
                    continue

            # Each spike becomes a short vertical segment around row index
            row_y = idx + 1.0
            segs = [((spk, row_y - 0.4), (spk, row_y + 0.4)) for spk in spikes]
            segments.extend(segs)

            fg = row.get('functional_group', None)
            pc = row.get('presynaptic_cell', None)
            color = fg_pc_color.get((fg, pc), (0.0, 0.0, 0.0, 1.0))
            colors.extend([color] * len(segs))

            key = f"FG{int(fg)}_PC{int(pc)}" if pd.notna(fg) and pd.notna(pc) else "FG?_PC?"
            if key not in legend_labels:
                legend_labels[key] = as_mpl_rgba(color)

        fig, ax = plt.subplots(figsize=figsize)
        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=0.7)
            ax.add_collection(lc)

        # X limits
        if time_window is not None:
            ax.set_xlim(time_window)
        else:
            try:
                all_spikes = []
                for _, r in synapses_sorted.iterrows():
                    st = r['spike_train']
                    if isinstance(st, str):
                        st = np.fromstring(st.replace('[', '').replace(']', ''), sep=' ')
                    elif isinstance(st, (float, int)) or st is None or (isinstance(st, np.ndarray) and st.ndim == 0):
                        continue
                    else:
                        st = np.array(st, dtype=float).flatten()
                    if st.size:
                        all_spikes.append(st)
                if len(all_spikes) > 0:
                    all_sp = np.hstack(all_spikes)
                    ax.set_xlim(float(np.nanmin(all_sp)), float(np.nanmax(all_sp)))
                else:
                    ax.set_xlim(0, 1)
            except Exception:
                ax.set_xlim(0, 1)

        ax.set_ylim(0.5, n_rows + 0.5)
        ax.set_ylabel("Synapse (FG/PC grouped)")
        ax.set_xlabel("Time (ms or sample)")
        ax.set_title("Spike Raster Plot (FG color, PC shade, legend)")

        if show_y_labels:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=6)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        # Legend — sanitize colors to valid RGBA
        patches = []
        for label, color in legend_labels.items():
            rgba = as_mpl_rgba(color)
            patches.append(Patch(facecolor=rgba, edgecolor='black', label=label))

        # Optionally limit legend length for readability
        if len(patches) > 25:
            patches = patches[:25]

        if len(patches) > 0:
            ax.legend(handles=patches, loc=legend_loc, title='FG_PC', fontsize=7, title_fontsize=8, frameon=True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()
        
    def analyze_cluster_statistics(self, 
                                 functional_group_id: int = None,
                                 synapse_type: str = None,
                                 show: bool = False) -> Dict:
        """
        Calculate statistics for synapse clusters.
        
        Args:
            functional_group_id: Optional specific functional group to analyze
            synapse_type: Optional synapse type to filter by
            
        Returns:
            Dictionary containing cluster statistics
        """
        # Filter synapses
        mask = pd.Series(True, index=self.synapses.index)
        if functional_group_id is not None:
            mask &= self.synapses['functional_group'] == functional_group_id
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type)
            
        filtered_synapses = self.synapses[mask]
        
        # Calculate statistics
        stats_dict = {
            'total_synapses': len(filtered_synapses),
            'mean_firing_rate': filtered_synapses['pc_mean_firing_rate'].mean(),
            'std_firing_rate': filtered_synapses['pc_mean_firing_rate'].std(),
            'mean_weight': filtered_synapses['initW'].mean(),
            'std_weight': filtered_synapses['initW'].std(),
        }
        
        # Calculate spike train statistics
        print(f"filtered_synapses['spike_train'].values: {filtered_synapses['spike_train'].values}")
        all_spikes = np.concatenate(filtered_synapses['spike_train'].values)
        if len(all_spikes) > 0:
            stats_dict.update({
                'mean_isi': np.mean(np.diff(np.sort(all_spikes))),
                'std_isi': np.std(np.diff(np.sort(all_spikes))),
                'total_spikes': len(all_spikes),
            })
            
        return stats_dict
        
    def plot_firing_rate_distribution(self,
                                    synapse_type: str = None,
                                    input_source: str = None,
                                    functional_group: int = None,
                                    figsize: Tuple[int, int] = (10, 6),
                                    save_path: Optional[str] = None,
                                    show: bool = False,
                                    dpi: int = 150) -> None:
        """
        Plot the distribution of firing rates across synapses.
        
        Args:
            synapse_type: Optional synapse type to filter by
            input_source: Optional input source to filter by
            functional_group: Optional functional group to filter by
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            dpi: DPI for saved figure (default 150, lower = faster)
        """
        # Vectorized filtering - much faster than multiple boolean operations
        mask = np.ones(len(self.synapses), dtype=bool)
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type, na=False).to_numpy()
        if input_source:
            mask &= (self.synapses['input_source'] == input_source).to_numpy()
        if functional_group is not None:
            mask &= (self.synapses['functional_group'] == functional_group).to_numpy()
            
        # Direct array access - faster than DataFrame slicing
        firing_rates = self.synapses.loc[mask, 'pc_mean_firing_rate'].to_numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use plt.hist instead of sns.histplot for ~3x speedup
        ax.hist(firing_rates, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Mean Firing Rate (Hz)')
        ax.set_ylabel('Count')
        
        # Create title based on filters
        title_parts = ['Distribution of Synapse Firing Rates']
        if input_source:
            title_parts.append(f'({input_source})')
        elif synapse_type:
            title_parts.append(f'({synapse_type})')
        ax.set_title(' '.join(title_parts))
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        
    def plot_weight_distribution(self,
                               synapse_type: str = None,
                               functional_group: int = None,
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[str] = None,
                               show: bool = False,
                               dpi: int = 150) -> None:
        """
        Plot the distribution of synapse weights.
        
        Args:
            synapse_type: Optional synapse type to filter by
            functional_group: Optional functional group to filter by
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
            dpi: DPI for saved figure (default 150, lower = faster)
        """
        # Vectorized filtering
        mask = np.ones(len(self.synapses), dtype=bool)
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type, na=False).to_numpy()
        if functional_group is not None:
            mask &= (self.synapses['functional_group'] == functional_group).to_numpy()
            
        # Direct array access
        weights = self.synapses.loc[mask, 'initW'].to_numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use plt.hist instead of sns.histplot for ~3x speedup
        ax.hist(weights, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Initial Weight')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Synapse Weights')
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_cluster_spatial_distribution(self,
                                        synapse_type: str = None,
                                        functional_group: int = None,
                                        figsize: Tuple[int, int] = (10, 10),
                                        save_path: Optional[str] = None,
                                        show: bool = False) -> None:
        """
        Create a 3D scatter plot of synapse locations, colored by functional group.
        
        Args:
            synapse_type: Optional synapse type to filter by
            functional_group: Optional functional group to filter by
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        if hasattr(self.synapses, 'pc_0'):
            x_coord_name = 'pc_0'
            y_coord_name = 'pc_1'
            z_coord_name = 'pc_2'
        elif hasattr(self.synapses, 'Coord X'):
            x_coord_name = 'Coord X'
            y_coord_name = 'Coord Y'
            z_coord_name = 'Coord Z'
        else:
            raise ValueError(f"No coordinate columns found in synapses.csv: {self.synapses.columns}")
        # Filter synapses
        mask = pd.Series(True, index=self.synapses.index)
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type)
        if functional_group is not None:
            mask &= self.synapses['functional_group'] == functional_group
            
        filtered_synapses = self.synapses[mask]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each functional group with a different color
        for fg in filtered_synapses['functional_group'].unique():
            fg_synapses = filtered_synapses[filtered_synapses['functional_group'] == fg]
            ax.scatter(fg_synapses[x_coord_name], 
                      fg_synapses[y_coord_name], 
                      fg_synapses[z_coord_name],
                      label=f'FG {fg}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Spatial Distribution of Synapses by Functional Group')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def calculate_correlation_matrix(self,
                                   synapse_type: str = None,
                                   functional_group: int = None,
                                   time_window: Tuple[float, float] = None) -> pd.DataFrame:
        """
        Calculate the correlation matrix between spike trains of synapses.
        
        Args:
            synapse_type: Optional synapse type to filter by
            functional_group: Optional functional group to filter by
            time_window: Optional time window to analyze
            
        Returns:
            DataFrame containing the correlation matrix
        """
        # Filter synapses
        mask = pd.Series(True, index=self.synapses.index)
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type)
        if functional_group is not None:
            mask &= self.synapses['functional_group'] == functional_group
            
        filtered_synapses = self.synapses[mask]
        
        # Convert spike trains to binary time series
        if time_window:
            t_start, t_end = time_window
            time_bins = np.arange(t_start, t_end, 1)  # 1ms bins
        else:
            # Find the maximum time across all spike trains
            max_time = max([max(spikes) for spikes in filtered_synapses['spike_train']])
            time_bins = np.arange(0, max_time + 1, 1)
            
        # Create binary spike trains
        binary_trains = np.zeros((len(filtered_synapses), len(time_bins)))
        for i, spikes in enumerate(filtered_synapses['spike_train']):
            if time_window:
                spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
            spike_bins = np.digitize(spikes, time_bins) - 1
            binary_trains[i, spike_bins] = 1
            
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(binary_trains)
        
        return pd.DataFrame(corr_matrix, 
                          index=filtered_synapses.index,
                          columns=filtered_synapses.index)
        
    def plot_correlation_matrix(self,
                              synapse_type: str = None,
                              functional_group: int = None,
                              time_window: Tuple[float, float] = None,
                              figsize: Tuple[int, int] = (12, 10),
                              save_path: Optional[str] = None,
                              show: bool = False) -> None:
        """
        Plot the correlation matrix between spike trains.
        
        Args:
            synapse_type: Optional synapse type to filter by
            functional_group: Optional functional group to filter by
            time_window: Optional time window to analyze
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        corr_matrix = self.calculate_correlation_matrix(synapse_type, functional_group, time_window)
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Spike Train Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        
    def plot_all_synapse_clusters(
        self,
        synapse_coord_cols=('pc_0', 'pc_1', 'pc_2'),
        plot_both_together=True,
        plot_each_type_separately=True,
        show=False,
        dpi=150,  # Lower DPI for faster rendering (was 300)
        skip_inh=False  # Option to skip inhibitory plots if not needed
    ):
        if not os.path.exists(os.path.join(self.sim_dir, 'clusters')):
            os.mkdir(os.path.join(self.sim_dir, 'clusters'))
        # Load parameters and segment data ONCE
        with open(os.path.join(self.sim_dir, "parameters.pickle"), 'rb') as file:
            parameters = pickle.load(file)
        seg_data = pd.read_csv(os.path.join(self.sim_dir, "segment_data.csv"))

        # Join synapses with segment info ONCE - avoid repeated merges
        synapses_with_seg_info = self.synapses.merge(
            seg_data, on='seg_id', how='left', suffixes=('', '_seg')
        )

        # Helper to extract coords by mask - use numpy for faster indexing
        def coords_for_mask(mask):
            cols = list(synapse_coord_cols)
            return synapses_with_seg_info.loc[mask, cols].to_numpy()  # to_numpy() is faster than .values

        # Masks to separate exc/inh dots (name format "exc_<input_source>_..."/"inh_...")
        exc_mask = synapses_with_seg_info['name'].str.startswith('exc_', na=False)
        inh_mask = synapses_with_seg_info['name'].str.startswith('inh_', na=False)
        exc_coords = coords_for_mask(exc_mask)
        inh_coords = coords_for_mask(inh_mask) if not skip_inh else None

        exc_cfg = getattr(parameters, 'exc_clustering', None)
        inh_cfg = getattr(parameters, 'inh_clustering', None) if not skip_inh else None

        # Plot both together (skip inh if missing)
        if plot_both_together and exc_cfg is not None:
            fig = plt.figure(figsize=(20, 10))

            ax1 = fig.add_subplot(121, projection='3d')
            plot_clusters(
                seg_data=seg_data,
                clustering_config=exc_cfg,
                synapse_coords=exc_coords,
                ax=ax1,
                elevation=20, azimuth=-100,
                title='Excitatory Clusters',
                logger = self.logger,
            )

            if inh_cfg is not None:
                ax2 = fig.add_subplot(122, projection='3d')
                plot_clusters(
                    seg_data=seg_data,
                    clustering_config=inh_cfg,
                    synapse_coords=inh_coords,
                    ax=ax2,
                    elevation=20, azimuth=-100,
                    title='Inhibitory Clusters',
                    logger = self.logger,
                )

            plt.tight_layout()
            fig.savefig(
                os.path.join(self.sim_dir, 'clusters', f'clusters_both.png'),
                dpi=dpi, bbox_inches='tight'
                )
            if show:
                plt.show()
            else:
                plt.close()

        # Plot each excitatory input_source separately - use consistent DPI
        if plot_each_type_separately and exc_cfg is not None:
            for input_source in exc_cfg.keys():
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                plot_clusters(
                    seg_data=seg_data,
                    clustering_config={input_source: exc_cfg[input_source]},
                    synapse_coords=exc_coords,
                    ax=ax,
                    elevation=20, azimuth=-100,
                    title=f'Excitatory Clusters - {input_source}',
                    logger=self.logger,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(self.sim_dir, 'clusters', f'clusters_exc_{input_source}.png'), dpi=dpi)
                if show:
                    plt.show()
                else:
                    plt.close()

        # Plot each inhibitory input_source separately (if provided)
        if plot_each_type_separately and inh_cfg is not None:
            for input_source in inh_cfg.keys():
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                plot_clusters(
                    seg_data=seg_data,
                    clustering_config={input_source: inh_cfg[input_source]},
                    synapse_coords=inh_coords,
                    ax=ax,
                    elevation=20, azimuth=-100,
                    title=f'Inhibitory Clusters - {input_source}',
                    logger=self.logger,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(self.sim_dir, 'clusters', f'clusters_inh_{input_source}.png'), dpi=dpi)
                if show:
                    plt.show()
                else:
                    plt.close()
