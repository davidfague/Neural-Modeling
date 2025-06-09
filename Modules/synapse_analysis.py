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
from Modules.plot_morphology import plot_clusters

class SynapseAnalyzer:
    def __init__(self, sim_dir: str):
        """Initialize the analyzer with the simulation directory."""
        self.sim_dir = sim_dir
        self.synapses = pd.read_csv(os.path.join(sim_dir, "synapses.csv"))
        # Convert spike train strings to numpy arrays
        self.synapses["spike_train"] = self.synapses["spike_train"].apply(
            lambda s: np.fromstring(s.strip("[]"), sep=" ")
        )
        
    def add_segment_data(self):
        segments = pd.read_csv(os.path.join(self.sim_dir, "segment_data.csv"))
        synapses_with_seg_info = self.synapses.merge(
        segments, 
        on='seg_id', 
        how='left',               # carry along all synapses even if a seg_id is missing
        suffixes=('','_seg')      # e.g. if both have a 'length' column
        )
        self.synapses = synapses_with_seg_info
        
    def plot_spike_raster(self, 
                          synapses: Optional[pd.DataFrame] = None,
                          time_window: Optional[Tuple[float, float]] = None,
                          synapse_types: Optional[List[str]] = None,
                          functional_groups: Optional[List[int]] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None):
        """
        Generate a spike raster plot for the synapses (optionally user-provided).
        """
        # Use user-provided synapses or default to self.synapses
        synapses_to_plot = synapses if synapses is not None else self.synapses
        
        mask = pd.Series(True, index=synapses_to_plot.index)
        if synapse_types is not None:
            mask &= synapses_to_plot['name'].str.contains('|'.join(synapse_types))
        if functional_groups is not None:
            mask &= synapses_to_plot['functional_group'].isin(functional_groups)
        filtered_synapses = synapses_to_plot[mask]

        plt.figure(figsize=figsize)
        for plot_idx, (idx, row) in enumerate(filtered_synapses.iterrows()):
            spikes = row['spike_train']
            if time_window:
                spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
            plt.plot(spikes, [plot_idx] * len(spikes), 'k.', markersize=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Synapse')
        plt.title('Spike Raster Plot')
        plt.ylim(-1, len(filtered_synapses))
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_shaded_color(base_rgb, pc_idx, n_pcs):
        """Return a lighter or darker shade for pc_idx out of n_pcs based on the base_rgb."""
        # Convert to HLS, vary lightness
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        # Lightness scale between 0.4 and 0.8
        l_new = 0.4 + 0.4 * (pc_idx / max(n_pcs-1, 1))
        return colorsys.hls_to_rgb(h, l_new, s)

    def plot_spike_raster_fgpc_legend(
        synapses, 
        time_window=None, 
        figsize=(12, 8), 
        save_path=None,
        yticklabel_stride=30,
        show_y_labels=True,
        legend_loc='upper right'
    ):
        synapses_sorted = synapses.sort_values(['functional_group', 'presynaptic_cell'])
        synapses_sorted = synapses_sorted.reset_index(drop=True)

        # Get unique FGs and assign a base color per FG using tab20
        unique_fgs = synapses_sorted['functional_group'].unique()
        cmap = plt.cm.get_cmap('tab20', len(unique_fgs))
        fg_base_colors = {fg: cmap(i)[:3] for i, fg in enumerate(unique_fgs)}

        # Find all PC indices per FG for consistent shading
        fg_to_pcs = {
            fg: sorted(synapses_sorted[synapses_sorted['functional_group']==fg]['presynaptic_cell'].unique())
            for fg in unique_fgs
        }
        fg_pc_color = {}
        for fg, pcs in fg_to_pcs.items():
            n_pcs = len(pcs)
            for i, pc in enumerate(pcs):
                fg_pc_color[(fg, pc)] = get_shaded_color(fg_base_colors[fg], i, n_pcs)

        segments = []
        colors = []
        legend_labels = {}
        yticklabels = []
        yticks = []

        for idx, row in synapses_sorted.iterrows():
            # Robust spike train parsing
            spikes = row['spike_train']
            if isinstance(spikes, str):
                spikes = np.fromstring(spikes.replace('[', '').replace(']', ''), sep=' ')
            elif isinstance(spikes, (float, int)) or spikes is None or (isinstance(spikes, np.ndarray) and spikes.ndim == 0):
                continue
            else:
                spikes = np.array(spikes).flatten()
            if spikes.size == 0:
                continue
            if time_window is not None:
                spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
            segs = [((spk, idx + 0.5), (spk, idx + 1.5)) for spk in spikes]
            segments.extend(segs)
            fg, pc = row['functional_group'], row['presynaptic_cell']
            color = fg_pc_color.get((fg, pc), (0,0,0))
            colors.extend([color]*len(segs))
            key = f'FG{int(fg)}_PC{int(pc)}'
            # Only keep first seen row for legend
            if key not in legend_labels:
                legend_labels[key] = color
            if show_y_labels and (idx % yticklabel_stride) == 0:
                yticklabels.append(key)
                yticks.append(idx + 1)

        fig, ax = plt.subplots(figsize=figsize)
        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=0.7)
            ax.add_collection(lc)

        # X limits
        if time_window is not None:
            ax.set_xlim(time_window)
        else:
            try:
                all_spikes = np.hstack([
                    np.fromstring(row['spike_train'].replace('[','').replace(']',''), sep=' ') if isinstance(row['spike_train'], str)
                    else np.array(row['spike_train']).flatten()
                    for _, row in synapses_sorted.iterrows()
                    if not isinstance(row['spike_train'], (float, int)) and row['spike_train'] is not None
                ])
                ax.set_xlim(np.nanmin(all_spikes), np.nanmax(all_spikes))
            except:
                ax.set_xlim(0, 1000)

        ax.set_ylim(0.5, len(synapses_sorted) + 0.5)
        ax.set_ylabel("Synapse (FG/PC grouped)")
        ax.set_xlabel("Time (ms or sample)")

        if show_y_labels:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=6)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.set_title("Spike Raster Plot (FG color, PC shade, legend)")

        # Legend
        patches = [Patch(color=color, label=label) for label, color in legend_labels.items()]
        # Optionally, only show legend for first N FG/PC combos
        if len(patches) > 25:
            patches = patches[:25]
        ax.legend(handles=patches, loc=legend_loc, title='FG_PC', fontsize=7, title_fontsize=8, frameon=True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
    def analyze_cluster_statistics(self, 
                                 functional_group_id: int = None,
                                 synapse_type: str = None) -> Dict:
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
                                    functional_group: int = None,
                                    figsize: Tuple[int, int] = (10, 6),
                                    save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of firing rates across synapses.
        
        Args:
            synapse_type: Optional synapse type to filter by
            functional_group: Optional functional group to filter by
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        # Filter synapses
        mask = pd.Series(True, index=self.synapses.index)
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type)
        if functional_group is not None:
            mask &= self.synapses['functional_group'] == functional_group
            
        filtered_synapses = self.synapses[mask]
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.histplot(data=filtered_synapses, x='pc_mean_firing_rate', bins=30)
        plt.xlabel('Mean Firing Rate (Hz)')
        plt.ylabel('Count')
        plt.title('Distribution of Synapse Firing Rates')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_weight_distribution(self,
                               synapse_type: str = None,
                               functional_group: int = None,
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of synapse weights.
        
        Args:
            synapse_type: Optional synapse type to filter by
            functional_group: Optional functional group to filter by
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        # Filter synapses
        mask = pd.Series(True, index=self.synapses.index)
        if synapse_type:
            mask &= self.synapses['name'].str.contains(synapse_type)
        if functional_group is not None:
            mask &= self.synapses['functional_group'] == functional_group
            
        filtered_synapses = self.synapses[mask]
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.histplot(data=filtered_synapses, x='initW', bins=30)
        plt.xlabel('Initial Weight')
        plt.ylabel('Count')
        plt.title('Distribution of Synapse Weights')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_cluster_spatial_distribution(self,
                                        synapse_type: str = None,
                                        functional_group: int = None,
                                        figsize: Tuple[int, int] = (10, 10),
                                        save_path: Optional[str] = None) -> None:
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
        plt.show()
        
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
                              save_path: Optional[str] = None) -> None:
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
        plt.show() 

        
    def plot_all_synapse_clusters(self,
        synapse_coord_cols=('pc_0', 'pc_1', 'pc_2'),
        plot_both_together=True,
        plot_each_type_separately=True,
        show=True
    ):
        """
        Plot synapse clusters (exc & inh) for a simulation directory.

        Args:
            sim_dir (str): Directory containing simulation files.
            synapses (pd.DataFrame): DataFrame of synapses, must have 'seg_id' and coordinate columns.
            synapse_coord_cols (tuple): Columns in synapses_with_seg_info to use as xyz.
            plot_both_together (bool): If True, plot both exc and inh clusters in one figure.
            plot_each_type_separately (bool): If True, plot each section type as a separate figure.
            show (bool): Whether to display the plots.
        """
        # Load parameters and segment data
        with open(os.path.join(self.sim_dir, "parameters.pickle"), 'rb') as file:
            parameters = pickle.load(file)
        seg_data = pd.read_csv(os.path.join(self.sim_dir, "segment_data.csv"))

        # Join synapses with segment info
        synapses_with_seg_info = self.synapses.merge(
            seg_data,
            on='seg_id',
            how='left',
            suffixes=('', '_seg')
        )
        synapse_coords = synapses_with_seg_info[list(synapse_coord_cols)].values

        # Plot both excitatory and inhibitory clusters together
        if plot_both_together:
            fig = plt.figure(figsize=(20, 10))
            # Excitatory clusters
            ax1 = fig.add_subplot(121, projection='3d')
            plot_clusters(
                seg_data=seg_data,
                clustering_config=parameters.exc_clustering,
                synapse_coords=synapse_coords,
                ax=ax1,
                elevation=20,
                azimuth=-100,
                title='Excitatory Clusters'
            )
            # Inhibitory clusters
            ax2 = fig.add_subplot(122, projection='3d')
            plot_clusters(
                seg_data=seg_data,
                clustering_config=parameters.inh_clustering,
                synapse_coords=synapse_coords,
                ax=ax2,
                elevation=20,
                azimuth=-100,
                title='Inhibitory Clusters'
            )
            plt.tight_layout()
            if show: plt.show()

        # Plot each excitatory section type
        if plot_each_type_separately:
            for sec_type in parameters.exc_clustering.keys():
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                plot_clusters(
                    seg_data=seg_data,
                    clustering_config={sec_type: parameters.exc_clustering[sec_type]},
                    synapse_coords=synapse_coords,
                    ax=ax,
                    elevation=20,
                    azimuth=-100,
                    title=f'Excitatory Clusters - {sec_type}'
                )
                plt.tight_layout()
                if show: plt.show()
            # Plot each inhibitory section type
            for sec_type in parameters.inh_clustering.keys():
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                plot_clusters(
                    seg_data=seg_data,
                    clustering_config={sec_type: parameters.inh_clustering[sec_type]},
                    synapse_coords=synapse_coords,
                    ax=ax,
                    elevation=20,
                    azimuth=-100,
                    title=f'Inhibitory Clusters - {sec_type}'
                )
                plt.tight_layout()
                if show: plt.show()