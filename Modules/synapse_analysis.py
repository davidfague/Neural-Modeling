import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from scipy import stats
import os

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
                         time_window: Tuple[float, float] = None,
                         synapse_types: List[str] = None,
                         functional_groups: List[int] = None,
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
        """
        Generate a spike raster plot for the synapses.
        
        Args:
            time_window: Tuple of (start_time, end_time) to plot
            synapse_types: List of synapse types to include (e.g. ['exc', 'inh'])
            functional_groups: List of functional group IDs to include
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        # Filter synapses based on criteria
        mask = pd.Series(True, index=self.synapses.index)
        if synapse_types:
            mask &= self.synapses['name'].str.contains('|'.join(synapse_types))
        if functional_groups is not None:
            mask &= self.synapses['functional_group'].isin(functional_groups)
        
        filtered_synapses = self.synapses[mask]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot each synapse's spikes
        for idx, row in filtered_synapses.iterrows():
            spikes = row['spike_train']
            if time_window:
                spikes = spikes[(spikes >= time_window[0]) & (spikes <= time_window[1])]
            plt.plot(spikes, [idx] * len(spikes), 'k.', markersize=1)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Synapse Index')
        plt.title('Spike Raster Plot')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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