#!/usr/bin/env python3
"""
Analyze temporal relationship between NMDA and CA spikes.
Determines whether NMDA spikes tend to cause CA spikes or vice versa.
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy import signal, stats
from pathlib import Path
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class SpikeAnalyzer:
    """Analyze causal relationships between NMDA and CA spikes."""
    
    def __init__(self, sim_path, time_step='saved_at_step_20000'):
        """
        Initialize analyzer with simulation data path.
        
        Parameters
        ----------
        sim_path : str or Path
            Path to simulation folder
        time_step : str
            Which time step to analyze (default: 'saved_at_step_20000')
        """
        self.sim_path = Path(sim_path)
        self.time_step = time_step
        self.raw_data_path = self.sim_path / 'raw_data' / time_step
        
        # Load spike tables
        self.nmda_spikes = self._load_spike_table('nmda.csv')
        self.ca_spikes = self._load_spike_table('ca.csv')
        self.na_spikes = self._load_spike_table('na.csv')
        
        # Load raw traces
        self.voltage = self._load_h5('v.h5')
        self.i_ca = self._load_h5('ica.h5')
        self.i_nmda = self._load_h5('inmda.h5')
        self.g_na = self._load_h5('gNaTa_t_NaTa_t.h5')
        
        # Time vector (assuming 0.025 ms dt)
        self.dt = 0.025  # ms
        self.time = np.arange(self.voltage.shape[0]) * self.dt
        
        print(f"Loaded data from {self.raw_data_path}")
        print(f"Voltage shape: {self.voltage.shape}")
        print(f"Time range: {self.time[0]:.2f} - {self.time[-1]:.2f} ms")
        print(f"NMDA spikes found: {len(self.nmda_spikes)}")
        print(f"CA spikes found: {len(self.ca_spikes)}")
        print(f"NA spikes found: {len(self.na_spikes)}")
        
        # Create output directory for plots
        self.output_dir = self.sim_path / 'nmda_ca_causality'
        self.output_dir.mkdir(exist_ok=True)
        print(f"Plots will be saved to: {self.output_dir}")
    
    def _load_spike_table(self, filename):
        """Load spike table CSV."""
        path = self.sim_path / filename
        if path.exists():
            df = pd.read_csv(path)
            # Filter out rows with NaN in spike bound columns and calculate spike time
            if 'nmda_lower_bound' in df.columns and 'nmda_upper_bound' in df.columns:
                df = df.dropna(subset=['nmda_lower_bound', 'nmda_upper_bound'])
                df['spike_time'] = (df['nmda_lower_bound'] + df['nmda_upper_bound']) / 2
            elif 'ca_lower_bound' in df.columns and 'ca_upper_bound' in df.columns:
                df = df.dropna(subset=['ca_lower_bound', 'ca_upper_bound'])
                df['spike_time'] = (df['ca_lower_bound'] + df['ca_upper_bound']) / 2
            elif 'na_lower_bound' in df.columns and 'na_upper_bound' in df.columns:
                df = df.dropna(subset=['na_lower_bound', 'na_upper_bound'])
                df['spike_time'] = (df['na_lower_bound'] + df['na_upper_bound']) / 2
            return df
        else:
            print(f"Warning: {filename} not found")
            return pd.DataFrame()
    
    def _load_h5(self, filename):
        """Load HDF5 data file."""
        path = self.raw_data_path / filename
        with h5py.File(path, 'r') as f:
            # Get first dataset (usually the only one)
            key = list(f.keys())[0]
            data = f[key][:]
        return data
    
    def detect_current_spikes(self, current, threshold_percentile=95, min_distance_ms=2):
        """
        Detect spikes in current trace using threshold crossing.
        
        Parameters
        ----------
        current : ndarray
            Current trace (time x locations)
        threshold_percentile : float
            Percentile for threshold (for negative currents, uses negative percentile)
        min_distance_ms : float
            Minimum distance between spikes in ms
        
        Returns
        -------
        spike_times : list of ndarray
            Spike times for each location
        spike_indices : list of ndarray
            Spike indices for each location
        """
        n_locs = current.shape[1]
        min_distance_samples = int(min_distance_ms / self.dt)
        
        spike_times = []
        spike_indices = []
        
        for loc in range(n_locs):
            trace = current[:, loc]
            
            # Use negative threshold for inward currents
            if np.mean(trace) < 0:
                threshold = np.percentile(trace, 100 - threshold_percentile)
                peaks, _ = signal.find_peaks(-trace, height=-threshold, distance=min_distance_samples)
            else:
                threshold = np.percentile(trace, threshold_percentile)
                peaks, _ = signal.find_peaks(trace, height=threshold, distance=min_distance_samples)
            
            spike_indices.append(peaks)
            spike_times.append(peaks * self.dt)
        
        return spike_times, spike_indices
    
    def cross_correlate_spikes(self, times1, times2, max_lag_ms=50):
        """
        Compute cross-correlation between two spike trains.
        
        Parameters
        ----------
        times1 : array-like
            First spike train times (ms)
        times2 : array-like
            Second spike train times (ms)
        max_lag_ms : float
            Maximum lag in ms
        
        Returns
        -------
        lags : ndarray
            Lag values in ms
        xcorr : ndarray
            Cross-correlation values
        """
        # Create binned spike trains
        bin_size = 0.5  # ms
        n_bins = int((self.time[-1] - self.time[0]) / bin_size)
        
        bins = np.linspace(self.time[0], self.time[-1], n_bins)
        train1, _ = np.histogram(times1, bins=bins)
        train2, _ = np.histogram(times2, bins=bins)
        
        # Compute cross-correlation
        max_lag_bins = int(max_lag_ms / bin_size)
        xcorr = signal.correlate(train1, train2, mode='full')
        
        # Extract relevant lags
        center = len(xcorr) // 2
        xcorr = xcorr[center - max_lag_bins:center + max_lag_bins + 1]
        lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
        
        # Normalize
        xcorr = xcorr / (np.sqrt(np.sum(train1**2)) * np.sqrt(np.sum(train2**2)) + 1e-10)
        
        return lags, xcorr
    
    def spike_triggered_average(self, trigger_times, signal, window_ms=[-20, 50]):
        """
        Compute spike-triggered average.
        
        Parameters
        ----------
        trigger_times : array-like
            Times to trigger on (ms)
        signal : ndarray
            Signal to average (time x locations)
        window_ms : list
            [pre, post] window in ms around trigger
        
        Returns
        -------
        sta : ndarray
            Spike-triggered average (window_samples x locations)
        time_window : ndarray
            Time vector for window
        """
        pre_samples = int(abs(window_ms[0]) / self.dt)
        post_samples = int(window_ms[1] / self.dt)
        
        n_locs = signal.shape[1] if signal.ndim > 1 else 1
        snippets = []
        
        for t in trigger_times:
            idx = int(t / self.dt)
            if idx - pre_samples >= 0 and idx + post_samples < len(signal):
                if signal.ndim > 1:
                    snippets.append(signal[idx - pre_samples:idx + post_samples, :])
                else:
                    snippets.append(signal[idx - pre_samples:idx + post_samples])
        
        if len(snippets) > 0:
            sta = np.mean(snippets, axis=0)
        else:
            sta = np.zeros((pre_samples + post_samples, n_locs))
        
        time_window = np.arange(-pre_samples, post_samples) * self.dt
        return sta, time_window
    
    def analyze_temporal_ordering(self, loc_idx=0):
        """
        Analyze temporal ordering between NMDA, CA, and NA spikes.
        
        Parameters
        ----------
        loc_idx : int
            Location index to analyze (segmentID)
        """
        # Get spikes for this location
        nmda_loc = self.nmda_spikes[self.nmda_spikes['segmentID'] == loc_idx]
        ca_loc = self.ca_spikes[self.ca_spikes['segmentID'] == loc_idx]
        
        if len(nmda_loc) == 0 or len(ca_loc) == 0:
            print(f"Not enough spikes at location {loc_idx}")
            return None
        
        # Extract spike times
        nmda_times = nmda_loc['spike_time'].values
        ca_times = ca_loc['spike_time'].values
        
        print(f"\nCA spikes detected: {len(ca_times)}")
        print(f"NMDA spikes detected: {len(nmda_times)}")
        
        if len(ca_times) == 0 or len(nmda_times) == 0:
            print("Not enough spikes detected for analysis")
            return None
        
        # For each CA spike, find nearest NMDA spike
        ca_to_nmda_lags = []
        for ca_t in ca_times:
            diffs = nmda_times - ca_t
            if len(diffs) > 0:
                nearest_idx = np.argmin(np.abs(diffs))
                ca_to_nmda_lags.append(diffs[nearest_idx])
        
        # For each NMDA spike, find nearest CA spike
        nmda_to_ca_lags = []
        for nmda_t in nmda_times:
            diffs = ca_times - nmda_t
            if len(diffs) > 0:
                nearest_idx = np.argmin(np.abs(diffs))
                nmda_to_ca_lags.append(diffs[nearest_idx])
        
        results = {
            'ca_spikes': ca_times,
            'nmda_spikes': nmda_times,
            'ca_to_nmda_lags': np.array(ca_to_nmda_lags),
            'nmda_to_ca_lags': np.array(nmda_to_ca_lags),
        }
        
        # Statistical summary
        print("\n" + "="*60)
        print("TEMPORAL ORDERING ANALYSIS")
        print("="*60)
        print(f"\nCA→NMDA lags (negative means NMDA before CA):")
        print(f"  Mean: {np.mean(ca_to_nmda_lags):.2f} ms")
        print(f"  Median: {np.median(ca_to_nmda_lags):.2f} ms")
        print(f"  Std: {np.std(ca_to_nmda_lags):.2f} ms")
        
        print(f"\nNMDA→CA lags (positive means CA after NMDA):")
        print(f"  Mean: {np.mean(nmda_to_ca_lags):.2f} ms")
        print(f"  Median: {np.median(nmda_to_ca_lags):.2f} ms")
        print(f"  Std: {np.std(nmda_to_ca_lags):.2f} ms")
        
        # Determine causality
        print("\n" + "-"*60)
        if np.median(nmda_to_ca_lags) > 0:
            print("CONCLUSION: NMDA spikes tend to PRECEDE CA spikes")
            print(f"  → NMDA likely CAUSES CA spikes")
        else:
            print("CONCLUSION: CA spikes tend to PRECEDE NMDA spikes")
            print(f"  → CA likely CAUSES NMDA spikes")
        print("-"*60)
        
        return results
    
    def plot_comprehensive_analysis(self, loc_idx=0, time_window=None, save_path=None):
        """
        Create comprehensive figure showing voltage, currents, and temporal analysis.
        
        Parameters
        ----------
        loc_idx : int
            Location index to analyze
        time_window : tuple or None
            (start_ms, end_ms) or None for full trace
        save_path : str or None
            Path to save figure
        """
        # Analyze temporal ordering
        results = self.analyze_temporal_ordering(loc_idx)
        
        if results is None:
            print("Cannot create plot - insufficient data")
            return
        
        # Extract results
        ca_spikes_loc = results['ca_spikes']
        nmda_spikes_loc = results['nmda_spikes']
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
        
        # Time window for plotting
        if time_window is not None:
            t_start, t_end = time_window
            mask = (self.time >= t_start) & (self.time <= t_end)
            time_plot = self.time[mask]
        else:
            mask = slice(None)
            time_plot = self.time
        
        # 1. Voltage trace with all spike markers
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_plot, self.voltage[mask, loc_idx], 'k-', linewidth=0.5, alpha=0.7)
        
        # Mark spikes
        for ca_t in ca_spikes_loc:
            if time_window is None or (ca_t >= time_window[0] and ca_t <= time_window[1]):
                ax1.axvline(ca_t, color='red', alpha=0.5, linewidth=1.5, label='CA spike' if ca_t == ca_spikes_loc[0] else '')
        
        for nmda_t in nmda_spikes_loc:
            if time_window is None or (nmda_t >= time_window[0] and nmda_t <= time_window[1]):
                ax1.axvline(nmda_t, color='blue', alpha=0.5, linewidth=1.5, label='NMDA spike' if nmda_t == nmda_spikes_loc[0] else '')
        
        ax1.set_ylabel('Voltage (mV)', fontsize=12)
        ax1.set_title(f'Voltage Trace with Spike Markers (Location {loc_idx})', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Calcium current
        ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
        ax2.plot(time_plot, self.i_ca[mask, loc_idx], 'r-', linewidth=0.8)
        for ca_t in ca_spikes_loc:
            if time_window is None or (ca_t >= time_window[0] and ca_t <= time_window[1]):
                ax2.axvline(ca_t, color='red', alpha=0.3, linewidth=1)
        ax2.set_ylabel('I_Ca (nA)', fontsize=12)
        ax2.set_title('Calcium Current', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. NMDA current
        ax3 = fig.add_subplot(gs[2, :], sharex=ax1)
        ax3.plot(time_plot, self.i_nmda[mask, loc_idx], 'b-', linewidth=0.8)
        for nmda_t in nmda_spikes_loc:
            if time_window is None or (nmda_t >= time_window[0] and nmda_t <= time_window[1]):
                ax3.axvline(nmda_t, color='blue', alpha=0.3, linewidth=1)
        ax3.set_ylabel('I_NMDA (nA)', fontsize=12)
        ax3.set_xlabel('Time (ms)', fontsize=12)
        ax3.set_title('NMDA Current', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Temporal lag distributions
        ax4 = fig.add_subplot(gs[3, 0])
        ax4.hist(results['nmda_to_ca_lags'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(0, color='black', linestyle='--', linewidth=2)
        ax4.axvline(np.median(results['nmda_to_ca_lags']), color='red', linestyle='-', linewidth=2, label=f"Median: {np.median(results['nmda_to_ca_lags']):.2f} ms")
        ax4.set_xlabel('Lag (ms)', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('NMDA→CA Lag Distribution\n(+ve: CA follows NMDA)', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cross-correlation
        ax5 = fig.add_subplot(gs[3, 1])
        if len(ca_spikes_loc) > 5 and len(nmda_spikes_loc) > 5:
            lags, xcorr = self.cross_correlate_spikes(nmda_spikes_loc, ca_spikes_loc, max_lag_ms=50)
            ax5.plot(lags, xcorr, 'k-', linewidth=2)
            ax5.axvline(0, color='red', linestyle='--', linewidth=2)
            peak_lag = lags[np.argmax(xcorr)]
            ax5.axvline(peak_lag, color='green', linestyle='-', linewidth=2, label=f'Peak: {peak_lag:.2f} ms')
            ax5.set_xlabel('Lag (ms)', fontsize=11)
            ax5.set_ylabel('Cross-correlation', fontsize=11)
            ax5.set_title('NMDA-CA Cross-Correlation\n(+ve lag: NMDA leads)', fontsize=11, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Spike-triggered average (NMDA triggers, CA response)
        ax6 = fig.add_subplot(gs[4, 0])
        if len(nmda_spikes_loc) > 5:
            sta_ca, sta_time = self.spike_triggered_average(nmda_spikes_loc, self.i_ca[:, loc_idx])
            ax6.plot(sta_time, sta_ca, 'r-', linewidth=2)
            ax6.axvline(0, color='black', linestyle='--', linewidth=2)
            ax6.set_xlabel('Time from NMDA spike (ms)', fontsize=11)
            ax6.set_ylabel('I_Ca (nA)', fontsize=11)
            ax6.set_title('NMDA-triggered CA Current Average', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Spike-triggered average (CA triggers, NMDA response)
        ax7 = fig.add_subplot(gs[4, 1])
        if len(ca_spikes_loc) > 5:
            sta_nmda, sta_time = self.spike_triggered_average(ca_spikes_loc, self.i_nmda[:, loc_idx])
            ax7.plot(sta_time, sta_nmda, 'b-', linewidth=2)
            ax7.axvline(0, color='black', linestyle='--', linewidth=2)
            ax7.set_xlabel('Time from CA spike (ms)', fontsize=11)
            ax7.set_ylabel('I_NMDA (nA)', fontsize=11)
            ax7.set_title('CA-triggered NMDA Current Average', fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle(f'NMDA-CA Spike Causality Analysis\nLocation: {loc_idx}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        else:
            # Save to default location
            default_path = self.output_dir / f"nmda_ca_causality_loc{loc_idx}_full.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {default_path}")
        
        plt.close()


def main():
    """Main analysis script."""
    # Set simulation path
    sim_path = "/home/drfrbc/Neural-Modeling/simulations/2025-12-10-19-57-tuning_perisomatic_inh_increased_nexus_excitation/sta_Complex_NexInhDen0.1_PeriInhDen0.11_DistBasInhDen0.1_TuftInhDen0.2_TuftDistExcDen6_DistBasLocL5ExcDen2.6_Np5000"
    
    # Initialize analyzer
    analyzer = SpikeAnalyzer(sim_path, time_step='saved_at_step_20000')
    
    # Find locations with both NMDA and CA spikes
    print("\n" + "="*80)
    print("FINDING LOCATIONS WITH BOTH SPIKE TYPES")
    print("="*80)
    
    nmda_locs = analyzer.nmda_spikes['segmentID'].unique()
    ca_locs = analyzer.ca_spikes['segmentID'].unique()
    common_locs = set(nmda_locs) & set(ca_locs)
    
    print(f"Locations with NMDA spikes: {len(nmda_locs)}")
    print(f"Locations with CA spikes: {len(ca_locs)}")
    print(f"Locations with BOTH: {len(common_locs)}")
    
    # Count spikes at each common location
    spike_counts = []
    for loc in common_locs:
        n_nmda = len(analyzer.nmda_spikes[analyzer.nmda_spikes['segmentID'] == loc])
        n_ca = len(analyzer.ca_spikes[analyzer.ca_spikes['segmentID'] == loc])
        spike_counts.append({'location': loc, 'nmda': n_nmda, 'ca': n_ca, 'total': n_nmda + n_ca})
    
    spike_counts_df = pd.DataFrame(spike_counts).sort_values('ca', ascending=False)
    print("\nTop 10 locations by CA spike count:")
    print(spike_counts_df.head(10))
    
    # Analyze best location
    best_loc = int(spike_counts_df.iloc[0]['location'])
    print(f"\n{'='*80}")
    print(f"ANALYZING LOCATION {best_loc} (highest CA spike count)")
    print(f"{'='*80}")
    
    # Create comprehensive analysis plot
    analyzer.plot_comprehensive_analysis(loc_idx=best_loc)
    
    # Analyze multiple locations for global statistics
    print("\n" + "="*80)
    print("GLOBAL ANALYSIS ACROSS MULTIPLE LOCATIONS")
    print("="*80)
    
    all_nmda_to_ca = []
    all_ca_to_nmda = []
    
    # Sample up to 20 locations with most spikes
    for i, row in spike_counts_df.head(20).iterrows():
        loc = int(row['location'])
        print(f"\nAnalyzing location {loc}...")
        result = analyzer.analyze_temporal_ordering(loc_idx=loc)
        if result is not None:
            all_nmda_to_ca.extend(result['nmda_to_ca_lags'])
            all_ca_to_nmda.extend(result['ca_to_nmda_lags'])
    
    # Global statistics
    if len(all_nmda_to_ca) > 0:
        all_nmda_to_ca = np.array(all_nmda_to_ca)
        
        print("\n" + "="*80)
        print("FINAL GLOBAL SUMMARY")
        print("="*80)
        print(f"\nTotal spike pairs analyzed: {len(all_nmda_to_ca):,}")
        print(f"\nNMDA→CA lag statistics:")
        print(f"  Mean: {np.mean(all_nmda_to_ca):.3f} ms")
        print(f"  Median: {np.median(all_nmda_to_ca):.3f} ms")
        print(f"  Std: {np.std(all_nmda_to_ca):.3f} ms")
        print(f"  25th percentile: {np.percentile(all_nmda_to_ca, 25):.3f} ms")
        print(f"  75th percentile: {np.percentile(all_nmda_to_ca, 75):.3f} ms")
        
        percent_after = 100 * np.mean(all_nmda_to_ca > 0)
        print(f"\nPercentage of CA spikes occurring AFTER NMDA: {percent_after:.1f}%")
        
        print("\n" + "="*80)
        if np.median(all_nmda_to_ca) > 0.5:
            print("✓ STRONG EVIDENCE: NMDA spikes CAUSE CA spikes")
            print(f"  CA spikes typically occur {np.median(all_nmda_to_ca):.3f} ms after NMDA spikes")
        elif np.median(all_nmda_to_ca) > 0:
            print("✓ MODERATE EVIDENCE: NMDA spikes tend to precede CA spikes")
            print(f"  CA spikes typically occur {np.median(all_nmda_to_ca):.3f} ms after NMDA spikes")
        elif np.median(all_nmda_to_ca) < -0.5:
            print("✓ STRONG EVIDENCE: CA spikes CAUSE NMDA spikes")
            print(f"  NMDA spikes typically occur {-np.median(all_nmda_to_ca):.3f} ms after CA spikes")
        else:
            print("✓ WEAK EVIDENCE: CA and NMDA spikes occur nearly simultaneously")
            print(f"  Median lag: {np.median(all_nmda_to_ca):.3f} ms")
        print("="*80)


if __name__ == "__main__":
    main()