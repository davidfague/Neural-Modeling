import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from Modules.dendritic_spikes.dendritic_spike_times import get_dendritic_spike_times

def plot_voltage(sim_data, indices, colors, xlims=None, title_suffix="", 
                 save_file=None, show=False, dendritic_dfs=None, 
                 start_step=None, end_step=None, plot_soma_spike_times=True, additional_title_suffixes=None):
    """
    Plot voltage traces for given segment indices, marking dendritic spikes
    and soma spikes if requested.

    Parameters
    ----------
    sim_data : dict
        Simulation data with keys 'v' (voltage matrix) and 'spktimes' (soma spike times).
    indices : list[int]
        Segment indices to plot.
    colors : list[str]
        Colors for each segment.
    xlims : tuple, optional
        x-axis limits (time).
    title_suffix : str
        Extra text for title.
    save_file : str, optional
        Base file path for saving figures (appends index).
    show : bool
        Whether to show the figure interactively.
    dendritic_dfs : dict[str, pd.DataFrame], optional
        Dict mapping spike type → DataFrame with spike times.
    start_step, end_step : int, optional
        Time filtering bounds for spikes.
    plot_soma_spike_times : bool
        If True, plot vertical dashed lines for soma spikes.
    """
    colors = [color.split('*')[0] for color in colors]
    t = np.arange(sim_data['v'].shape[0])  # assume row index is time

    # Spike marker color scheme
    spike_colors = {
        "ca": {"lower": "#FF8C00", "upper": "#FFA500"},   # dark orange, light orange
        "nmda": {"lower": "k", "upper": "grey"},          # black, grey
        "na": {"lower": "c", "upper": None},              # cyan, no upper
    }

    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 6))
        plt.plot(sim_data['v'][:, idx], colors[i], label=f"Seg {idx} voltage")
        plt.ylim([-80, 0])

        if xlims:
            plt.xlim(xlims)

        plt.axhline(y=-60, color=colors[i], linestyle='--')
        plt.title(f'Voltage at index {idx} {title_suffix} {additional_title_suffixes[i] if additional_title_suffixes else ""}')

        # y_max = np.max(sim_data['v'][start:end, idx]) # Get the maximum y (voltage) in the plotted window for dspike marker placement
        y_max = np.max(sim_data['v'][:, idx]) # Get the maximum y (voltage) for dspike marker placement

        # Add dendritic spike markers if provided
        if dendritic_dfs is not None:
            for spike_type, df in dendritic_dfs.items():
                spike_times = get_dendritic_spike_times(df, idx, spike_type, start_step, end_step)
                if len(spike_times["lower_bound"]) > 0:
                    plt.scatter(spike_times["lower_bound"], np.full_like(spike_times["lower_bound"], y_max), #sim_data['v'][spike_times["lower_bound"], idx],  # can choose to place on the voltage instead of at y_max
                                marker='*', color=spike_colors[spike_type]["lower"], label=f"{spike_type} start")
                if len(spike_times["upper_bound"]) > 0:
                    plt.scatter(spike_times["upper_bound"], 
                                np.full_like(spike_times["upper_bound"], y_max), # sim_data['v'][spike_times["upper_bound"], idx], # can choose to place on the voltage instead of at y_max
                                marker='*', color=spike_colors[spike_type]["upper"], label=f"{spike_type} end")


        # Add soma spike markers
        if plot_soma_spike_times and "spktimes" in sim_data:
            soma_spike_times = np.array(sim_data['spktimes'])
            soma_spike_times = soma_spike_times / 0.1  # Convert from ms to steps assuming dt=0.1ms
            if start_step is not None and end_step is not None:
                soma_spike_times = soma_spike_times[(soma_spike_times >= start_step) & (soma_spike_times <= end_step)]
            for event_time in soma_spike_times:
                plt.axvline(event_time, linestyle='--', color='grey', label="soma spike" if event_time == soma_spike_times[0] else None)

        plt.legend()

        if save_file:
            plt.savefig(f"{save_file}_{idx}.png", format='png', bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        plt.close()


def plot_voltages_centered_on_spikes(sim_data, seg_ids, dendritic_dfs, output_dir, 
                                      window_ms=200, dt=0.1, show=False, seed=42, seg_descriptions=None):
    """
    Plot voltage traces centered around random dendritic spikes for each spike type.
    Creates separate plots for NMDA, Na, and Ca spikes for each segment.
    
    Parameters
    ----------
    sim_data : dict
        Simulation data with keys 'v' (voltage matrix) and 'spktimes' (soma spike times).
    seg_ids : list[int]
        Segment IDs to plot.
    dendritic_dfs : dict[str, pd.DataFrame]
        Dict mapping spike type ('ca', 'nmda', 'na') → DataFrame with spike times.
    output_dir : str
        Base directory for saving plots (will create subdirs for each spike type).
    window_ms : float
        Time window in ms around the spike center (default: 200ms = ±100ms from spike).
    dt : float
        Timestep in ms (default: 0.1ms).
    show : bool
        Whether to show the figure interactively.
    seed : int
        Random seed for selecting spike times.
    seg_descriptions : dict, optional
        Dictionary mapping seg_id to description string (e.g., {1547: "[used in nexus calc]"}).
    """
    if dendritic_dfs is None:
        print("[plot_voltage.py] No dendritic spike data provided.")
        return
    
    rng = np.random.default_rng(seed)
    spike_types = ["nmda", "na", "ca"]
    
    # Spike marker color scheme
    spike_colors = {
        "ca": {"lower": "#FF8C00", "upper": "#FFA500"},
        "nmda": {"lower": "k", "upper": "grey"},
        "na": {"lower": "c", "upper": None},
    }
    
    # Create output directories for each spike type
    for spike_type in spike_types:
        spike_dir = os.path.join(output_dir, spike_type.upper())
        os.makedirs(spike_dir, exist_ok=True)
    
    # For each spike type, plot all segments together with centered window
    for spike_type in spike_types:
        spike_dir = os.path.join(output_dir, spike_type.upper())
        
        if spike_type not in dendritic_dfs:
            # Create empty plot with "No spikes" message
            for seg_id in seg_ids:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, f"No {spike_type.upper()} spikes", 
                       ha='center', va='center', fontsize=20, transform=ax.transAxes)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.axis('off')
                plt.savefig(os.path.join(spike_dir, f"seg_{seg_id}.png"), 
                           format='png', bbox_inches="tight", dpi=300)
                if show:
                    plt.show()
                plt.close(fig)
            continue
        
        df = dendritic_dfs[spike_type]
        
        # Collect all spike times across all segments to find common window
        all_spike_times = []
        seg_spike_dict = {}
        
        for seg_id in seg_ids:
            spike_times = get_dendritic_spike_times(df, seg_id, spike_type)
            lower_bounds = spike_times["lower_bound"]
            
            if len(lower_bounds) > 0:
                # Randomly select one spike time for this segment
                selected_spike = rng.choice(lower_bounds)
                seg_spike_dict[seg_id] = selected_spike
                all_spike_times.append(selected_spike)
            else:
                seg_spike_dict[seg_id] = None
        
        # If no spikes found, create empty plots
        if len(all_spike_times) == 0:
            for seg_id in seg_ids:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, f"No {spike_type.upper()} spikes", 
                       ha='center', va='center', fontsize=20, transform=ax.transAxes)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.axis('off')
                plt.savefig(os.path.join(spike_dir, f"seg_{seg_id}.png"), 
                           format='png', bbox_inches="tight", dpi=300)
                if show:
                    plt.show()
                plt.close(fig)
            continue
        
        # Create a plot for each segment, centered on THAT segment's spike
        window_steps = int(window_ms / dt)
        half_window = window_steps // 2
        
        for seg_id in seg_ids:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            selected_spike = seg_spike_dict.get(seg_id)
            
            if selected_spike is None:
                # No spikes for this segment
                ax.text(0.5, 0.5, f"No {spike_type.upper()} spikes", 
                       ha='center', va='center', fontsize=20, transform=ax.transAxes)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.axis('off')
            else:
                # Center the window on THIS segment's selected spike
                center_time = selected_spike
                start_step = max(0, center_time - half_window)
                end_step = min(sim_data['v'].shape[0], center_time + half_window)
                
                # Plot voltage trace
                time_steps = np.arange(start_step, end_step)
                time_ms = time_steps * dt  # Convert steps to milliseconds
                voltage_trace = sim_data['v'][start_step:end_step, seg_id]
                
                ax.plot(time_ms, voltage_trace, 'b-', label=f"Seg {seg_id} voltage")
                ax.set_ylim([-80, 0])
                ax.axhline(y=-60, color='gray', linestyle='--', alpha=0.5)
                
                # Mark all spikes of this type in this segment within the window
                spike_times = get_dendritic_spike_times(df, seg_id, spike_type, start_step, end_step)
                y_max = np.max(voltage_trace)
                
                if len(spike_times["lower_bound"]) > 0:
                    ax.scatter(spike_times["lower_bound"] * dt, 
                              np.full_like(spike_times["lower_bound"], y_max, dtype=float),
                              marker='*', s=100, color=spike_colors[spike_type]["lower"], 
                              label=f"{spike_type.upper()} start", zorder=5)
                
                if spike_type in ["ca", "nmda"] and len(spike_times["upper_bound"]) > 0:
                    ax.scatter(spike_times["upper_bound"] * dt, 
                              np.full_like(spike_times["upper_bound"], y_max, dtype=float),
                              marker='*', s=100, color=spike_colors[spike_type]["upper"], 
                              label=f"{spike_type.upper()} end", zorder=5)
                
                # Mark the selected spike with a vertical line
                ax.axvline(selected_spike * dt, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label='Selected spike')
                
                # Add soma spike markers if available
                if "spktimes" in sim_data:
                    soma_spike_times_ms = np.array(sim_data['spktimes'])  # Already in ms
                    soma_spike_times_ms = soma_spike_times_ms[
                        (soma_spike_times_ms >= start_step * dt) & (soma_spike_times_ms <= end_step * dt)
                    ]
                    for i, event_time in enumerate(soma_spike_times_ms):
                        ax.axvline(event_time, linestyle='--', color='grey', alpha=0.5,
                                  label="Soma spike" if i == 0 else None)
                
                # Get segment description if available
                seg_desc = ""
                if seg_descriptions and seg_id in seg_descriptions:
                    seg_desc = f" {seg_descriptions[seg_id]}"
                
                ax.set_xlabel(f"Time (ms)")
                ax.set_ylabel("Voltage (mV)")
                ax.set_title(f'{spike_type.upper()} spike at seg {seg_id}{seg_desc} (centered at t={center_time * dt:.1f} ms)')
                ax.set_xlim([start_step * dt, end_step * dt])  # Explicitly set x-axis limits in ms
                ax.legend(loc='best')
            
            plt.tight_layout()
            plt.savefig(os.path.join(spike_dir, f"seg_{seg_id}.png"), 
                       format='png', bbox_inches="tight", dpi=300)
            if show:
                plt.show()
            plt.close(fig)
        
        print(f"[plot_voltage.py] Saved {spike_type.upper()} spike-centered plots to {spike_dir}")


# def plot_voltage(sim_data, indices, colors, xlims=None, title_suffix="", save_file = None, show=False):
#     colors = [color.split('*')[0] for color in colors]
#     for i, idx in enumerate(indices):
#         plt.figure(figsize=(12, 6))

#         # plt.subplot(1, 1, 1)
#         # plt.plot(sim_data['v'][time_points, idx], colors[i])
#         plt.plot(sim_data['v'][:, idx], colors[i])
#         plt.ylim([-80, 0])
#         # plt.xlim(100000,110000) # reduced
#         # plt.xlim(50000, 60000) # complex
#         if xlims:
#             plt.xlim(xlims)
#         plt.axhline(y=-60, color=colors[i], linestyle='--')
#         plt.title(f'Voltage at index {idx} {title_suffix}')
        
#         # plt.subplot(1, 2, 2)
#         # plt.plot(sim_data['v'][time_points, segment_mapping[idx]], colors[i])
#         # plt.ylim([-90, 10])
#         # plt.title(f'Refactored Model - Voltage at index {segment_mapping[idx]} {title_suffix}')
#         if save_file:
#             plt.savefig(f"{save_file}_{idx}.png", format='png', bbox_inches="tight", dpi=300)
#         if show:
#             plt.show()