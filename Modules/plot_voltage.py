import matplotlib.pyplot as plt
import numpy as np
from Modules.dendritic_spike_times import get_dendritic_spike_times

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