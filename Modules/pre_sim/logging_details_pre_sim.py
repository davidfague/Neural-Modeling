"""
Modules/pre_sim/logging_details_pre_sim.py

Report generation and formatting functions for pre-simulation stage.
Extracted from scripts/AA_pre_sim.py.
"""

import os
import pandas as pd


def write_spike_train_config(sims_dir, inh_syn_properties, exc_syn_properties):
    """
    Write spike_train_mode configurations to a text file in the simulation set folder.
    
    Renamed from write_spike_train_modes() in AA_pre_sim.py.
    
    Args:
        sims_dir: Path to simulations directory
        inh_syn_properties: Dictionary of inhibitory synapse properties
        exc_syn_properties: Dictionary of excitatory synapse properties
    """
    with open(os.path.join(sims_dir, 'spike_train_modes.txt'), 'w') as f:
        f.write("Inhibitory Synapse Spike Train Modes:\n")
        for input_source, props in inh_syn_properties.items():
            f.write(f"{input_source}: {props['spike_train_mode']}\n")
        f.write("\nExcitatory Synapse Spike Train Modes:\n")
        for input_source, props in exc_syn_properties.items():
            f.write(f"{input_source}: {props['spike_train_mode']}\n")


def format_synapse_densities(params, syn_counts):
    """
    Format synapse density information for exc and inh synapses.
    
    Refactored from write_synapse_densities() in AA_pre_sim.py.
    Now returns formatted string instead of printing directly.
    
    Args:
        params: SimulationParameters object
        syn_counts: Dictionary mapping input_source to synapse count
    
    Returns:
        Formatted string with synapse density information
    """
    lines = []
    lines.append("Exc synapse densities and counts:")
    for input_source, props in params.exc_syn_properties.items():
        dens = props.get("syn_density")
        dens_str = "None" if dens is None else round(dens, 4)
        n_syn = syn_counts.get(input_source, 0)
        lines.append(f"{input_source}: density={dens_str}, n_synapses={n_syn}")
    lines.append("")
    
    lines.append("Inh synapse densities and counts:")
    for input_source, props in params.inh_syn_properties.items():
        dens = props.get("syn_density")
        dens_str = "None" if dens is None else round(dens, 4)
        n_syn = syn_counts.get(input_source, 0)
        lines.append(f"{input_source}: density={dens_str}, n_synapses={n_syn}")
    lines.append("")
    
    return "\n".join(lines)


def format_weight_distributions(params):
    """
    Format weight distribution information for exc and inh synapses.
    
    Refactored from write_weight_distributions() in AA_pre_sim.py.
    Now returns formatted string instead of printing directly.
    
    Args:
        params: SimulationParameters object
    
    Returns:
        Formatted string with weight distribution information
    """
    lines = []
    lines.append("Exc synapse weight distributions:")
    for input_source, props in params.exc_syn_properties.items():
        init_wt_dist = props.get("initial_weight_distribution", {}) or {}
        wt_func = init_wt_dist.get("function")
        wt_func_name = wt_func.__name__ if wt_func else "None"
        wt_params = init_wt_dist.get("params", {}) or {}
        lines.append(f"{input_source}: {wt_func_name}: {wt_params}")
    lines.append("")
    
    lines.append("Inh synapse weight distributions:")
    for input_source, props in params.inh_syn_properties.items():
        init_wt_dist = props.get("initial_weight_distribution", {}) or {}
        wt_func = init_wt_dist.get("function")
        wt_func_name = wt_func.__name__ if wt_func else "None"
        wt_params = init_wt_dist.get("params", {}) or {}
        lines.append(f"{input_source}: {wt_func_name}: {wt_params}")
    lines.append("")
    
    return "\n".join(lines)


def format_firing_rate_distributions(params):
    """
    Format firing rate distribution information for exc and inh synapses.
    
    Refactored from write_firing_rate_distributions() in AA_pre_sim.py.
    Now returns formatted string instead of printing directly.
    
    Args:
        params: SimulationParameters object
    
    Returns:
        Formatted string with firing rate distribution information
    """
    lines = []
    lines.append("Exc synapse firing rate distributions:")
    for input_source, props in params.exc_syn_properties.items():
        fr_dist = props.get("mean_firing_rate_distribution", {}) or {}
        fr_func = fr_dist.get("function")
        fr_func_name = fr_func.__name__ if fr_func else "None"
        fr_params = fr_dist.get("params", {}) or {}
        fr_shift = props.get("fr_shift", 0)
        fr_shift_str = f'fr_shift: {fr_shift}' if fr_shift != 0 else ''
        lines.append(f"{input_source}: {fr_func_name}: {fr_params} {fr_shift_str}")
    lines.append("")
    
    lines.append("Inh synapse firing rate distributions:")
    for input_source, props in params.inh_syn_properties.items():
        fr_dist = props.get("mean_firing_rate_distribution", {}) or {}
        fr_func = fr_dist.get("function")
        fr_func_name = fr_func.__name__ if fr_func else "None"
        fr_params = fr_dist.get("params", {}) or {}
        fr_shift = props.get("fr_shift", 0)
        fr_shift_str = f'fr_shift: {fr_shift}' if fr_shift != 0 else ''
        lines.append(f"{input_source}: {fr_func_name}: {fr_params} {fr_shift_str}")
    lines.append("")
    
    return "\n".join(lines)


def format_actual_firing_rate_stats(syn_df):
    """
    Format actual firing rate statistics from synapses.csv.
    
    Refactored from write_actual_firing_rate_stats() in AA_pre_sim.py.
    Now returns formatted string instead of printing directly.
    
    Args:
        syn_df: DataFrame with synapse data from synapses.csv
    
    Returns:
        Formatted string with actual firing rate statistics
    """
    lines = []
    lines.append("Actual firing rate statistics from synapses:")
    exc_syns = syn_df[syn_df['name'] == 'exc']
    inh_syns = syn_df[syn_df['name'] == 'inh']
    
    if not exc_syns.empty and 'pc_mean_firing_rate' in exc_syns.columns:
        lines.append("\nExcitatory synapses by input_source:")
        for input_source in exc_syns['input_source'].unique():
            source_syns = exc_syns[exc_syns['input_source'] == input_source]
            fr_data = source_syns['pc_mean_firing_rate']
            lines.append(f"  {input_source}:")
            lines.append(f"    mean={fr_data.mean():.4f} Hz, std={fr_data.std():.4f} Hz")
            lines.append(f"    min={fr_data.min():.4f} Hz, max={fr_data.max():.4f} Hz")
            lines.append(f"    median={fr_data.median():.4f} Hz")
    
    if not inh_syns.empty and 'pc_mean_firing_rate' in inh_syns.columns:
        lines.append("\nInhibitory synapses by input_source:")
        for input_source in inh_syns['input_source'].unique():
            source_syns = inh_syns[inh_syns['input_source'] == input_source]
            fr_data = source_syns['pc_mean_firing_rate']
            lines.append(f"  {input_source}:")
            lines.append(f"    mean={fr_data.mean():.4f} Hz, std={fr_data.std():.4f} Hz")
            lines.append(f"    min={fr_data.min():.4f} Hz, max={fr_data.max():.4f} Hz")
            lines.append(f"    median={fr_data.median():.4f} Hz")
    
    return "\n".join(lines)


def write_synapse_info(sim_dir, parameters):
    """
    Write comprehensive synapse information to synapse_info.txt for a single simulation.
    
    Refactored from write_synapse_info_txt_file() in AA_pre_sim.py.
    Now handles a single sim_dir instead of iterating over all sims.
    
    Args:
        sim_dir: Path to single simulation directory
        parameters: SimulationParameters object
    """
    # Try to load synapses.csv to get counts per input_source
    syn_csv_path = os.path.join(sim_dir, "synapses.csv")
    if os.path.exists(syn_csv_path):
        syn_df = pd.read_csv(syn_csv_path)
        syn_counts = syn_df["input_source"].value_counts().to_dict()
    else:
        syn_counts = {}
        syn_df = None

    out_path = os.path.join(sim_dir, "synapse_info.txt")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write(f"Simulation: {os.path.basename(sim_dir)}\n\n")
        
        # Write synapse densities
        out_f.write(format_synapse_densities(parameters, syn_counts))
        out_f.write("\n")
        
        # Write weight distributions
        out_f.write(format_weight_distributions(parameters))
        out_f.write("\n")
        
        # Write firing rate distributions
        out_f.write(format_firing_rate_distributions(parameters))
        out_f.write("\n")
        
        # Write actual firing rate statistics from synapses.csv
        if syn_df is not None:
            out_f.write(format_actual_firing_rate_stats(syn_df))
            out_f.write("\n")
