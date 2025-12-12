#!/usr/bin/env python3
# AA_pre_sim.py

import sys
from neuron import h  # kept from your snippet (ok if unused)
import os
sys.path.append('..')
sys.path.append('../Modules')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
import copy
import Modules.analysis as analysis

from Modules.constants import HayParameters
from configure_sim_params import configure_sim_params
from Modules.simulation_slurm import Simulator
from Modules.segments_file import generate_segments_csv
from Modules.synapses_file import (
    PreSimSynapseGenerator,
    replace_N_synapses,
    update_spike_trains_for_sim
)
from Modules.synapse_analysis import SynapseAnalyzer
from Modules.plot_morphology import plot_morphology_flex, plot_reduced_morphology
from Modules.logger import Logger

# ---------------- basics ----------------
SIMULATIONS_FOLDER = "../simulations"  # added leading slash

def _now():
    # Local time; change to .utcnow() if you prefer UTC
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

def use_pssg(sim_dir):
    pssg = PreSimSynapseGenerator(sim_dir)
    pssg.generate_synapse_locations()
    pssg.synapses.to_csv(os.path.join(sim_dir, "synapses.csv"), index=False)
    pssg.generate_spike_trains_for_synapses()  # adds column spike_train according to parameters.

def write_spike_train_modes(sims_dir, inh_syn_properties, exc_syn_properties):
    # write spike_train_mode configurations to a text file in the simulation set folder
    with open(os.path.join(sims_dir, 'spike_train_modes.txt'), 'w') as f:
        f.write("Inhibitory Synapse Spike Train Modes:\n")
        for input_source, props in inh_syn_properties.items():
            f.write(f"{input_source}: {props['spike_train_mode']}\n")
        f.write("\nExcitatory Synapse Spike Train Modes:\n")
        for input_source, props in exc_syn_properties.items():
            f.write(f"{input_source}: {props['spike_train_mode']}\n")

def write_synapse_densities(p, params, syn_counts):
    """Write synapse density information for exc and inh synapses."""
    p("Exc synapse densities and counts:")
    for input_source, props in params.exc_syn_properties.items():
        dens = props.get("syn_density")
        dens_str = "None" if dens is None else round(dens, 4)
        n_syn = syn_counts.get(input_source, 0)
        p(f"{input_source}: density={dens_str}, n_synapses={n_syn}")
    p("")
    
    p("Inh synapse densities and counts:")
    for input_source, props in params.inh_syn_properties.items():
        dens = props.get("syn_density")
        dens_str = "None" if dens is None else round(dens, 4)
        n_syn = syn_counts.get(input_source, 0)
        p(f"{input_source}: density={dens_str}, n_synapses={n_syn}")
    p("")

def write_weight_distributions(p, params):
    """Write weight distribution information for exc and inh synapses."""
    p("Exc synapse weight distributions:")
    for input_source, props in params.exc_syn_properties.items():
        init_wt_dist = props.get("initial_weight_distribution", {}) or {}
        wt_func = init_wt_dist.get("function")
        wt_func_name = wt_func.__name__ if wt_func else "None"
        wt_params = init_wt_dist.get("params", {}) or {}
        p(f"{input_source}: {wt_func_name}: {wt_params}")
    p("")
    
    p("Inh synapse weight distributions:")
    for input_source, props in params.inh_syn_properties.items():
        init_wt_dist = props.get("initial_weight_distribution", {}) or {}
        wt_func = init_wt_dist.get("function")
        wt_func_name = wt_func.__name__ if wt_func else "None"
        wt_params = init_wt_dist.get("params", {}) or {}
        p(f"{input_source}: {wt_func_name}: {wt_params}")
    p("")

def write_firing_rate_distributions(p, params):
    """Write firing rate distribution information for exc and inh synapses."""
    p("Exc synapse firing rate distributions:")
    for input_source, props in params.exc_syn_properties.items():
        fr_dist = props.get("mean_firing_rate_distribution", {}) or {}
        fr_func = fr_dist.get("function")
        fr_func_name = fr_func.__name__ if fr_func else "None"
        fr_params = fr_dist.get("params", {}) or {}
        fr_shift = props.get("fr_shift", 0)
        p(f"{input_source}: {fr_func_name}: {fr_params} {f'fr_shift: {fr_shift}' if fr_shift != 0 else ''}")
    p("")
    
    p("Inh synapse firing rate distributions:")
    for input_source, props in params.inh_syn_properties.items():
        fr_dist = props.get("mean_firing_rate_distribution", {}) or {}
        fr_func = fr_dist.get("function")
        fr_func_name = fr_func.__name__ if fr_func else "None"
        fr_params = fr_dist.get("params", {}) or {}
        fr_shift = props.get("fr_shift", 0)
        p(f"{input_source}: {fr_func_name}: {fr_params} {f'fr_shift: {fr_shift}' if fr_shift != 0 else ''}")
    p("")

def write_actual_firing_rate_stats(p, syn_df):
    """Write actual firing rate statistics from synapses.csv."""
    p("Actual firing rate statistics from synapses:")
    exc_syns = syn_df[syn_df['name'] == 'exc']
    inh_syns = syn_df[syn_df['name'] == 'inh']
    
    if not exc_syns.empty and 'pc_mean_firing_rate' in exc_syns.columns:
        p("\nExcitatory synapses by input_source:")
        for input_source in exc_syns['input_source'].unique():
            source_syns = exc_syns[exc_syns['input_source'] == input_source]
            fr_data = source_syns['pc_mean_firing_rate']
            p(f"  {input_source}:")
            p(f"    mean={fr_data.mean():.4f} Hz, std={fr_data.std():.4f} Hz")
            p(f"    min={fr_data.min():.4f} Hz, max={fr_data.max():.4f} Hz")
            p(f"    median={fr_data.median():.4f} Hz")
    
    if not inh_syns.empty and 'pc_mean_firing_rate' in inh_syns.columns:
        p("\nInhibitory synapses by input_source:")
        for input_source in inh_syns['input_source'].unique():
            source_syns = inh_syns[inh_syns['input_source'] == input_source]
            fr_data = source_syns['pc_mean_firing_rate']
            p(f"  {input_source}:")
            p(f"    mean={fr_data.mean():.4f} Hz, std={fr_data.std():.4f} Hz")
            p(f"    min={fr_data.min():.4f} Hz, max={fr_data.max():.4f} Hz")
            p(f"    median={fr_data.median():.4f} Hz")

def write_synapse_info_txt_file(sims_dir: str) -> None:
    """Write comprehensive synapse information to synapse_info.txt for each simulation."""
    # iterate over subfolders only
    for sim_dir in [d for d in os.listdir(sims_dir) if os.path.isdir(os.path.join(sims_dir, d))]:
        full_sim_dir = os.path.join(sims_dir, sim_dir)

        pkl_path = os.path.join(full_sim_dir, "parameters.pickle")
        if not os.path.exists(pkl_path):
            continue  # skip if this sim doesn't have parameters

        # try to load synapses.csv to get counts per input_source
        syn_csv_path = os.path.join(full_sim_dir, "synapses.csv")
        if os.path.exists(syn_csv_path):
            syn_df = pd.read_csv(syn_csv_path)
            # value_counts gives a dict: input_source -> n_synapses
            syn_counts = syn_df["input_source"].value_counts().to_dict()
        else:
            syn_counts = {}
            syn_df = None

        with open(pkl_path, "rb") as f:
            params = pickle.load(f)

        out_path = os.path.join(full_sim_dir, "synapse_info.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            p = lambda *a, **k: print(*a, file=out_f, **k)

            p(f"Simulation: {sim_dir}")
            p("")
            
            # Write synapse densities
            write_synapse_densities(p, params, syn_counts)
            
            # Write weight distributions
            write_weight_distributions(p, params)
            
            # Write firing rate distributions
            write_firing_rate_distributions(p, params)
            
            # Write actual firing rate statistics from synapses.csv
            if syn_df is not None:
                write_actual_firing_rate_stats(p, syn_df)
                
def plot_morphology(sim_dir):
    ### each sec_type
    seg_data = pd.read_csv(os.path.join(sim_dir, "segment_data.csv"))
    parameters = analysis.DataReader.load_parameters(sim_dir) # load parameters
    logger = Logger(sim_dir)
    figs, axs = plot_morphology_flex(
            seg_data,
            option='each_sec_type',
            parameters=parameters,
            out_dir=os.path.join(sim_dir, "morphology"),
            figsize=(10,6),
            show=True,          # set True if you want interactive windows
            save=True,           # saves PNGs named <sec_type>.png
            color='red',         # all types get same color; can change
            logger=logger,
        )

def plot_firing_rate_distributions(simulator, synapse_analyzers, logger):
    log(f"\n[AA_pre_sim] Generating firing rate distribution plots on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("firing_rate_distribution_plots")
    # Generate firing rate distribution plots for exc and inh synapses
    for synapse_analyzer in synapse_analyzers:
        distributions_folder = os.path.join(synapse_analyzer.sim_dir, "firing_rate_distributions")
        os.makedirs(distributions_folder, exist_ok=True)
        
        # Overall plots by synapse type
        synapse_analyzer.plot_firing_rate_distribution(
            synapse_type='exc',
            save_path=os.path.join(distributions_folder, 'exc_firing_rate_dist.png'),
            show=False
        )
        
        synapse_analyzer.plot_firing_rate_distribution(
            synapse_type='inh',
            save_path=os.path.join(distributions_folder, 'inh_firing_rate_dist.png'),
            show=False
        )
        
        synapse_analyzer.plot_firing_rate_distribution(
            save_path=os.path.join(distributions_folder, 'all_firing_rate_dist.png'),
            show=False
        )
        
        for input_source in synapse_analyzer.synapses['input_source'].unique():
            synapse_type_prefix = 'exc' if synapse_analyzer.synapses[synapse_analyzer.synapses['input_source'] == input_source]['name'].iloc[0].startswith('exc') else 'inh'
            synapse_analyzer.plot_firing_rate_distribution(
                input_source=input_source,
                save_path=os.path.join(distributions_folder, f'{synapse_type_prefix}_{input_source}_firing_rate_dist.png'),
                show=False
            )
    logger.log_runtime("AA_pre_sim", "firing_rate_distribution_plots", timer_name="firing_rate_distribution_plots")

def plot_spike_rasters(simulator, synapse_analyzers, logger):
    log(f"\n[AA_pre_sim] Generating spike raster plots on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("spike_raster_plots")
    # Generate a spike raster plot for  and inhibitory synapses
    for synapse_analyzer in synapse_analyzers:
        parameters = analysis.DataReader.load_parameters(synapse_analyzer.sim_dir)
        spike_rasters_folder = os.path.join(synapse_analyzer.sim_dir, "spike_rasters")
        os.makedirs(spike_rasters_folder, exist_ok=True)
        synapse_analyzer.plot_spike_raster(
            synapse_types=['exc'],
            time_window=(parameters.h_tstop-1000, parameters.h_tstop),  # last second in sim
            save_path=os.path.join(spike_rasters_folder, 'exc_spike_raster_end.png'),
            title="Excitatory Synapse Spike Raster"
        )
        synapse_analyzer.plot_spike_raster(
            synapse_types=['inh'],
            time_window=(parameters.h_tstop-1000, parameters.h_tstop),
            save_path=os.path.join(spike_rasters_folder, 'inh_spike_raster_end.png'),
            title="Inhibitory Synapse Spike Raster"
        )
        synapse_analyzer.plot_spike_raster(
            synapse_types=['exc'],
            time_window=(0, 1000),  # first second in sim
            save_path=os.path.join(spike_rasters_folder, 'exc_spike_raster_start.png'),
            title="Excitatory Synapse Spike Raster"
        )
        synapse_analyzer.plot_spike_raster(
            synapse_types=['inh'],
            time_window=(0, 1000),
            save_path=os.path.join(spike_rasters_folder, 'inh_spike_raster_start.png'),
            title="Inhibitory Synapse Spike Raster"
        )
    logger.log_runtime("AA_pre_sim", "spike_raster_plots", timer_name="spike_raster_plots")


def run_pre_sim():
    # Create a temporary logger for initial setup
    temp_logger = Logger()
    temp_logger.start_timer("total_pre_sim")
    
    # ---------------- load config ----------------
    temp_logger.start_timer("configure_sim_params")
    (
        all_parameter_sets,
        all_sim_titles,
        inh_bg_rate,
        exc_bg_rate,
        N_bg_synapses,
        SIM_SET_TITLE,
        inh_syn_properties,
        exc_syn_properties,
    ) = configure_sim_params(parameters_pkl_path=None)  # optionally set parameters_pkl_path to an existing simulation's parameters.pkl to override defaults and base new sims on the existing sim.
    temp_logger.log_runtime("AA_pre_sim", "configure_sim_params", timer_name="configure_sim_params")

    # All parameter sets are now ready!
    temp_logger.start_timer("create_simulation_folders")
    simulator = Simulator(
        sim_set_title=SIM_SET_TITLE,
        sim_titles=all_sim_titles,
        parameter_sets=all_parameter_sets,
        sims_root=SIMULATIONS_FOLDER
    )
    simulator.create_simulation_folders()
    temp_logger.log_runtime("AA_pre_sim", "create_simulation_folders", timer_name="create_simulation_folders")
    
    # Now create a proper logger with the sims_dir
    logger = Logger(simulator.sims_dir)
    # Transfer the total timer to the new logger
    logger._timers = temp_logger._timers

    # Build a list of full sim paths
    sim_dirs = [
        os.path.join(simulator.sims_dir, d)
        for d in os.listdir(simulator.sims_dir)
        if os.path.isdir(os.path.join(simulator.sims_dir, d))
    ]

    write_spike_train_modes(simulator.sims_dir, inh_syn_properties, exc_syn_properties)

    # generate segments CSV
    log(f"\n[AA_pre_sim] Generating segments CSV on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("generate_segments_csv")
    simulator.run_on_all_sims_parallel(simulator.sims_dir, generate_segments_csv)
    logger.log_runtime("AA_pre_sim", "generate_segments_csv", timer_name="generate_segments_csv")

    # plot morphology
    log(f"\n[AA_pre_sim] Plotting morphology on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("plot_morphology")
    simulator.run_on_all_sims_parallel(simulator.sims_dir, plot_morphology)
    logger.log_runtime("AA_pre_sim", "plot_morphology", timer_name="plot_morphology")
    
    # generate synapses
    log(f"\n[AA_pre_sim] Running synapse generation on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("synapse_generation")
    simulator.run_on_all_sims_parallel(simulator.sims_dir, use_pssg)
    logger.log_runtime("AA_pre_sim", "synapse_generation", timer_name="synapse_generation")

    # Update background spike trains
    if N_bg_synapses > 0:
        log(f"\n[AA_pre_sim] Setting {N_bg_synapses} synapses to background on all sims in: {simulator.sims_dir}\n")
        logger.start_timer("replace_background_synapses")
        simulator.run_on_all_sims_parallel(
            simulator.sims_dir,
            replace_N_synapses,
            process_fns_args=(N_bg_synapses,)
        )
        logger.log_runtime("AA_pre_sim", "replace_background_synapses", timer_name="replace_background_synapses")
        
        log(f"\n[AA_pre_sim] Setting background spike trains on all sims in: {simulator.sims_dir}\n")
        logger.start_timer("update_background_spike_trains")
        for sim_dir in sim_dirs:
            update_spike_trains_for_sim(sim_dir, inh_bg_rate, exc_bg_rate)
        logger.log_runtime("AA_pre_sim", "update_background_spike_trains", timer_name="update_background_spike_trains")

    # create synapse analyzers for plotting synapse clusters, spikes rasters, firing rate distributions
    synapse_analyzers = [SynapseAnalyzer(sim_dir) for sim_dir in sim_dirs]

    # synapse cluster plots
    log(f"\n[AA_pre_sim] Generating synapse cluster plots on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("synapse_cluster_plots")
    for synapse_analyzer in synapse_analyzers:
        synapse_analyzer.plot_all_synapse_clusters(
            plot_both_together=True,
            plot_each_type_separately=False  # Skip individual input_source plots to save time
        )
    logger.log_runtime("AA_pre_sim", "synapse_cluster_plots", timer_name="synapse_cluster_plots")

    # spike rasters
    plot_spike_rasters(simulator, synapse_analyzers, logger)

    # firing rate distributions (temporarily disabled)
    # plot_firing_rate_distributions(simulator, synapse_analyzers, logger)

    # write synapse info
    logger.start_timer("write_synapse_info")
    write_synapse_info_txt_file(simulator.sims_dir)
    logger.log_runtime("AA_pre_sim", "write_synapse_info", timer_name="write_synapse_info")

    logger.log_runtime("AA_pre_sim", "total_pre_sim", timer_name="total_pre_sim") # log total runtime

    # h.load_file('stdrun.hoc')
    # if not os.path.exists('x86_64'):
    #     h.nrn_load_dll('x86_64/.libs/libnrnmech.so')

    return simulator

if __name__ == "__main__":
    run_pre_sim()