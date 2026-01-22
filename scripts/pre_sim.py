#!/usr/bin/env python3
# AA_pre_sim.py

import sys
import os
from functools import partial

sys.path.append('..')
sys.path.append('../Modules')

import Modules.post_sim.analysis as analysis

from configure_sim_params import configure_sim_params
from Modules.sim.simulation_slurm import Simulator
from Modules.pre_sim.segments_file import generate_segments_csv
from Modules.pre_sim.synapses_file import (
    replace_N_synapses,
    update_spike_trains_for_sim
)
from Modules.pre_sim.synapse_analysis import SynapseAnalyzer
from Modules.logger import Logger

# Import new refactored modules
from Modules.pre_sim.pre_sim_funcs import (
    generate_synapses_for_sim,
    plot_morphology_for_sim,
    plot_firing_rate_distributions_for_sim,
    plot_spike_rasters_for_sim,
    log  # Import helper functions too
)
from Modules.pre_sim.logging_details_pre_sim import (
    write_spike_train_config,
    write_synapse_info
)

# ---------------- basics ----------------
SIMULATIONS_FOLDER = "../simulations"


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

    # Determine optimal worker count (consistent with run_pipeline.py)
    n_simulations = len(sim_dirs)
    MAX_WORKERS = min(9, n_simulations)
    log(f"[AA_pre_sim] Using {MAX_WORKERS} workers for {n_simulations} simulations")

    write_spike_train_config(simulator.sims_dir, inh_syn_properties, exc_syn_properties)

    # generate segments CSV with morphology caching
    log(f"\n[AA_pre_sim] Generating segments CSV on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("generate_segments_csv")
    
    # Use functools.partial to bind sims_dir for morphology caching (picklable)
    generate_segments_with_cache = partial(generate_segments_csv, sims_dir=simulator.sims_dir)
    
    simulator.run_on_all_sims_parallel(simulator.sims_dir, generate_segments_with_cache, max_workers=MAX_WORKERS)
    logger.log_runtime("AA_pre_sim", "generate_segments_csv", timer_name="generate_segments_csv")

    # plot morphology
    # log(f"\n[AA_pre_sim] Plotting morphology on all sims in: {simulator.sims_dir}\n")
    # logger.start_timer("plot_morphology")
    # simulator.run_on_all_sims_parallel(simulator.sims_dir, plot_morphology_for_sim, max_workers=MAX_WORKERS)
    # logger.log_runtime("AA_pre_sim", "plot_morphology", timer_name="plot_morphology")
    
    # generate synapses
    log(f"\n[AA_pre_sim] Running synapse generation on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("synapse_generation")
    simulator.run_on_all_sims_parallel(simulator.sims_dir, generate_synapses_for_sim, max_workers=MAX_WORKERS)
    logger.log_runtime("AA_pre_sim", "synapse_generation", timer_name="synapse_generation")

    # Update background spike trains
    if N_bg_synapses > 0:
        log(f"\n[AA_pre_sim] Setting {N_bg_synapses} synapses to background on all sims in: {simulator.sims_dir}\n")
        logger.start_timer("replace_background_synapses")
        simulator.run_on_all_sims_parallel(
            simulator.sims_dir,
            replace_N_synapses,
            process_fns_args=(N_bg_synapses,),
            max_workers=MAX_WORKERS
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
    plot_spike_rasters_for_sim(simulator, synapse_analyzers, logger)

    # firing rate distributions (temporarily disabled)
    # plot_firing_rate_distributions_for_sim(simulator, synapse_analyzers, logger)

    # write synapse info
    logger.start_timer("write_synapse_info")
    for sim_dir in sim_dirs:
        params = analysis.DataReader.load_parameters(sim_dir)
        write_synapse_info(sim_dir, params)
    logger.log_runtime("AA_pre_sim", "write_synapse_info", timer_name="write_synapse_info")

    logger.log_runtime("AA_pre_sim", "total_pre_sim", timer_name="total_pre_sim") # log total runtime

    # h.load_file('stdrun.hoc')
    # if not os.path.exists('x86_64'):
    #     h.nrn_load_dll('x86_64/.libs/libnrnmech.so')

    return simulator

if __name__ == "__main__":
    run_pre_sim()