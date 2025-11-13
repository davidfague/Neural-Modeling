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
from scripts.configure_sim_params import configure_sim_params
from Modules.simulation_slurm import Simulator
from Modules.segments_file import generate_segments_csv
from Modules.synapses_file import (
    PreSimSynapseGenerator,
    replace_N_synapses,
    update_spike_trains_for_sim
)
from Modules.synapse_analysis import SynapseAnalyzer
from Modules.plot_morphology import plot_morphology_flex, plot_reduced_morphology

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

def write_synapse_densities_to_txt(sims_dir: str) -> None:
    # iterate over subfolders only
    for sim_dir in [d for d in os.listdir(sims_dir) if os.path.isdir(os.path.join(sims_dir, d))]:
        pkl_path = os.path.join(sims_dir, sim_dir, "parameters.pickle")
        if not os.path.exists(pkl_path):
            continue  # skip if this sim doesn't have parameters

        with open(pkl_path, "rb") as f:
            params = pickle.load(f)

        out_path = os.path.join(sims_dir, sim_dir, "synapse_info.txt")
        with open(out_path, "w", encoding="utf-8") as out_f:
            p = lambda *a, **k: print(*a, file=out_f, **k)

            p("Exc synapse densities:")
            for input_source, props in params.exc_syn_properties.items():
                p(f"{input_source}: {round(props.get('syn_density'), 4)}")

            p("")  # blank line
            p("Inh synapse densities:")
            for input_source, props in params.inh_syn_properties.items():
                p(f"{input_source}: {round(props.get('syn_density'), 4)}")

            p("")  # blank line
            p("Exc synapse weights:")
            for input_source, props in params.exc_syn_properties.items():
                init_wt_dist = props.get("initial_weight_distribution", {}) or {}
                wt_params = init_wt_dist.get("params", {}) or {}
                p(f"{input_source}:  {wt_params}")

            p("")  # blank line
            p("Inh synapse weights:")
            for input_source, props in params.inh_syn_properties.items():
                init_wt_dist = props.get("initial_weight_distribution", {}) or {}
                wt_params = init_wt_dist.get("params", {}) or {}
                p(f"{input_source}:  {wt_params}")

def plot_morphology(sim_dir):
    ### each sec_type
    seg_data = pd.read_csv(os.path.join(sim_dir, "segment_data.csv"))
    parameters = analysis.DataReader.load_parameters(sim_dir) # load parameters
    figs, axs = plot_morphology_flex(
            seg_data,
            option='each_sec_type',
            parameters=parameters,
            out_dir=os.path.join(sim_dir, "morphology"),
            figsize=(10,6),
            show=True,          # set True if you want interactive windows
            save=True,           # saves PNGs named <sec_type>.png
            color='red',         # all types get same color; can change
        )


def run_pre_sim():
    # ---------------- load config ----------------
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

    # All parameter sets are now ready!
    simulator = Simulator(
        sim_set_title=SIM_SET_TITLE,
        sim_titles=all_sim_titles,
        parameter_sets=all_parameter_sets,
        sims_root=SIMULATIONS_FOLDER
    )
    simulator.create_simulation_folders()

    write_spike_train_modes(simulator.sims_dir, inh_syn_properties, exc_syn_properties)

    # Run generators across all sims
    log(f"\n[AA_pre_sim] Generating segments CSV on all sims in: {simulator.sims_dir}\n")
    simulator.run_on_all_sims_parallel(simulator.sims_dir, generate_segments_csv)
    log(f"\n[AA_pre_sim] Plotting morphology on all sims in: {simulator.sims_dir}\n")
    plot_morphology(sim_dir)
    log(f"\n[AA_pre_sim] Running synapse generation on all sims in: {simulator.sims_dir}\n")
    simulator.run_on_all_sims_parallel(simulator.sims_dir, use_pssg)

    # Build a list of full sim paths
    sim_dirs = [
        os.path.join(simulator.sims_dir, d)
        for d in os.listdir(simulator.sims_dir)
        if os.path.isdir(os.path.join(simulator.sims_dir, d))
    ]

    # Update spike trains for each sim
    if N_bg_synapses > 0:
        log(f"\n[AA_pre_sim] Setting {N_bg_synapses} synapses to background on all sims in: {simulator.sims_dir}\n")
        simulator.run_on_all_sims_parallel(
            simulator.sims_dir,
            replace_N_synapses,
            process_fns_args=(N_bg_synapses,)
        )
        log(f"\n[AA_pre_sim] Setting background spike trains on all sims in: {simulator.sims_dir}\n")
        for sim_dir in sim_dirs:
            update_spike_trains_for_sim(sim_dir, inh_bg_rate, exc_bg_rate)

    # Analyze & plot
    synapse_analyzers = [SynapseAnalyzer(sim_dir) for sim_dir in sim_dirs]

    log(f"\n[AA_pre_sim] Generating synapse cluster plots on all sims in: {simulator.sims_dir}\n")
    for synapse_analyzer in synapse_analyzers:
        synapse_analyzer.plot_all_synapse_clusters(
            plot_both_together=True,
            plot_each_type_separately=True
        )

    log(f"\n[AA_pre_sim] Generating spike raster plots on all sims in: {simulator.sims_dir}\n")
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

    write_synapse_densities_to_txt(simulator.sims_dir)

    # h.load_file('stdrun.hoc')
    # if not os.path.exists('x86_64'):
    #     h.nrn_load_dll('x86_64/.libs/libnrnmech.so')

    return simulator

if __name__ == "__main__":
    run_pre_sim()