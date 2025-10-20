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


# ---------------- basics ----------------
SIMULATIONS_FOLDER = "/home/drfrbc/Neural-Modeling/simulations"  # added leading slash

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

def write_spike_train_modes(inh_syn_properties, exc_syn_properties):
    # write spike_train_mode configurations to a text file in the simulation set folder
    with open(os.path.join(simulator.sims_dir, 'spike_train_modes.txt'), 'w') as f:
        f.write("Inhibitory Synapse Spike Train Modes:\n")
        for input_source, props in inh_syn_properties.items():
            f.write(f"{input_source}: {props['spike_train_mode']}\n")
        f.write("\nExcitatory Synapse Spike Train Modes:\n")
        for input_source, props in exc_syn_properties.items():
            f.write(f"{input_source}: {props['spike_train_mode']}\n")


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

write_spike_train_modes(inh_syn_properties, exc_syn_properties)

# Run generators across all sims
log(f"\n[AA_pre_sim] Generating segments CSV on all sims in: {simulator.sims_dir}\n", flush=True)
simulator.run_on_all_sims_parallel(simulator.sims_dir, generate_segments_csv)
log(f"\n[AA_pre_sim] Running synapse generation on all sims in: {simulator.sims_dir}\n", flush=True)
simulator.run_on_all_sims_parallel(simulator.sims_dir, use_pssg)

# NOTE: tuple fix for single arg
log(f"\n[AA_pre_sim] Setting N synapses to background on all sims in: {simulator.sims_dir}\n", flush=True)
simulator.run_on_all_sims_parallel(
    simulator.sims_dir,
    replace_N_synapses,
    process_fns_args=(N_bg_synapses,)
)

# Build a list of full sim paths
sim_dirs = [
    os.path.join(simulator.sims_dir, d)
    for d in os.listdir(simulator.sims_dir)
    if os.path.isdir(os.path.join(simulator.sims_dir, d))
]

# Update spike trains for each sim
log(f"\n[AA_pre_sim] Setting background spike trains on all sims in: {simulator.sims_dir}\n", flush=True)
for sim_dir in sim_dirs:
    update_spike_trains_for_sim(sim_dir, inh_bg_rate, exc_bg_rate)

# Analyze & plot
synapse_analyzers = [SynapseAnalyzer(sim_dir) for sim_dir in sim_dirs]

log(f"\n[AA_pre_sim] Generating synapse cluster plots on all sims in: {simulator.sims_dir}\n", flush=True)
for synapse_analyzer in synapse_analyzers:
    synapse_analyzer.plot_all_synapse_clusters(
        plot_both_together=True,
        plot_each_type_separately=True
    )

log(f"\n[AA_pre_sim] Generating spike raster plots on all sims in: {simulator.sims_dir}\n", flush=True)
# Generate a spike raster plot for excitatory synapses
for synapse_analyzer in synapse_analyzers:
    synapse_analyzer.plot_spike_raster(
        synapse_types=['exc'],
        time_window=(49000, 50000),  # last second in a 50s sim
        save_path=os.path.join(synapse_analyzer.sim_dir, 'exc_spike_raster.png'),
        title="Excitatory Synapse Spike Raster"
    )
# Generate a spike raster plot for inhibitory synapses
for synapse_analyzer in synapse_analyzers:
    synapse_analyzer.plot_spike_raster(
        synapse_types=['inh'],
        time_window=(49000, 50000),
        save_path=os.path.join(synapse_analyzer.sim_dir, 'inh_spike_raster.png'),
        title="Inhibitory Synapse Spike Raster"
    )

# h.load_file('stdrun.hoc')
# if not os.path.exists('x86_64'):
#     h.nrn_load_dll('x86_64/.libs/libnrnmech.so')
