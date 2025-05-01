#!/usr/bin/env python3

"""
Automated Synapse Tuning (AST) - Parallel Version

This script implements parallel simulation and optimization for synapse tuning,
focusing on location-dependent PSC tuning.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution
import os
import multiprocessing as mp
import sys
import neuron
from neuron import h
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
modules_dir = os.path.join(root_dir, 'Modules')
modfiles_dir = os.path.join(root_dir, 'modfiles')
sys.path.append(modules_dir)

# Setup bmtool
bmtool_dir = os.path.join(current_dir, 'bmtool')
if not os.path.exists(bmtool_dir):
    print("Cloning bmtool repository...")
    os.system(f"git clone https://github.com/davidfague/bmtool.git {bmtool_dir}")

# Add bmtool to path
sys.path.append(bmtool_dir)

print(f"Current directory: {current_dir}")
print(f"Root directory: {root_dir}")
print(f"Modules directory: {modules_dir}")
print(f"Bmtool directory: {bmtool_dir}")
print(f"Modfiles directory: {modfiles_dir}")
print(f"Current sys.path: {sys.path}")

try:
    from general_settings_for_AST import (
        general_settings,
        conn_type_settings,
        tuner_configs,
        InitializeSysnapseTuner,
        target_metrics,
        location_types_by_synapse_type,
        distributions_to_test,
        log_norm_dist,
        norm_dist,
        load_hay_cell
    )
except ImportError as e:
    print(f"Error importing general_settings_for_AST: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Looking for module in: {modules_dir}")
    sys.exit(1)

# Global settings
TOTAL_SAMPLES_PER_WEIGHT_DISTRIBUTION = 100
USE_HAY_CELL = True

def setup_cell_and_mechanisms():
    """Initialize the cell model and load NEURON mechanisms."""
    # Load NEURON mechanisms first
    if os.path.isdir(os.path.join(modfiles_dir, 'x86_64')):
        os.system(f"rm -rf {os.path.join(modfiles_dir, 'x86_64')}")
    
    if not os.path.isdir(os.path.join(modfiles_dir, 'x86_64')):
        os.chdir(modfiles_dir)
        os.system("nrnivmodl > /dev/null 2>&1")
        os.chdir(current_dir)
    
    neuron.load_mechanisms(modfiles_dir)
    
    # Load required NEURON libraries
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")
    h.load_file("nrngui.hoc")
    
    # Load template file
    template_path = os.path.join(root_dir, 'cells', 'templates', 'L5PCtemplateMediumRes.hoc')
    if not os.path.exists(template_path):
        print(f"Error: Template file not found at {template_path}")
        sys.exit(1)
    h.load_file(template_path)
    
    if USE_HAY_CELL:
        template_arg = load_hay_cell(conn_type_settings)
    else:
        template_arg = None
    
    return template_arg

def get_sec_ids_from_type(section_type):
    """Get section IDs based on section type."""
    cell = h.L5PCtemplate("../../../../cells/templates/cell1.asc")

    if section_type == 'distal_apic':
        sec_ids_to_use = [idx for idx, sec in enumerate(cell.all) 
                         if (sec in cell.apic) and (h.distance(cell.soma[0](0.5), sec(0.5)) > 100)]
    elif section_type == 'distal_basal':
        sec_ids_to_use = [idx for idx, sec in enumerate(cell.all) 
                         if (sec in cell.dend) and (h.distance(cell.soma[0](0.5), sec(0.5)) > 100)]
    elif section_type == 'perisomatic':
        sec_ids_to_use = [idx for idx, sec in enumerate(cell.all) 
                         if ((h.distance(cell.soma[0](0.5), sec(0.5)) < 100) and (sec not in list(cell.axon)))]
    else:
        del cell
        raise NotImplementedError(f"{section_type} not implemented for get_sec_ids_from_type")

    del cell
    return sec_ids_to_use

def get_segments(synapse_tuner_obj, location_type):
    """Get segments and their probabilities for a given location type."""
    segments = [seg for sec in synapse_tuner_obj.cell.all for seg in sec]
    sec_ids_to_use = get_sec_ids_from_type(location_type)
    possible_segments = [seg for sec_id in sec_ids_to_use 
                        for seg in list(synapse_tuner_obj.cell.all)[sec_id]]
    seg_probs = [(seg.sec.L / seg.sec.nseg) for seg in possible_segments]
    return segments, possible_segments, seg_probs

def move_synapse_to_new_location(synapse_tuner_obj, possible_segments, seg_probs, all_segments):
    """Move synapse to a new random location based on segment probabilities."""
    seg_to_place_syn_on = np.random.choice(
        possible_segments, 1, True, seg_probs / np.sum(seg_probs))[0]
    synapse_tuner_obj.syn.loc(seg_to_place_syn_on)
    segment_index = all_segments.index(seg_to_place_syn_on)
    return segment_index

def change_synapse_weight(synapse_tuner_obj, distributions_to_test, synapse_type, 
                         location_type, use_norm_dist=False):
    """Change synapse weight based on distribution parameters."""
    if use_norm_dist:
        new_weight = norm_dist(
            distributions_to_test[synapse_type][location_type]['mean'],
            distributions_to_test[synapse_type][location_type]['std'],
            1,
            (0, 10*distributions_to_test[synapse_type][location_type]['mean'])
        )
    else:
        exc_mean = (np.log(0.45) - 0.5 * np.log((0.35/0.45)**2+1))
        exc_std = np.sqrt(np.log((0.35/0.45)**2 + 1))
        exc_clip = (1e-15, 5)
        new_weight = log_norm_dist(
            exc_mean,
            exc_std,
            1,
            exc_clip,
            distributions_to_test[synapse_type][location_type]['exc_scalar']
        )
    synapse_tuner_obj.syn.initW = new_weight
    return new_weight

def simulate_PSC(synapse_type, tuner_configs, location_type, distributions_to_test, use_norm_dist):
    """Run a single PSC simulation and return results."""
    synapse_tuner_obj = InitializeSysnapseTuner(
        template_arg=template_arg, 
        **tuner_configs[True][synapse_type]
    )
    
    all_segments, possible_segments, seg_probs = get_segments(
        synapse_tuner_obj, 
        location_type
    )
    
    weight = change_synapse_weight(
        synapse_tuner_obj,
        distributions_to_test,
        'exc' if 'exc' in synapse_type.lower() else 'inh',
        location_type,
        use_norm_dist
    )
    
    loc = move_synapse_to_new_location(
        synapse_tuner_obj, 
        possible_segments, 
        seg_probs, 
        all_segments
    )
    
    PSC_mag = max(abs(synapse_tuner_obj.SingleEvent(plot_and_print=False)))
    return PSC_mag, weight, loc

def run_parallel_simulations(synapse_type, location_type, total_samples=100, use_norm_dist=False):
    """Run parallel simulations and collect results."""
    cpu_cores = mp.cpu_count()
    simulation_batch_size = min(total_samples, cpu_cores - 1)
    number_of_batches = int(np.ceil(total_samples / simulation_batch_size))
    
    PSC_mags = []
    weights = []
    locs = []
    
    start_time = time.time()
    
    with mp.Pool(processes=simulation_batch_size) as pool:
        for _ in tqdm(range(number_of_batches), desc="Running simulation batches"):
            batch_args = [
                (synapse_type, tuner_configs, location_type, distributions_to_test, use_norm_dist)
                for _ in range(simulation_batch_size)
            ]
            
            results = pool.starmap(simulate_PSC, batch_args)
            
            for PSC_mag, weight, loc in results:
                PSC_mags.append(PSC_mag)
                weights.append(weight)
                locs.append(loc)
    
    elapsed_time = time.time() - start_time
    print(f"Total simulation time: {elapsed_time:.2f} seconds")
    
    return np.array(PSC_mags), np.array(weights), np.array(locs)

def objective_function(params, synapse_type, location_type, target_metric):
    """Objective function for optimization."""
    # Update distribution parameters
    distributions_to_test[synapse_type][location_type]['mean'] = params[0]
    distributions_to_test[synapse_type][location_type]['std'] = params[1]
    
    # Run simulations
    PSC_mags, weights, _ = run_parallel_simulations(
        synapse_type=synapse_type,
        location_type=location_type,
        total_samples=50,  # Reduced for optimization
        use_norm_dist=True
    )
    
    # Calculate error
    mean_PSC = np.mean(PSC_mags)
    error = abs(mean_PSC - target_metric)
    
    return error

def optimize_synapse_parameters(synapse_type, location_type, target_metric, bounds):
    """Optimize synapse parameters using differential evolution."""
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(synapse_type, location_type, target_metric),
        workers=-1,  # Use all available cores
        updating='deferred',
        disp=True
    )
    
    return result

def plot_results(weights, PSC_mags, synapse_type, location_type):
    """Plot the results of the simulation."""
    plt.figure(figsize=(10, 6))
    plt.scatter(weights, PSC_mags, alpha=0.5)
    plt.xlabel('Synapse Weight')
    plt.ylabel('PSC Magnitude')
    plt.title(f'PSC vs Weight for {synapse_type} synapses in {location_type}')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Initialize cell and mechanisms
    template_arg = setup_cell_and_mechanisms()
    
    # Example usage
    synapse_type = 'exc'
    location_type = 'distal_basal'
    
    print(f"Running parallel simulations for {synapse_type} synapses in {location_type}...")
    PSC_mags, weights, locs = run_parallel_simulations(
        synapse_type=synapse_type,
        location_type=location_type,
        total_samples=100
    )
    
    # Plot results
    plot_results(weights, PSC_mags, synapse_type, location_type)
    
    # Example optimization
    target_metric = 0.5  # Example target PSC magnitude
    bounds = [(0.1, 1.0), (0.01, 0.5)]  # Mean and std bounds
    
    print(f"\nOptimizing parameters for {synapse_type} synapses in {location_type}...")
    result = optimize_synapse_parameters(synapse_type, location_type, target_metric, bounds)
    
    print(f"\nOptimization results:")
    print(f"Optimal mean: {result.x[0]:.4f}")
    print(f"Optimal std: {result.x[1]:.4f}")
    print(f"Final error: {result.fun:.4f}") 