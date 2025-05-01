#!/usr/bin/env python3

"""
Automated Synapse Tuning (AST) - Parallel Version

This script implements parallel simulation and optimization for synapse tuning,
focusing on location-dependent PSC tuning.
"""

import time
import numpy as np
from scipy.optimize import minimize
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
modfiles_dir = '/users/drfrbc/Neural-Modeling/notebooks/bmtool/examples/synapses/modfiles'
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

# Globals for worker processes
GLOBAL_tuner_configs = None
GLOBAL_distributions_to_test = None
GLOBAL_conn_type_settings = None
GLOBAL_template_arg = None

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
    # Now load the cell/template
    if USE_HAY_CELL:
        template_arg = load_hay_cell(conn_type_settings)
    else:
        template_arg = None
    return template_arg

def get_sec_ids_from_type(section_type):
    """Get section IDs based on section type."""
    cell = h.L5PCtemplate("/users/drfrbc/Neural-Modeling/cells/templates/cell1.asc")

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

def simulate_PSC(synapse_type, location_type, use_norm_dist):
    """Simulate PSC for a given synapse type and location."""
    global GLOBAL_tuner_configs, GLOBAL_distributions_to_test, GLOBAL_template_arg
    synapse_tuner_obj = InitializeSysnapseTuner(
        template_arg=GLOBAL_template_arg, 
        **GLOBAL_tuner_configs[True][synapse_type]
    )
    all_segments, possible_segments, seg_probs = get_segments(
        synapse_tuner_obj, 
        location_type
    )
    weight = change_synapse_weight(
        synapse_tuner_obj,
        GLOBAL_distributions_to_test,
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
    # Use a local tuner to get segment count
    synapse_tuner_obj = InitializeSysnapseTuner(template_arg=template_arg, **tuner_configs[True][synapse_type])
    all_segments = [seg for sec in synapse_tuner_obj.cell.all for seg in sec]
    total_segments = len(all_segments)
    PSCs_by_segment = {seg_idx: [] for seg_idx in range(total_segments)}
    start_time = time.time()
    with mp.Pool(processes=simulation_batch_size, initializer=worker_init) as pool:
        for batch_idx in tqdm(range(number_of_batches), desc="Running simulation batches"):
            batch_args = [
                (synapse_type, location_type, use_norm_dist)
                for _ in range(simulation_batch_size)
            ]
            results = pool.starmap(simulate_PSC, batch_args)
            for PSC_mag, weight, loc in results:
                PSC_mags.append(PSC_mag)
                weights.append(weight)
                locs.append(loc)
                PSCs_by_segment[loc].append(PSC_mag)
            print(f"Batch {batch_idx+1}/{number_of_batches}: mean PSC={np.mean(PSC_mags):.3f}, std PSC={np.std(PSC_mags):.3f}")
    elapsed_time = time.time() - start_time
    print(f"Total simulation time: {elapsed_time:.2f} seconds")
    return np.array(PSC_mags), np.array(weights), np.array(locs), PSCs_by_segment

def objective_function(params, synapse_type, location_type, target_metric):
    """Objective function for optimization."""
    # Update distribution parameters
    distributions_to_test[synapse_type][location_type]['mean'] = params[0]
    distributions_to_test[synapse_type][location_type]['std'] = params[1]
    
    # Run simulations
    PSC_mags, weights, _ = run_parallel_simulations(
        synapse_type=synapse_type,
        location_type=location_type,
        total_samples=50,
        use_norm_dist=True
    )
    
    mean_PSC = np.mean(PSC_mags)
    std_PSC = np.std(PSC_mags)
    
    # Use both mean and std in the error
    error = (mean_PSC - target_metric['mean'])**2 + (std_PSC - target_metric['std'])**2
    
    return error

def optimize_synapse_parameters(synapse_type, location_type, target_metric, bounds):
    """Optimize synapse parameters using scipy.optimize.minimize."""
    # Initial guess: midpoint of bounds
    x0 = [np.mean([b[0], b[1]]) for b in bounds]
    result = minimize(
        objective_function,
        x0=x0,
        args=(synapse_type, location_type, target_metric),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50, 'disp': True}
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

def plot_validation_results(weights, PSC_mags, synapse_type, location_type, target_metric, save_dir):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(PSC_mags, bins=30, alpha=0.7, label='Simulated PSCs')
    plt.axvline(target_metric['mean'], color='r', linestyle='--', label='Target Mean')
    plt.axvline(np.mean(PSC_mags), color='g', linestyle='-', label='Simulated Mean')
    plt.axvline(target_metric['mean'] + target_metric['std'], color='r', linestyle=':', label='Target ±Std')
    plt.axvline(target_metric['mean'] - target_metric['std'], color='r', linestyle=':')
    plt.axvline(np.mean(PSC_mags) + np.std(PSC_mags), color='g', linestyle=':', label='Simulated ±Std')
    plt.axvline(np.mean(PSC_mags) - np.std(PSC_mags), color='g', linestyle=':')
    plt.xlabel('PSC Magnitude')
    plt.ylabel('Count')
    plt.title(f'PSC Distribution for {synapse_type} in {location_type}')
    plt.legend()
    plot_path = os.path.join(save_dir, f'PSC_hist_{synapse_type}_{location_type}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved validation plot to {plot_path}")

def worker_init():
    global GLOBAL_tuner_configs, GLOBAL_distributions_to_test, GLOBAL_conn_type_settings, GLOBAL_template_arg
    import neuron
    # modfiles_dir = '/users/drfrbc/Neural-Modeling/notebooks/bmtool/examples/synapses/modfiles'
    # neuron.load_mechanisms(modfiles_dir)
    from general_settings_for_AST import (
        tuner_configs, distributions_to_test, conn_type_settings, load_hay_cell
    )
    GLOBAL_tuner_configs = tuner_configs
    GLOBAL_distributions_to_test = distributions_to_test
    GLOBAL_conn_type_settings = conn_type_settings
    GLOBAL_template_arg = load_hay_cell(conn_type_settings)

if __name__ == '__main__':
    # Initialize cell and mechanisms
    template_arg = setup_cell_and_mechanisms()
    
    # Example usage
    synapse_type = 'exc'
    location_type = 'distal_basal'
    
    print(f"Running parallel simulations for {synapse_type} synapses in {location_type}...")
    PSC_mags, weights, locs, PSCs_by_segment = run_parallel_simulations(
        synapse_type=synapse_type,
        location_type=location_type,
        total_samples=100
    )
    
    # Plot results
    plot_results(weights, PSC_mags, synapse_type, location_type)
    
    # Example optimization
    target_metric = {'mean': 0.5, 'std': 0.1}  # Example target PSC magnitude
    bounds = [(0.1, 1.0), (0.01, 0.5)]  # Mean and std bounds
    
    print(f"\nOptimizing parameters for {synapse_type} synapses in {location_type}...")
    result = optimize_synapse_parameters(synapse_type, location_type, target_metric, bounds)
    print(f"\nOptimization results:")
    print(f"Optimal mean: {result.x[0]:.4f}")
    print(f"Optimal std: {result.x[1]:.4f}")
    print(f"Final error: {result.fun:.4f}")

    # Post-optimization validation
    print("\nRunning post-optimization validation...")
    # Set optimized parameters
    distributions_to_test[synapse_type][location_type]['mean'] = result.x[0]
    distributions_to_test[synapse_type][location_type]['std'] = result.x[1]
    # Rerun simulations
    PSC_mags_val, weights_val, locs_val, _ = run_parallel_simulations(
        synapse_type=synapse_type,
        location_type=location_type,
        total_samples=100
    )
    # Print summary
    print(f"Validation: Simulated mean PSC = {np.mean(PSC_mags_val):.4f}, std = {np.std(PSC_mags_val):.4f}")
    print(f"Target: mean = {target_metric['mean']:.4f}, std = {target_metric['std']:.4f}")
    # Plot and save
    plot_validation_results(weights_val, PSC_mags_val, synapse_type, location_type, target_metric, save_dir='AA_results') 
    from general_settings_for_AST import save_simulation_results, save_summary_stats, save_pscs_by_segment
    save_simulation_results(weights, PSC_mags, locs, 'AA_results/simulation_results.csv')
    save_summary_stats(np.mean(PSC_mags), np.std(PSC_mags), target_metric, 'AA_results/summary.txt')
    save_pscs_by_segment(PSCs_by_segment, 'AA_results/pscs_by_segment.pkl')