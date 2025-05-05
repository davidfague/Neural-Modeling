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
from datetime import datetime

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
        load_hay_cell,
        save_simulation_results,
        save_summary_stats,
        save_pscs_by_segment
    )
except ImportError as e:
    print(f"Error importing general_settings_for_AST: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Looking for module in: {modules_dir}")
    sys.exit(1)

# Global settings
TOTAL_SAMPLES_PER_WEIGHT_DISTRIBUTION = 3000
USE_HAY_CELL = True

# Globals for worker processes
GLOBAL_tuner_configs = None
GLOBAL_distributions_to_test = None
GLOBAL_conn_type_settings = None
GLOBAL_template_arg = None
GLOBAL_optimization_histories = None

def setup_cell_and_mechanisms():
    """Initialize the cell model and load NEURON mechanisms."""
    # First check if mechanisms need to be compiled
    if not os.path.isdir(os.path.join(modfiles_dir, 'x86_64')):
        print("Compiling NEURON mechanisms...")
        os.chdir(modfiles_dir)
        os.system("nrnivmodl > /dev/null 2>&1")
        os.chdir(current_dir)
    
    # Load mechanisms first
    print("Loading NEURON mechanisms...")
    neuron.load_mechanisms(modfiles_dir)
    
    # Now load the cell/template
    if USE_HAY_CELL:
        print("Loading Hay cell template...")
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
            distributions_to_test[synapse_type][location_type].get('exc_scalar', 1.0)  # Default to 1.0 if not present
        )
    synapse_tuner_obj.syn.initW = new_weight
    return new_weight

def simulate_PSC(synapse_type, location_type, use_norm_dist):
    """Simulate PSC for a given synapse type and location with error handling."""
    try:
        global GLOBAL_tuner_configs, GLOBAL_distributions_to_test, GLOBAL_template_arg
        synapse_tuner_obj = InitializeSysnapseTuner(
            template_arg=GLOBAL_template_arg, 
            **GLOBAL_tuner_configs[True][synapse_type]
        )
        all_segments, possible_segments, seg_probs = get_segments(
            synapse_tuner_obj, 
            location_type
        )
        
        # Determine if we should use normal distribution based on synapse type
        use_norm_dist = 'inh' in synapse_type.lower()
        
        weight = change_synapse_weight(
            synapse_tuner_obj,
            GLOBAL_distributions_to_test,
            synapse_type,
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
    except Exception as e:
        print(f"Error in simulate_PSC: {str(e)}")
        return None, None, None

def run_parallel_simulations(synapse_type, location_type, total_samples=100):
    """Run parallel simulations and collect results."""
    use_norm_dist = 'inh' in synapse_type.lower()
    cpu_cores = mp.cpu_count()
    simulation_batch_size = min(total_samples, cpu_cores)
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
    
    try:
        # Create pool with maxtasksperchild to prevent memory leaks
        with mp.Pool(processes=simulation_batch_size, 
                    initializer=worker_init,
                    maxtasksperchild=10) as pool:
            for batch_idx in tqdm(range(number_of_batches), desc="Running simulation batches"):
                batch_args = [
                    (synapse_type, location_type, use_norm_dist)
                    for _ in range(simulation_batch_size)
                ]
                try:
                    results = pool.starmap(simulate_PSC, batch_args)
                    for PSC_mag, weight, loc in results:
                        if PSC_mag is not None:  # Check for valid results
                            PSC_mags.append(PSC_mag)
                            weights.append(weight)
                            locs.append(loc)
                            PSCs_by_segment[loc].append(PSC_mag)
                    print(f"Batch {batch_idx+1}/{number_of_batches}: mean PSC={np.mean(PSC_mags):.3f}, std PSC={np.std(PSC_mags):.3f}")
                except Exception as e:
                    print(f"Error in batch {batch_idx+1}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        raise
    finally:
        # Ensure pool is properly closed
        if 'pool' in locals():
            pool.close()
            pool.join()
    
    elapsed_time = time.time() - start_time
    print(f"Total simulation time: {elapsed_time:.2f} seconds")
    return np.array(PSC_mags), np.array(weights), np.array(locs), PSCs_by_segment

def objective_function(params, synapse_type, location_type, target_metric):
    """Objective function for optimization."""
    # Update distribution parameters
    distributions_to_test[synapse_type][location_type]['mean'] = params[0]
    distributions_to_test[synapse_type][location_type]['std'] = params[1]
    
    # Run simulations
    PSC_mags, weights, locs, PSCs_by_segment = run_parallel_simulations(
        synapse_type=synapse_type,
        location_type=location_type,
        total_samples=TOTAL_SAMPLES_PER_WEIGHT_DISTRIBUTION
    )
    
    mean_PSC = np.mean(PSC_mags)
    std_PSC = np.std(PSC_mags)
    
    # Use both mean and std in the error
    error = (mean_PSC - target_metric['mean'])**2 + (std_PSC - target_metric['std'])**2
    
    # Store results in optimization history
    optimization_histories[(synapse_type, location_type)].append({
        'params': params,
        'PSC_mags': PSC_mags,
        'weights': weights,
        'error': error
    })
    
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
        options={'maxiter': 10, 'disp': True}
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
    """Initialize worker process with proper error handling."""
    global GLOBAL_tuner_configs, GLOBAL_distributions_to_test, GLOBAL_conn_type_settings, GLOBAL_template_arg, GLOBAL_optimization_histories
    
    try:
        import neuron
        from general_settings_for_AST import (
            tuner_configs, distributions_to_test, conn_type_settings, load_hay_cell
        )
        GLOBAL_tuner_configs = tuner_configs
        GLOBAL_distributions_to_test = distributions_to_test
        GLOBAL_conn_type_settings = conn_type_settings
        GLOBAL_template_arg = load_hay_cell(conn_type_settings)
        GLOBAL_optimization_histories = {}
    except Exception as e:
        print(f"Error in worker initialization: {str(e)}")
        raise

def plot_optimization_history(history, synapse_type, location_type, save_dir):
    """Plot the optimization history showing parameter evolution and PSC distributions."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot parameter evolution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    means = [h['params'][0] for h in history]
    stds = [h['params'][1] for h in history]
    plt.plot(means, label='Mean')
    plt.plot(stds, label='Std')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title(f'Parameter Evolution for {synapse_type} in {location_type}')
    plt.legend()
    
    # Plot PSC distributions
    plt.subplot(1, 2, 2)
    for i, h in enumerate(history):
        PSC_mags = h['PSC_mags']
        plt.hist(PSC_mags, bins=30, alpha=0.3, label=f'Iter {i}')
    plt.xlabel('PSC Magnitude')
    plt.ylabel('Count')
    plt.title(f'PSC Distribution Evolution')
    plt.legend()
    
    plot_path = os.path.join(save_dir, f'optimization_history_{synapse_type}_{location_type}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved optimization history plot to {plot_path}")

if __name__ == '__main__':
    # Initialize cell and mechanisms
    template_arg = setup_cell_and_mechanisms()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'AA_PSC_tuning_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    optimization_histories = {}
    
    # Create documentation file
    with open(os.path.join(results_dir, 'simulation_parameters.txt'), 'w') as f:
        f.write("=== Simulation Parameters ===\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total CPU Cores Available: {mp.cpu_count()}\n")
        f.write(f"Total Samples per Weight Distribution: {TOTAL_SAMPLES_PER_WEIGHT_DISTRIBUTION}\n")
        f.write(f"Optimization Method: L-BFGS-B\n")
        f.write(f"Maximum Optimization Iterations: 10\n")
        f.write("\n=== Target Metrics ===\n")
        for syn_type, metrics in target_metrics.items():
            f.write(f"\n{syn_type}:\n")
            f.write(f"  Target Mean PSC: {metrics['magnitude']['mean']} pA\n")
            f.write(f"  Target Std PSC: {metrics['magnitude']['std']} pA\n")
        f.write("\n=== Location Types ===\n")
        for syn_type, loc_types in location_types_by_synapse_type.items():
            f.write(f"\n{syn_type}: {', '.join(loc_types)}\n")
    
    # Track total execution time
    total_start_time = time.time()
    
    # Iterate over all synapse types and their location types
    for synapse_type, location_types in location_types_by_synapse_type.items():
        for location_type in location_types:
            print(f"\nProcessing {synapse_type} synapses in {location_type}...")
            
            # Track time for this synapse/location combination
            start_time = time.time()
            
            # Get target metrics
            target_metric = target_metrics[synapse_type]['magnitude']
            
            # Set appropriate bounds based on synapse type
            if 'exc' in synapse_type:
                bounds = [(0.1, 1.0), (0.01, 0.5)]  # Mean and std bounds for excitatory
            else:
                bounds = [(0.5, 2.0), (0.01, 0.5)]  # Mean and std bounds for inhibitory
            
            # Initialize optimization history
            optimization_histories[(synapse_type, location_type)] = []
            
            # Run initial simulation
            print(f"Running initial simulations...")
            PSC_mags, weights, locs, PSCs_by_segment = run_parallel_simulations(
                synapse_type=synapse_type,
                location_type=location_type,
                total_samples=TOTAL_SAMPLES_PER_WEIGHT_DISTRIBUTION
            )
            
            # Store initial results
            optimization_histories[(synapse_type, location_type)].append({
                'params': [distributions_to_test[synapse_type][location_type]['mean'],
                          distributions_to_test[synapse_type][location_type]['std']],
                'PSC_mags': PSC_mags,
                'weights': weights,
                'error': (np.mean(PSC_mags) - target_metric['mean'])**2 + 
                        (np.std(PSC_mags) - target_metric['std'])**2
            })
            
            # Optimize parameters
            print(f"Optimizing parameters...")
            result = optimize_synapse_parameters(synapse_type, location_type, target_metric, bounds)
            
            # Store optimized parameters
            distributions_to_test[synapse_type][location_type]['mean'] = result.x[0]
            distributions_to_test[synapse_type][location_type]['std'] = result.x[1]
            
            # Run validation simulations
            print(f"Running validation simulations...")
            PSC_mags_val, weights_val, locs_val, _ = run_parallel_simulations(
                synapse_type=synapse_type,
                location_type=location_type,
                total_samples=TOTAL_SAMPLES_PER_WEIGHT_DISTRIBUTION
            )
            
            # Store final results
            optimization_histories[(synapse_type, location_type)].append({
                'params': result.x,
                'PSC_mags': PSC_mags_val,
                'weights': weights_val,
                'error': result.fun
            })
            
            # Calculate execution time for this combination
            elapsed_time = time.time() - start_time
            
            # Save results
            save_dir = os.path.join(results_dir, f"{synapse_type}_{location_type}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot and save results
            plot_validation_results(weights_val, PSC_mags_val, synapse_type, location_type, 
                                  target_metric, save_dir)
            plot_optimization_history(optimization_histories[(synapse_type, location_type)],
                                    synapse_type, location_type, save_dir)
            
            # Save data
            save_simulation_results(weights_val, PSC_mags_val, locs_val, 
                                  os.path.join(save_dir, 'simulation_results.csv'))
            save_summary_stats(np.mean(PSC_mags_val), np.std(PSC_mags_val), target_metric,
                             os.path.join(save_dir, 'summary.txt'))
            save_pscs_by_segment(PSCs_by_segment, 
                               os.path.join(save_dir, 'pscs_by_segment.pkl'))
            
            # Store results for final summary
            all_results.append({
                "Synapse Type": synapse_type,
                "Location Type": location_type,
                "initW_mean": round(result.x[0], 3),
                "initW_std": round(result.x[1], 3),
                "PSC Mean": round(np.mean(PSC_mags_val), 3),
                "PSC Std": round(np.std(PSC_mags_val), 3),
                "PSC_mean_error": round(target_metric['mean'] - np.mean(PSC_mags_val), 3),
                "PSC_std_error": round(target_metric['std'] - np.std(PSC_mags_val), 3),
                "Final Error": round(result.fun, 3),
                "Execution Time (s)": round(elapsed_time, 1),
                "Optimization Iterations": result.nit,
                "Optimization Success": result.success
            })
    
    # Calculate total execution time
    total_elapsed_time = time.time() - total_start_time
    
    # Save final summary
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, 'final_summary.csv'))
    
    # Update documentation with execution times
    with open(os.path.join(results_dir, 'simulation_parameters.txt'), 'a') as f:
        f.write("\n=== Execution Summary ===\n")
        f.write(f"Total Execution Time: {total_elapsed_time:.1f} seconds\n")
        f.write(f"Average Time per Synapse/Location: {total_elapsed_time/len(all_results):.1f} seconds\n")
        f.write("\n=== Individual Execution Times ===\n")
        for result in all_results:
            f.write(f"\n{result['Synapse Type']} - {result['Location Type']}:\n")
            f.write(f"  Execution Time: {result['Execution Time (s)']} seconds\n")
            f.write(f"  Optimization Iterations: {result['Optimization Iterations']}\n")
            f.write(f"  Optimization Success: {result['Optimization Success']}\n")
    
    print("\nFinal summary saved to", os.path.join(results_dir, 'final_summary.csv'))
    print(f"Total execution time: {total_elapsed_time:.1f} seconds")