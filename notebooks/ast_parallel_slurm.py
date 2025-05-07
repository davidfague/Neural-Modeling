#!/usr/bin/env python3

"""
Automated Synapse Tuning (AST) - Parallel Version

This script implements parallel simulation and optimization for synapse tuning,
focusing on location-dependent PSC tuning. Supports both local and SLURM-based
distributed computing across multiple nodes.
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
import subprocess
import socket
import json
import tempfile
from pathlib import Path
import csv
import pickle

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
modules_dir = os.path.join(root_dir, 'Modules')
MODFILES_DIR = '/users/drfrbc/Neural-Modeling/notebooks/bmtool/examples/synapses/modfiles'  # <-- Edit here if needed
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
print(f"Modfiles directory: {MODFILES_DIR}")
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

# SLURM configuration
SLURM_CONFIG = {
    'partition': 'standard',  # Default partition
    'time': '24:00:00',      # Default time limit
    'nodes': 8,              # All 8 nodes
    'ntasks_per_node': None, # Will be set based on node CPU cores
    'cpus_per_task': 1,      # Single CPU per task for better distribution
    'mem': None,             # Will be set based on node memory
    'use_slurm': True        # Enable SLURM for distributed computing
}

# --- Parameterized Paths ---
CELL_TEMPLATE_PATH = "/users/drfrbc/Neural-Modeling/cells/templates/cell1.asc"  # <-- Edit here if needed
modfiles_dir = MODFILES_DIR

# --- Logging Utility ---
def log(msg):
    print(f"[LOG {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def log_timing(results_dir, synapse_type, location_type, elapsed_time, total=False):
    timing_file = os.path.join(results_dir, 'timing_summary.csv')
    write_header = not os.path.exists(timing_file)
    with open(timing_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['Synapse Type', 'Location Type', 'Execution Time (s)'])
        if total:
            writer.writerow(['Total', 'All', f'{elapsed_time:.1f}'])
        else:
            writer.writerow([synapse_type, location_type, f'{elapsed_time:.1f}'])

# --- Save Intermediate Results ---
def save_intermediate(results_dir, optimization_histories, all_results):
    # Save optimization histories
    with open(os.path.join(results_dir, 'optimization_histories.pkl'), 'wb') as f:
        pickle.dump(optimization_histories, f)
    # Save all_results as CSV
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, 'all_results_intermediate.csv'), index=False)
    log("Intermediate results saved.")

def get_slurm_node_list():
    """Get list of nodes allocated by SLURM or direct SSH."""
    # First try SLURM
    if 'SLURM_JOB_NODELIST' in os.environ:
        node_list = os.environ['SLURM_JOB_NODELIST']
        try:
            # Use scontrol to expand the node list
            cmd = ['scontrol', 'show', 'hostnames', node_list]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout.strip().split('\n')
        except Exception as e:
            print(f"Error getting SLURM node list: {e}")
    
    # If SLURM fails or isn't available, try direct SSH approach
    try:
        # For CloudLab, we know we have 8 nodes
        nodes = [f'node{i}' for i in range(SLURM_CONFIG['nodes'])]
        
        # Verify SSH access to each node
        accessible_nodes = []
        for node in nodes:
            try:
                # Try a simple command to verify SSH access
                result = subprocess.run(['ssh', node, 'hostname'], 
                                     capture_output=True, 
                                     timeout=5)
                if result.returncode == 0:
                    accessible_nodes.append(node)
            except subprocess.TimeoutExpired:
                print(f"Timeout connecting to {node}")
            except Exception as e:
                print(f"Error connecting to {node}: {e}")
        
        if accessible_nodes:
            print(f"Using direct SSH access to nodes: {accessible_nodes}")
            return accessible_nodes
    except Exception as e:
        print(f"Error in direct SSH approach: {e}")
    
    # If all else fails, return local host
    print("Falling back to local execution")
    return [socket.gethostname()]

def create_slurm_script(synapse_type, location_type, total_samples, results_dir):
    """Create a SLURM script for distributed simulation."""
    script = f"""#!/bin/bash
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --nodes={SLURM_CONFIG['nodes']}
#SBATCH --ntasks-per-node={SLURM_CONFIG['ntasks_per_node']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --job-name=ast_{synapse_type}_{location_type}
#SBATCH --output={results_dir}/slurm_%j.out

# Load required modules
module load python/3.8
module load neuron/7.8

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Run the simulation
python {os.path.abspath(__file__)} \\
    --synapse_type {synapse_type} \\
    --location_type {location_type} \\
    --total_samples {total_samples} \\
    --results_dir {results_dir} \\
    --slurm_node
"""
    return script

def submit_slurm_job(script_content, job_name):
    """Submit a SLURM job and return the job ID."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        cmd = ['sbatch', script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id
    finally:
        os.unlink(script_path)

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
    cell = h.L5PCtemplate(CELL_TEMPLATE_PATH)
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

class ResourceMonitor:
    """Monitor and log resource utilization during execution."""
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.start_time = time.time()
        self.resource_log = os.path.join(results_dir, 'resource_utilization.csv')
        self.performance_log = os.path.join(results_dir, 'performance_metrics.csv')
        self.initialize_logs()
        
    def initialize_logs(self):
        """Initialize log files with headers."""
        # Resource utilization log
        with open(self.resource_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Node', 'CPU_Usage_Percent', 'Memory_Usage_GB', 
                           'Memory_Available_GB', 'Tasks_Running', 'Simulations_Completed'])
        
        # Performance metrics log
        with open(self.performance_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Synapse_Type', 'Location_Type', 
                           'Simulations_Completed', 'Total_Simulations', 
                           'Average_Time_Per_Simulation', 'Estimated_Time_Remaining'])
    
    def log_resource_usage(self, node, simulations_completed):
        """Log current resource usage for a node."""
        try:
            # Get CPU usage
            cpu_info = subprocess.run(['ssh', node, 'top', '-bn1'], capture_output=True, text=True)
            cpu_usage = float(cpu_info.stdout.split('\n')[2].split()[1])
            
            # Get memory usage
            mem_info = subprocess.run(['ssh', node, 'free', '-g'], capture_output=True, text=True)
            mem_lines = mem_info.stdout.split('\n')
            total_mem = int(mem_lines[1].split()[1])
            used_mem = int(mem_lines[1].split()[2])
            available_mem = int(mem_lines[1].split()[6])
            
            # Get running tasks
            task_info = subprocess.run(['ssh', node, 'ps', 'aux'], capture_output=True, text=True)
            tasks_running = len([line for line in task_info.stdout.split('\n') 
                               if 'python' in line and 'ast_parallel_slurm.py' in line])
            
            # Log the data
            with open(self.resource_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    node,
                    cpu_usage,
                    used_mem,
                    available_mem,
                    tasks_running,
                    simulations_completed
                ])
        except Exception as e:
            log(f"Error logging resource usage for {node}: {e}")
    
    def log_performance_metrics(self, synapse_type, location_type, 
                              completed, total, avg_time):
        """Log performance metrics for the current batch."""
        try:
            elapsed_time = time.time() - self.start_time
            estimated_remaining = (total - completed) * avg_time if completed > 0 else 0
            
            with open(self.performance_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    synapse_type,
                    location_type,
                    completed,
                    total,
                    avg_time,
                    estimated_remaining
                ])
        except Exception as e:
            log(f"Error logging performance metrics: {e}")
    
    def generate_summary_report(self):
        """Generate a summary report of resource utilization and performance."""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Read the logs
            resource_df = pd.read_csv(self.resource_log)
            performance_df = pd.read_csv(self.performance_log)
            
            # Create summary directory
            summary_dir = os.path.join(self.results_dir, 'summary_reports')
            os.makedirs(summary_dir, exist_ok=True)
            
            # Generate resource utilization plots
            plt.figure(figsize=(15, 10))
            
            # CPU Usage
            plt.subplot(2, 2, 1)
            for node in resource_df['Node'].unique():
                node_data = resource_df[resource_df['Node'] == node]
                plt.plot(node_data['Timestamp'], node_data['CPU_Usage_Percent'], 
                        label=node)
            plt.title('CPU Usage Over Time')
            plt.xlabel('Time')
            plt.ylabel('CPU Usage (%)')
            plt.legend()
            
            # Memory Usage
            plt.subplot(2, 2, 2)
            for node in resource_df['Node'].unique():
                node_data = resource_df[resource_df['Node'] == node]
                plt.plot(node_data['Timestamp'], node_data['Memory_Usage_GB'], 
                        label=node)
            plt.title('Memory Usage Over Time')
            plt.xlabel('Time')
            plt.ylabel('Memory Usage (GB)')
            plt.legend()
            
            # Simulations Progress
            plt.subplot(2, 2, 3)
            for syn_type in performance_df['Synapse_Type'].unique():
                syn_data = performance_df[performance_df['Synapse_Type'] == syn_type]
                plt.plot(syn_data['Timestamp'], 
                        syn_data['Simulations_Completed'] / syn_data['Total_Simulations'] * 100,
                        label=syn_type)
            plt.title('Simulation Progress')
            plt.xlabel('Time')
            plt.ylabel('Progress (%)')
            plt.legend()
            
            # Average Time per Simulation
            plt.subplot(2, 2, 4)
            for syn_type in performance_df['Synapse_Type'].unique():
                syn_data = performance_df[performance_df['Synapse_Type'] == syn_type]
                plt.plot(syn_data['Timestamp'], syn_data['Average_Time_Per_Simulation'],
                        label=syn_type)
            plt.title('Average Time per Simulation')
            plt.xlabel('Time')
            plt.ylabel('Time (seconds)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'resource_utilization.png'))
            plt.close()
            
            # Generate text summary
            with open(os.path.join(summary_dir, 'execution_summary.txt'), 'w') as f:
                f.write("=== Execution Summary ===\n\n")
                
                # Overall statistics
                f.write("Overall Statistics:\n")
                f.write(f"Total Execution Time: {time.time() - self.start_time:.2f} seconds\n")
                f.write(f"Total Simulations Completed: {performance_df['Simulations_Completed'].max()}\n")
                f.write(f"Average Time per Simulation: {performance_df['Average_Time_Per_Simulation'].mean():.2f} seconds\n\n")
                
                # Resource utilization statistics
                f.write("Resource Utilization Statistics:\n")
                f.write(f"Average CPU Usage: {resource_df['CPU_Usage_Percent'].mean():.2f}%\n")
                f.write(f"Peak CPU Usage: {resource_df['CPU_Usage_Percent'].max():.2f}%\n")
                f.write(f"Average Memory Usage: {resource_df['Memory_Usage_GB'].mean():.2f} GB\n")
                f.write(f"Peak Memory Usage: {resource_df['Memory_Usage_GB'].max():.2f} GB\n\n")
                
                # Per synapse type statistics
                f.write("Per Synapse Type Statistics:\n")
                for syn_type in performance_df['Synapse_Type'].unique():
                    syn_data = performance_df[performance_df['Synapse_Type'] == syn_type]
                    f.write(f"\n{syn_type}:\n")
                    f.write(f"  Total Simulations: {syn_data['Total_Simulations'].iloc[0]}\n")
                    f.write(f"  Average Time per Simulation: {syn_data['Average_Time_Per_Simulation'].mean():.2f} seconds\n")
                    f.write(f"  Total Execution Time: {syn_data['Estimated_Time_Remaining'].iloc[-1]:.2f} seconds\n")
            
            log(f"Summary report generated in {summary_dir}")
            
        except Exception as e:
            log(f"Error generating summary report: {e}")

def run_parallel_simulations(synapse_type, location_type, total_samples=100):
    """Run parallel simulations and collect results."""
    use_norm_dist = 'inh' in synapse_type.lower()
    
    # Initialize resource monitor
    monitor = ResourceMonitor(results_dir)
    
    # Get list of available nodes
    nodes = get_slurm_node_list()
    print(f"Using nodes: {nodes}")
    
    # Calculate samples per node
    samples_per_node = total_samples // len(nodes)
    remaining_samples = total_samples % len(nodes)
    
    # Submit jobs to each node
    job_ids = []
    for i, node in enumerate(nodes):
        node_samples = samples_per_node + (1 if i < remaining_samples else 0)
        
        if SLURM_CONFIG['use_slurm']:
            # Use SLURM submission
            script = create_slurm_script(synapse_type, location_type, node_samples, results_dir)
            job_id = submit_slurm_job(script, f"ast_{synapse_type}_{location_type}_{node}")
            job_ids.append(job_id)
            print(f"Submitted SLURM job {job_id} to {node} for {node_samples} samples")
        else:
            # Use direct SSH execution
            try:
                # Create a temporary script for the node
                script_content = f"""#!/bin/bash
cd {os.getcwd()}
python {os.path.abspath(__file__)} \\
    --synapse_type {synapse_type} \\
    --location_type {location_type} \\
    --total_samples {node_samples} \\
    --results_dir {results_dir} \\
    --node_name {node}
"""
                script_path = os.path.join(results_dir, f"node_script_{node}.sh")
                with open(script_path, 'w') as f:
                    f.write(script_content)
                os.chmod(script_path, 0o755)
                
                # Execute the script on the remote node
                cmd = ['ssh', node, f'bash {script_path}']
                process = subprocess.Popen(cmd)
                job_ids.append(process)
                print(f"Started direct execution on {node} for {node_samples} samples")
            except Exception as e:
                print(f"Error starting execution on {node}: {e}")
    
    # Monitor jobs and collect results
    PSC_mags = []
    weights = []
    locs = []
    PSCs_by_segment = {}
    completed_simulations = 0
    
    while completed_simulations < total_samples:
        for i, node in enumerate(nodes):
            # Log resource usage
            monitor.log_resource_usage(node, completed_simulations)
            
            # Check for completed jobs
            node_results = os.path.join(results_dir, f"results_{synapse_type}_{location_type}_{node}.json")
            if os.path.exists(node_results):
                with open(node_results, 'r') as f:
                    node_data = json.load(f)
                    PSC_mags.extend(node_data['PSC_mags'])
                    weights.extend(node_data['weights'])
                    locs.extend(node_data['locs'])
                    for seg, pscs in node_data['PSCs_by_segment'].items():
                        if seg not in PSCs_by_segment:
                            PSCs_by_segment[seg] = []
                        PSCs_by_segment[seg].extend(pscs)
                    completed_simulations += len(node_data['PSC_mags'])
            
            # Log performance metrics
            avg_time = (time.time() - monitor.start_time) / completed_simulations if completed_simulations > 0 else 0
            monitor.log_performance_metrics(synapse_type, location_type, 
                                         completed_simulations, total_samples, avg_time)
        
        print(f"Progress: {completed_simulations}/{total_samples} simulations completed")
        time.sleep(10)  # Check every 10 seconds
    
    # Generate final summary report
    monitor.generate_summary_report()
    
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
    
    log(f"Objective: Params={params}, Error={error}")
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
    log(f"Optimization result: {result}")
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
    global GLOBAL_tuner_configs, GLOBAL_distributions_to_test, GLOBAL_conn_type_settings, GLOBAL_template_arg, GLOBAL_optimization_histories
    import neuron
    from general_settings_for_AST import (
        tuner_configs, distributions_to_test, conn_type_settings, load_hay_cell
    )
    GLOBAL_tuner_configs = tuner_configs
    GLOBAL_distributions_to_test = distributions_to_test
    GLOBAL_conn_type_settings = conn_type_settings
    GLOBAL_template_arg = load_hay_cell(conn_type_settings)
    GLOBAL_optimization_histories = {}

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

def get_node_info():
    """Gather information about the current node and its resources."""
    node_info = {
        'hostname': socket.gethostname(),
        'cpu_count': mp.cpu_count(),
        'memory': None,
        'slurm_info': {}
    }
    
    # Try to get memory information
    try:
        import psutil
        node_info['memory'] = {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available
        }
    except ImportError:
        node_info['memory'] = "psutil not available"
    
    # Get SLURM information if available
    slurm_vars = [
        'SLURM_JOB_ID', 'SLURM_NODEID', 'SLURM_CPUS_ON_NODE',
        'SLURM_MEM_PER_NODE', 'SLURM_NODELIST'
    ]
    for var in slurm_vars:
        if var in os.environ:
            node_info['slurm_info'][var] = os.environ[var]
    
    return node_info

def save_node_info(results_dir):
    """Save information about all nodes to a text file."""
    nodes = get_slurm_node_list()
    node_info_file = os.path.join(results_dir, 'node_info.txt')
    
    with open(node_info_file, 'w') as f:
        f.write("=== Node Information ===\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Nodes: {len(nodes)}\n\n")
        
        for node in nodes:
            f.write(f"\nNode: {node}\n")
            f.write("-" * 50 + "\n")
            
            # Get node info
            node_info = get_node_info()
            
            # Write basic info
            f.write(f"Hostname: {node_info['hostname']}\n")
            f.write(f"CPU Count: {node_info['cpu_count']}\n")
            
            # Write memory info
            if isinstance(node_info['memory'], dict):
                f.write(f"Total Memory: {node_info['memory']['total'] / (1024**3):.2f} GB\n")
                f.write(f"Available Memory: {node_info['memory']['available'] / (1024**3):.2f} GB\n")
            else:
                f.write(f"Memory Info: {node_info['memory']}\n")
            
            # Write SLURM info
            if node_info['slurm_info']:
                f.write("\nSLURM Information:\n")
                for key, value in node_info['slurm_info'].items():
                    f.write(f"{key}: {value}\n")
            
            f.write("\n")

def get_available_partitions():
    """Get list of available SLURM partitions."""
    try:
        result = subprocess.run(['sinfo', '-o', '%P'], capture_output=True, text=True)
        partitions = [p.strip() for p in result.stdout.split('\n') if p.strip()]
        return partitions
    except Exception as e:
        log(f"Error getting partitions: {e}")
        return ['standard']  # Default to standard if can't get partitions

def get_node_specs():
    """Get specifications of allocated nodes."""
    try:
        # Get node list
        nodes = get_slurm_node_list()
        node_specs = {}
        
        for node in nodes:
            # Get CPU info
            cpu_info = subprocess.run(['ssh', node, 'nproc'], capture_output=True, text=True)
            cpu_count = int(cpu_info.stdout.strip())
            
            # Get memory info
            mem_info = subprocess.run(['ssh', node, 'free', '-g'], capture_output=True, text=True)
            total_mem = int(mem_info.stdout.split('\n')[1].split()[1])
            
            node_specs[node] = {
                'cpus': cpu_count,
                'memory_gb': total_mem
            }
        
        return node_specs
    except Exception as e:
        log(f"Error getting node specs: {e}")
        return None

def update_slurm_config():
    """Update SLURM configuration based on available resources."""
    # Get available partitions
    partitions = get_available_partitions()
    if partitions:
        SLURM_CONFIG['partition'] = partitions[0]  # Use first available partition
        log(f"Using partition: {SLURM_CONFIG['partition']}")
    
    # Get node specifications
    node_specs = get_node_specs()
    if node_specs:
        # Use minimum values across all nodes to ensure compatibility
        min_cpus = min(spec['cpus'] for spec in node_specs.values())
        min_memory = min(spec['memory_gb'] for spec in node_specs.values())
        
        # Set tasks per node to use all available CPUs
        SLURM_CONFIG['ntasks_per_node'] = min_cpus
        SLURM_CONFIG['mem'] = f"{min_memory}G"
        
        log(f"Node specifications:")
        for node, specs in node_specs.items():
            log(f"  {node}: {specs['cpus']} CPUs, {specs['memory_gb']}GB memory")
        log(f"Using minimum values: {min_cpus} CPUs, {min_memory}GB memory per node")
    else:
        # Default values if can't get node specs
        SLURM_CONFIG['ntasks_per_node'] = 16  # Common default
        SLURM_CONFIG['mem'] = '32G'  # Common default
        log("Using default values for node specifications")

def validate_slurm_config():
    """Validate and print SLURM configuration."""
    log("Validating SLURM configuration...")
    log(f"Partition: {SLURM_CONFIG['partition']}")
    log(f"Total nodes: {SLURM_CONFIG['nodes']}")
    log(f"Tasks per node: {SLURM_CONFIG['ntasks_per_node']}")
    log(f"CPUs per task: {SLURM_CONFIG['cpus_per_task']}")
    log(f"Total CPUs: {SLURM_CONFIG['nodes'] * SLURM_CONFIG['ntasks_per_node'] * SLURM_CONFIG['cpus_per_task']}")
    log(f"Memory per node: {SLURM_CONFIG['mem']}")
    log(f"Total memory: {SLURM_CONFIG['nodes'] * int(SLURM_CONFIG['mem'].replace('G', ''))}G")
    log(f"Time limit: {SLURM_CONFIG['time']}")
    log(f"Using SLURM: {SLURM_CONFIG['use_slurm']}")

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Automated Synapse Tuning')
    parser.add_argument('--synapse_type', help='Specific synapse type to process')
    parser.add_argument('--location_type', help='Specific location type to process')
    parser.add_argument('--total_samples', type=int, default=100, help='Total number of samples')
    parser.add_argument('--results_dir', help='Results directory')
    parser.add_argument('--slurm_node', action='store_true', help='Flag indicating this is running on a SLURM node')
    args = parser.parse_args()
    
    # Initialize cell and mechanisms
    template_arg = setup_cell_and_mechanisms()
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or f'AA_PSC_tuning_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Update and validate SLURM configuration
    if SLURM_CONFIG['use_slurm']:
        update_slurm_config()
        validate_slurm_config()
    save_node_info(results_dir)
    
    if args.slurm_node:
        # This is running on a SLURM node, process specific synapse/location
        PSC_mags, weights, locs, PSCs_by_segment = run_parallel_simulations(
            args.synapse_type,
            args.location_type,
            args.total_samples
        )
        
        # Save results for this node
        node_results = {
            'PSC_mags': PSC_mags.tolist(),
            'weights': weights.tolist(),
            'locs': locs.tolist(),
            'PSCs_by_segment': {str(k): v for k, v in PSCs_by_segment.items()}
        }
        
        with open(os.path.join(results_dir, f"results_{args.synapse_type}_{args.location_type}_{socket.gethostname()}.json"), 'w') as f:
            json.dump(node_results, f)
    else:
        # Original main code for local execution or SLURM job submission
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
                
                log_timing(results_dir, synapse_type, location_type, elapsed_time)
                save_intermediate(results_dir, optimization_histories, all_results)
        
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
        log_timing(results_dir, 'Total', 'All', total_elapsed_time, total=True)