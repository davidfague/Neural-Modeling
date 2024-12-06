# FOR DOING MULTIPLE SEEDS
import sys
sys.path.append("../")
sys.path.append("../Modules/")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import analysis
from logger import Logger
from scipy.integrate import simps
from sklearn.metrics import mean_squared_error, r2_score

# Global constant to control Nexus plotting
USE_NEXUS_PLOTTING = False # not implemented. need to simulate nexus current injection and separate simulation folders based on whether nexus current inj
XLIM = None#(0,2)

def filter_spike_times(spike_times, start_time, end_time):
    return spike_times[(spike_times >= start_time) & (spike_times <= end_time)]

def analyze_and_log(soma_spikes, parameters):
    filtered_spike_times = filter_spike_times(soma_spikes, parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration)
    firing_rate = len(filtered_spike_times) * 1000 / parameters.h_i_duration if len(filtered_spike_times) > 0 else 0
    return firing_rate

def calculate_rmse_and_r2(amplitudes, firing_rates, reference_amplitudes, reference_firing_rates):
    # Interpolate reference firing rates to match the amplitudes of the current group
    reference_interpolation = np.interp(amplitudes, reference_amplitudes, reference_firing_rates)
    
    rmse = np.sqrt(mean_squared_error(firing_rates, reference_interpolation))
    r2 = r2_score(firing_rates, reference_interpolation)
    
    return rmse, r2

def plot_all_fi_curves(grouped_data, save=False, save_directory=""):
    fig, ax = plt.subplots(figsize=(10, 8))
    statistics = {}
    
    # Define a color map for different cell types
    colors = plt.cm.get_cmap('tab20', len(grouped_data))
    seed_colors = {}
    seed_index = 0
    
    # Assign different base colors to different seeds
    for celltype, seeds_data in grouped_data.items():
        for seed in seeds_data:
            if seed not in seed_colors:
                seed_colors[seed] = colors(seed_index)
                seed_index += 1
    
    # Plot each cell type and seed with appropriate color shades
    for celltype, seeds_data in grouped_data.items():
        for seed, (amplitudes, firing_rates) in seeds_data.items():
            base_color = seed_colors[seed]
            if celltype == "Complex":
                shade = 0.5
            else:
                shade = 1.0
                
            label = f"{celltype}_{seed}"
            ax.plot(amplitudes, firing_rates, label=label, color=np.array(base_color) * shade, alpha = 0.6)

            if celltype != "Complex":
                reference_amplitudes, reference_firing_rates = grouped_data.get("Complex", {}).get(seed, (None, None))
                if reference_amplitudes is not None and reference_firing_rates is not None:
                    rmse, r2 = calculate_rmse_and_r2(amplitudes, firing_rates, reference_amplitudes, reference_firing_rates)
                    statistics[label] = (rmse, r2)

    ax.set_title("FI Curve")
    ax.set_xlabel('Current Injection (nA)')
    ax.set_ylabel('Firing Rate (Hz)')
    if XLIM is not None:
        ax.set_xlim(XLIM)
    ax.legend()

    plt.tight_layout()
    plt.show()

    if save:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        fig.savefig(os.path.join(save_directory, "FI.png"), dpi=fig.dpi)
        with open(os.path.join(save_directory, 'comparative_statistics.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Curve', 'RMSE', 'R2'])
            for base_name, (rmse, r2) in statistics.items():
                writer.writerow([base_name, rmse, r2])

def plot_mean_and_std(grouped_data, save=False, save_directory="", x_axis: str =None):
    '''Plots FI mean and std of multiple seeds for multiple cells'''
    fig, ax = plt.subplots(figsize=(5, 4))
    
    celltype_data = {}

    # Collect data for each cell type
    for celltype, seeds_data in grouped_data.items():
        all_amplitudes = []
        all_firing_rates = []
        for seed, (amplitudes, firing_rates) in seeds_data.items():
            all_amplitudes.append(amplitudes)
            all_firing_rates.append(firing_rates)
        
        # Convert lists to numpy arrays for easier manipulation
        all_amplitudes = np.array(all_amplitudes)
        all_firing_rates = np.array(all_firing_rates)
        
        # Calculate mean and standard deviation
        mean_firing_rates = np.mean(all_firing_rates, axis=0)
        std_firing_rates = np.std(all_firing_rates, axis=0)
        
        celltype_data[celltype] = (all_amplitudes[0], mean_firing_rates, std_firing_rates)
    
    print(celltype_data)
    # Plot mean and standard deviation for each cell type
    for celltype, (amplitudes, mean_firing_rates, std_firing_rates) in celltype_data.items():
        ax.plot(amplitudes, mean_firing_rates, label=f"{celltype} Mean")
        ax.fill_between(amplitudes, mean_firing_rates - std_firing_rates, mean_firing_rates + std_firing_rates, alpha=0.3)
    
    ax.set_title("Mean FI Curve with Standard Deviation")
    if x_axis is None:
        ax.set_xlabel('Current Injection (nA)')
    else:
        ax.set_xlabel(x_axis)
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlim(XLIM)
    ax.legend()

    plt.tight_layout()
    plt.show()

    if save:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        fig.savefig(os.path.join(save_directory, "Mean_FI.png"), dpi=fig.dpi)

def sort_amplitudes_and_firing_rates(data):
    '''Sorts F/I by cell and seed'''
    sorted_data = {}
    for celltype, seeds_data in data.items():
        sorted_seeds_data = {}
        for seed, (amplitudes, firing_rates) in seeds_data.items():
            sorted_indices = np.argsort(amplitudes)
            sorted_amplitudes = np.array(amplitudes)[sorted_indices]
            sorted_firing_rates = np.array(firing_rates)[sorted_indices]
            sorted_seeds_data[seed] = (sorted_amplitudes.tolist(), sorted_firing_rates.tolist())
        sorted_data[celltype] = sorted_seeds_data
    return sorted_data

def collect_fi_data(sim_directories, get_amp=True):
    '''Collects a single cell's FI data from a list of its simulations'''
    seed_data = {}
    for seed, dirs in sim_directories.items():
        amplitudes = []
        firing_rates = []
        for sim_directory in dirs:
            soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
            parameters = analysis.DataReader.load_parameters(sim_directory)
            if get_amp:
                amplitudes.append(parameters.h_i_amplitude)
            else:
                amplitudes.append(parameters.excFR_increase) #@MARK handle better
            firing_rates.append(analyze_and_log(soma_spikes, parameters))
        seed_data[seed] = (amplitudes, firing_rates)
    return seed_data

def group_directories_by_prefix(directory_path):
    '''Groupes simulations by cell'''
    grouped_directories = {}
    for folder_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(full_path):
            parts = folder_name.split('_')
            if len(parts) > 2:  # case where basename_seed_ampINT
                celltype = parts[0]
                seed = parts[1]
                if celltype not in grouped_directories:
                    grouped_directories[celltype] = {}
                if seed not in grouped_directories[celltype]:
                    grouped_directories[celltype][seed] = []
                grouped_directories[celltype][seed].append(full_path)
            else:
                raise ValueError(f"Folder name {folder_name} format is incorrect, expected 'basename_seed_ampINT'")
    return grouped_directories