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
XLIM = (0,2)

def calc_input_resistance():
    pass
    # Calculate the input resistance based on voltage from -1 amp simulations

def filter_spike_times(spike_times, start_time, end_time):
    return spike_times[(spike_times >= start_time) & (spike_times <= end_time)]

def analyze_and_log(soma_spikes, parameters, base_name):
    filtered_spike_times = filter_spike_times(soma_spikes, parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration)
    firing_rate = len(filtered_spike_times) * 1000 / parameters.h_i_duration if len(filtered_spike_times) > 0 else 0
    return firing_rate

def collect_fi_data(sim_directories, base_name):
    amplitudes = []
    firing_rates = []
    for sim_directory in sim_directories:
        soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
        parameters = analysis.DataReader.load_parameters(sim_directory)
        amplitudes.append(parameters.h_i_amplitude)
        firing_rates.append(analyze_and_log(soma_spikes, parameters, base_name))
    return amplitudes, firing_rates

def calculate_area_under_curve(amplitudes, firing_rates):
    return simps(firing_rates, amplitudes)

def calculate_rmse_and_r2(amplitudes, firing_rates, reference_amplitudes, reference_firing_rates):
    # Interpolate reference firing rates to match the amplitudes of the current group
    reference_interpolation = np.interp(amplitudes, reference_amplitudes, reference_firing_rates)
    
    rmse = np.sqrt(mean_squared_error(firing_rates, reference_interpolation))
    r2 = r2_score(firing_rates, reference_interpolation)
    
    return rmse, r2

def calculate_statistics(data, reference_data):
    statistics = {}
    reference_area, reference_rmse, reference_r2 = reference_data

    for base_name, (area, rmse, r2) in data.items():
        percent_change_area = ((area - reference_area) / reference_area * 100) if reference_area != 0 else float('inf')
        percent_change_rmse = ((rmse - reference_rmse) / reference_rmse * 100) if reference_rmse != 0 else float('inf')
        percent_change_r2 = ((r2 - reference_r2) / reference_r2 * 100) if reference_r2 != 0 else float('inf')
        statistics[base_name] = (percent_change_area, percent_change_rmse, percent_change_r2)
    
    return statistics

def plot_all_fi_curves(grouped_data, save=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    statistics = {}

    reference_amplitudes, reference_firing_rates = grouped_data.get("Complex", (None, None))
    if reference_amplitudes is None or reference_firing_rates is None:
        raise ValueError("Reference group 'Complex' not found")

    for base_name, data in grouped_data.items():
        amplitudes, firing_rates = data
        label = base_name
        ax.plot(amplitudes, firing_rates, label=label)

        if base_name != "Complex":
            rmse, r2 = calculate_rmse_and_r2(amplitudes, firing_rates, reference_amplitudes, reference_firing_rates)
            statistics[base_name] = (rmse, r2)

    ax.set_title("FI Curve")
    ax.set_xlabel('Current Injection (nA)')
    ax.set_ylabel('Firing Rate (Hz)')
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

def group_directories_by_prefix(directory_path):
    grouped_directories = {}
    for folder_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(full_path):
            prefix = folder_name.split('_')[0]
            if prefix not in grouped_directories:
                grouped_directories[prefix] = []
            grouped_directories[prefix].append(full_path)
    return grouped_directories

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv
    if "-s" in sys.argv:
        save_directory = sys.argv[sys.argv.index("-s") + 1]

    grouped_directories = group_directories_by_prefix(sim_directory)
    all_data = {}
    for base_name, directories in grouped_directories.items():
        try:
            print("Collecting FI curves data for", base_name)
            amplitudes, firing_rates = collect_fi_data(directories, base_name)
            all_data[base_name] = (amplitudes, firing_rates)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    plot_all_fi_curves(all_data, save)

# @DEPRACATING
# import sys
# sys.path.append("../")
# sys.path.append("../Modules/")

# import analysis
# from logger import Logger
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import os
# import traceback

# def calc_input_resistance():
#     pass
#     # need to get the voltage from the -1 amp simulations. Then calculate the difference in voltage from (t = 0+delay/2) and (t = delay+(duration*3/4)) and divide by -1 nA.

# def filter_spike_times(spike_times, start_time, end_time):
#     """
#     Filters spike times to keep only those within a specified time window.

#     Args:
#     spike_times (array-like): Array of spike times.
#     start_time (float): Start of the time window.
#     end_time (float): End of the time window.

#     Returns:
#     array-like: Filtered spike times within the specified time window.
#     """
#     return spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
    
# def analyze_and_log(soma_spikes, parameters, base_name):
#     # Filtering spike times based on h_i_delay and h_i_duration
#     #print(soma_spikes)
#     filtered_spike_times = filter_spike_times(soma_spikes, parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration)
#     #print(filtered_spike_times)
    
#     # Calculate the firing rate based on the filtered spike times and the stimulation period
#     if len(filtered_spike_times) == 0:
#       firing_rate=0
#     else:
#       firing_rate = len(filtered_spike_times) * 1000 / (parameters.h_i_duration)  # Assuming spikes are in ms, and h_i_duration is also in ms
#     #print(firing_rate)
#     # Log the result
#     #logger.log(f"{base_name} Soma firing rate at {parameters.h_i_amplitude}: {round(firing_rate, 2)} Hz")
        
#     return firing_rate

# def plot_fi(sim_directories, base_name):
#     #print(f"Analyzing {base_name} with directories: {sim_directories}")
#     amplitudes=[]
#     firing_rates=[]
#     for sim_directory in sim_directories:
#       soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
#       parameters = analysis.DataReader.load_parameters(sim_directory)
#       amplitudes.append(parameters.h_i_amplitude)
#       firing_rates.append(analyze_and_log(soma_spikes, parameters, base_name))
      
#     plt.figure()
#     plt.plot(amplitudes,firing_rates)
#     plt.show()
#     if save:
#         fig.savefig(os.path.join(sim_directory, f"FI.png"), dpi = fig.dpi)

# def group_directories_by_prefix(directory_path):
#     """
#     Groups directories by the prefix before '_amp' in their names and includes the full path.

#     Args:
#     directory_path (str): The path to the directory containing the folders.

#     Returns:
#     dict: A dictionary with keys as prefixes and values as lists of directory paths.
#     """
#     grouped_directories = {}
#     for folder_name in os.listdir(directory_path):
#         full_path = os.path.join(directory_path, folder_name)
#         if os.path.isdir(full_path):
#             # Extract the part of the folder name before '_amp'
#             prefix = folder_name.split('_amp')[0]
#             if prefix not in grouped_directories:
#                 grouped_directories[prefix] = []
#             grouped_directories[prefix].append(full_path)
#     return grouped_directories

# def collect_fi_data(sim_directories, base_name):
#     amplitudes = []
#     firing_rates = []
#     for sim_directory in sim_directories:
#         soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
#         parameters = analysis.DataReader.load_parameters(sim_directory)
#         amplitudes.append(parameters.h_i_amplitude)
#         firing_rates.append(analyze_and_log(soma_spikes, parameters, base_name))
        
#     return amplitudes, firing_rates
    
# def plot_all_fi_curves(grouped_data):
#     fig = plt.figure(figsize=(10, 6))
#     for base_name, data in grouped_data.items():
#         amplitudes, firing_rates = data
#         plt.plot(amplitudes, firing_rates, label=base_name.split('_')[0])
#     plt.xlabel('Current Injection (nA)')
#     plt.ylabel('Firing Rate (Hz)')
#     plt.title('F/I')
#     plt.legend()
#     plt.show()
#     if save:
#         fig.savefig(os.path.join(sim_directory, f"FI.png"), dpi = fig.dpi)

# #if __name__ == "__main__":
# #    if "-d" in sys.argv:
# #        sim_directory = sys.argv[sys.argv.index("-d") + 1] # Fixed variable name to match usage
# #    else:
# #        raise RuntimeError("Directory not specified")
# #
# #    save = "-s" in sys.argv # (global)
# #
# #    # New logic to group directories and analyze them
# #    grouped_directories = group_directories_by_prefix(sim_directory)
# #    for base_name, directories in grouped_directories.items():
# #        #try:
# #            print("Analyzing FI curves for", base_name)
# #            plot_fi(directories, base_name) # Pass the list of directories and base name
# #        #except Exception as e:
# #        #    print(f"Error processing {base_name}: {e}")

# if __name__ == "__main__":
#     if "-d" in sys.argv:
#         sim_directory = sys.argv[sys.argv.index("-d") + 1]
#     else:
#         raise RuntimeError("Directory not specified")

#     save = "-s" in sys.argv  # (global)

#     grouped_directories = group_directories_by_prefix(sim_directory)
#     all_data = {}  # Collect all data before plotting

#     for base_name, directories in grouped_directories.items():
#         try:
#             print("Collecting FI curves data for", base_name)
#             amplitudes, firing_rates = collect_fi_data(directories, base_name)
#             all_data[base_name] = (amplitudes, firing_rates)
#         except Exception as e:
#             print(f"Error processing {base_name}: {e}")

#     plot_all_fi_curves(all_data)