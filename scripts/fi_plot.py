import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback

def calc_input_resistance():
    pass
    # need to get the voltage from the -1 amp simulations. Then calculate the difference in voltage from (t = 0+delay/2) and (t = delay+(duration*3/4)) and divide by -1 nA.

def filter_spike_times(spike_times, start_time, end_time):
    """
    Filters spike times to keep only those within a specified time window.

    Args:
    spike_times (array-like): Array of spike times.
    start_time (float): Start of the time window.
    end_time (float): End of the time window.

    Returns:
    array-like: Filtered spike times within the specified time window.
    """
    return spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
    
def analyze_and_log(soma_spikes, parameters, base_name):
    # Filtering spike times based on h_i_delay and h_i_duration
    #print(soma_spikes)
    filtered_spike_times = filter_spike_times(soma_spikes, parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration)
    #print(filtered_spike_times)
    
    # Calculate the firing rate based on the filtered spike times and the stimulation period
    if len(filtered_spike_times) == 0:
      firing_rate=0
    else:
      firing_rate = len(filtered_spike_times) * 1000 / (parameters.h_i_duration)  # Assuming spikes are in ms, and h_i_duration is also in ms
    #print(firing_rate)
    # Log the result
    #logger.log(f"{base_name} Soma firing rate at {parameters.h_i_amplitude}: {round(firing_rate, 2)} Hz")
        
    return firing_rate

def plot_fi(sim_directories, base_name):
    #print(f"Analyzing {base_name} with directories: {sim_directories}")
    amplitudes=[]
    firing_rates=[]
    for sim_directory in sim_directories:
      soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
      parameters = analysis.DataReader.load_parameters(sim_directory)
      amplitudes.append(parameters.h_i_amplitude)
      firing_rates.append(analyze_and_log(soma_spikes, parameters, base_name))
      
    plt.figure()
    plt.plot(amplitudes,firing_rates)
    plt.show()

def group_directories_by_prefix(directory_path):
    """
    Groups directories by the prefix before '_amp' in their names and includes the full path.

    Args:
    directory_path (str): The path to the directory containing the folders.

    Returns:
    dict: A dictionary with keys as prefixes and values as lists of directory paths.
    """
    grouped_directories = {}
    for folder_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(full_path):
            # Extract the part of the folder name before '_amp'
            prefix = folder_name.split('_amp')[0]
            if prefix not in grouped_directories:
                grouped_directories[prefix] = []
            grouped_directories[prefix].append(full_path)
    return grouped_directories


if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # Fixed variable name to match usage
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv # (global)

    # New logic to group directories and analyze them
    grouped_directories = group_directories_by_prefix(sim_directory)
    for base_name, directories in grouped_directories.items():
        #try:
            print("Analyzing FI curves for", base_name)
            plot_fi(directories, base_name) # Pass the list of directories and base name
        #except Exception as e:
        #    print(f"Error processing {base_name}: {e}")
