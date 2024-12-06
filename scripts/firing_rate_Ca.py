import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger

import statsmodels.api as sm
import statsmodels.formula.api as smf

def filter_spike_times(spike_times, start_time, end_time):
    """
    Filters spike times to keep only those within a specified time window.
    """
    return spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
    
def analyze_and_log(soma_spikes, parameters):
    """
    Calculate the firing rate based on filtered spike times and the stimulation period.
    """
    #filtered_spike_times = filter_spike_times(soma_spikes, parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration)
    if len(soma_spikes) == 0:
        firing_rate = 0
    else:
        firing_rate = len(soma_spikes) * 1000 / parameters.h_tstop  # Assuming spikes are in ms, and h_i_duration is also in ms
    return firing_rate

def group_directories_by_model_and_change(directory_path):
    grouped_directories = {}
    for folder_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(full_path):
            parts = folder_name.split('_')
            # Check if the directory name is for a Baseline model
            if "Baseline" in parts:
                percent_change = "100"  # Treat Baseline as 100%
                model = '_'.join(parts[1:])  # Adjust this according to your directory naming convention if needed
            elif len(parts) >= 3 and parts[1].isdigit():
                percent_change, model = parts[1], '_'.join(parts[2:])
            else:
                print(f"Skipping directory {folder_name} as it does not match expected format.")
                continue

            group_name = f'Ca_{percent_change}_{model}'

            if group_name not in grouped_directories:
                grouped_directories[group_name] = []
            grouped_directories[group_name].append(full_path)
    return grouped_directories



def collect_fi_data(sim_directories):
    """
    Collect firing rate data for a group of simulation directories.
    """
    amplitudes = []
    firing_rates = []
    for sim_directory in sim_directories:
        soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")[0]
        parameters = analysis.DataReader.load_parameters(sim_directory)
        amplitudes.append(parameters.h_i_amplitude)
        firing_rates.append(analyze_and_log(soma_spikes, parameters))
        
    return np.mean(amplitudes), np.mean(firing_rates)  # Use mean amplitude and firing rate for simplicity

def plot_fi_curves_by_ca_change(grouped_data):
    fig = plt.figure(figsize=(10, 6))
    data_for_plotting = {}

    # Organize data by model and ensure percent change is valid
    for base_name, (amplitude, firing_rate) in grouped_data.items():
        _, percent_change, model = base_name.split('_')
        try:
            percent_change = int(percent_change)  # Safeguard against invalid conversion
        except ValueError:
            print(f"Skipping {base_name} due to invalid percent change.")
            continue  # Skip this entry

        if model not in data_for_plotting:
            data_for_plotting[model] = []
        data_for_plotting[model].append((percent_change, firing_rate))

    # Plotting
    for model, data in data_for_plotting.items():
        data = sorted(data, key=lambda x: x[0])  # Sort by Ca percent change
        percent_changes, firing_rates = zip(*data)
        plt.plot(percent_changes, firing_rates, label=model)

    plt.xlabel('Ca2+ Channel Conductance Percentage')
    plt.ylabel('Somatic Firing Rate (Hz)')
    plt.title('Firing Rate vs. Ca')
    plt.legend()
    
    # Set the y-axis minimum limit to 0
    #plt.ylim(bottom=0)
    
    plt.show()
    if save:
        fig.savefig(os.path.join(sim_directory, f"FR_Ca_percent.png"), dpi=fig.dpi)


def perform_statistical_test(data_for_plotting):
    """
    Perform a Linear Mixed Model statistical test to compare firing rates between models
    across different calcium percent changes.

    Args:
    data_for_plotting (dict): A dictionary with models as keys and lists of tuples (Ca percent change, firing rate) as values.
    """
    # Flatten data for LMM
    lmm_data = {
        'Model': [],
        'Ca_Percent_Change': [],
        'Firing_Rate': []
    }
    for model, data_points in data_for_plotting.items():
        for ca_percent, firing_rate in data_points:
            lmm_data['Model'].append(model)
            lmm_data['Ca_Percent_Change'].append(ca_percent)
            lmm_data['Firing_Rate'].append(firing_rate)

    df_lmm = pd.DataFrame(lmm_data)

    # Define and fit the model
    # Model specification includes an interaction between Ca_Percent_Change and Model
    # to see if the slope (effect of Ca_Percent_Change on Firing_Rate) differs by model.
    md = smf.mixedlm("Firing_Rate ~ Ca_Percent_Change * Model", df_lmm, groups=df_lmm["Model"])
    mdf = md.fit()
    print(mdf.summary())


if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv  # (global)

    grouped_directories = group_directories_by_model_and_change(sim_directory)
    all_data = {}  # Collect all data before plotting

    for base_name, directories in grouped_directories.items():
        try:
            print(f"Collecting FI curves data for {base_name}")
            amplitude, firing_rate = collect_fi_data(directories)
            all_data[base_name] = (amplitude, firing_rate)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    plot_fi_curves_by_ca_change(all_data)
    
    # Organize data for plotting into the expected structure for statistical testing
    data_for_stat_test = {}
    for base_name, (amplitude, firing_rate) in all_data.items():
        _, percent_change, model = base_name.split('_')
        if model not in data_for_stat_test:
            data_for_stat_test[model] = []
        data_for_stat_test[model].append((int(percent_change), firing_rate))
    
    perform_statistical_test(data_for_stat_test)
