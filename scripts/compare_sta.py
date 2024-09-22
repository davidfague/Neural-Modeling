import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger
# import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback

def _compute_sta_for_each_train_in_a_list(sim_dir, list_of_trains, spikes, win_length=60) -> np.ndarray:

    parameters = analysis.DataReader.load_parameters(sim_dir)

    stas = []
    if len(list_of_trains) == 0:
        return None
    for train in list_of_trains:
        if len(train) == 0: 
            stas.append(np.zeros((1, win_length)))
            continue
        cont_train = np.zeros(parameters.h_tstop)
        cont_train[train] = 1

        # Skip spikes that are in the beginning of the trace
        cont_train[:parameters.skip] = 0
        sta = analysis.SummaryStatistics.spike_triggered_average(cont_train.reshape((1, -1)), spikes, win_length)

        # Normalize by average
        sta = (sta - np.mean(cont_train)) / (np.mean(cont_train) + 1e-15) * 100 # percent change from mean
        if len(sta.shape) == 1:
            sta = np.reshape(sta, (1, sta.shape[0]))
        stas.append(sta)

    stas = np.concatenate(stas)
    return stas
    # if len(stas) > 0: 
    #     stas = np.concatenate(stas)
    #     return stas
    # else: # no spikes
    #     return None

def _map_stas_to_quantiles(
        sim_dir,
        sta, 
        spikes, 
        section,
        elec_dist_from,
        indexes = None,
        quantiles = None) -> None:
    
    if section not in ["apic", "dend"]: raise ValueError
    if elec_dist_from not in ["soma", "nexus"]: raise ValueError
    
    EDs = pd.read_csv(os.path.join(sim_dir, f"elec_distance_{elec_dist_from}.csv"))
    morph = pd.read_csv(os.path.join(sim_dir, "segment_data.csv"))

    if indexes is not None:
        EDs = EDs.iloc[indexes, :]
        morph = morph.iloc[indexes, :]
    if quantiles is None:
        quantiles = analysis.SummaryStatistics.get_quantiles_based_on_elec_dist(
            morph = morph,
            elec_dist = EDs,
            spikes = spikes,
            section = section
        )

    sta_binned = analysis.SummaryStatistics.bin_matrix_to_quantiles(
        matrix = sta,
        quantiles = quantiles, 
        var_to_bin = EDs
    )

    return quantiles, sta_binned, EDs


def _analyze_spike_relationships(sim_directory, spike_type, wrt_spike_type, section, elec_dist, quantiles = None):
    v = analysis.DataReader.read_data(sim_directory, "v")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    ica = analysis.DataReader.read_data(sim_directory, "ica")
    inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")
    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    indexes = seg_data[seg_data["section"] == section].index

    try:
        if spike_type == "Na":
            spikes = []
            for i in range(len(v)):
                spike_times, spike_ends, bAps = analysis.VoltageTrace.get_Na_spikes(v[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
                spikes.append(spike_times)
        elif spike_type == "Ca":
            spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                spikes.append(left_bounds)
        elif spike_type == "NMDA":
            spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                spikes.append(left_bounds)
        elif spike_type == "soma_spikes":
            spikes = soma_spikes
        else:
            raise ValueError("Invalid spike type")

        if wrt_spike_type == "soma_spikes":
            wrt_spikes = soma_spikes
        elif wrt_spike_type == "Na":
            wrt_spikes = []
            for i in range(len(v)):
                spike_times, spike_ends, bAps = analysis.VoltageTrace.get_Na_spikes(v[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
                wrt_spikes.extend(spike_times)
            wrt_spikes = np.sort(np.unique(wrt_spikes))
            
        elif wrt_spike_type == "Ca":
            wrt_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                wrt_spikes.extend(left_bounds)
            wrt_spikes = np.sort(np.unique(wrt_spikes))
        elif wrt_spike_type == "NMDA":
            wrt_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                wrt_spikes.extend(left_bounds)
            wrt_spikes = np.sort(np.unique(wrt_spikes))
                
        else:
            raise ValueError("Invalid 'with respect to' spike type")

        print(f"sim_directory: {sim_directory}")

        
        # import pdb; pdb.set_trace()
        sta = _compute_sta_for_each_train_in_a_list(sim_directory, spikes, wrt_spikes)
        
        if sta is None:
            return None, None, None, None
        # try:sta = _compute_sta_for_each_train_in_a_list(sim_directory, spikes, wrt_spikes)
        # except Exception as e: print(e); import pdb;pdb.set_trace()
        quantiles, sta_binned, EDs = _map_stas_to_quantiles(
            sim_directory,
            sta=sta,
            spikes=spikes,
            section=section,
            elec_dist_from=elec_dist,
            # title=f"{sim_directory.split('/')[-1]} model {section} {spike_type} spike rate around {wrt_spike_type} spiketimes",
            # xlabel_spike_type=wrt_spike_type,
            # cbar_spike_type=spike_type,
            indexes=indexes,
            quantiles = quantiles
        )
        print(f"SUCCESS analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:") 
        return sta, sta_binned, quantiles, EDs       
    except Exception as e:
        raise(e)
        print(f"Error analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:")
        print(traceback.format_exc())
        print(str(e))
        
def analyze_simulation_directories(grouped_directories, output_dir="../notebooks/STA/Metric/", save=True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger()

    # Parameters
    sections = ["apic", "dend"]
    spike_types = ["Na", "Ca", "NMDA", "soma_spikes"]
    elec_dists = ["soma", "nexus"]
    
    # Dictionary to hold expected STAs from the first directory
    r_squareds = {}
    maes = {}
    mses = {}
    sta_data = {}
    for cell_type,seeds in grouped_directories.items():
        r_squareds[cell_type] = {}
        maes[cell_type] = {}
        mses[cell_type] = {}
        sta_data[cell_type] = {}
        for seed in seeds:
            r_squareds[cell_type][seed] = {}
            maes[cell_type][seed] = {}
            mses[cell_type][seed] = {}
            sta_data[cell_type][seed] = {}

    # Analyze the complex directory for expected STAs
    expected_sim_name = 'Complex'
    expected_sta_sim_directory = grouped_directories[expected_sim_name]

    for seed,sim_folder in expected_sta_sim_directory.items():
        # print(seed, sim_folder)
        parameters = analysis.DataReader.load_parameters(sim_folder)
        sta_data[expected_sim_name][seed] = {}
        for spike_type in spike_types:
            for wrt_spike_type in spike_types:
                for section in sections:
                    for elec_dist in elec_dists:
                        if (elec_dist == "nexus") and ((spike_type != "Ca") or (wrt_spike_type != "Ca")):
                            continue  # Only Ca-related STAs need to be analyzed with nexus electrotonic distance
                        if (section == "dend") and ((spike_type == "Ca") or (wrt_spike_type == "Ca")):
                            continue  # Dend sections should not have Ca spikes
                        if spike_type == "soma_spikes":
                            continue  # Skipping how soma spikes cause other spikes for now
                        sta, sta_binned, quantiles, EDs = _analyze_spike_relationships(sim_folder, spike_type, wrt_spike_type, section, elec_dist)
                        key = f"{spike_type}_{wrt_spike_type}_{section}_{elec_dist}"
                        sta_data[expected_sim_name][seed][key] = {
                            'sta': sta,
                            'sta_binned': sta_binned,
                            'quantiles': quantiles,
                            'EDs': EDs
                        }
    for cell_type,seeds in grouped_directories.items():
        if cell_type != expected_sim_name:
            cell_sims = grouped_directories[cell_type]
            for seed, sim_folder in cell_sims.items():
                for spike_type in spike_types:
                    for wrt_spike_type in spike_types:
                        for section in sections:
                            for elec_dist in elec_dists:
                                if (elec_dist == "nexus") and ((spike_type != "Ca") or (wrt_spike_type != "Ca")):
                                    continue
                                if (section == "dend") and ((spike_type == "Ca") or (wrt_spike_type == "Ca")):
                                    continue
                                if spike_type == "soma_spikes":
                                    continue
                                key = f"{spike_type}_{wrt_spike_type}_{section}_{elec_dist}"
                                try:
                                    observed_sta, observed_sta_binned, quantiles, EDs = _analyze_spike_relationships(sim_folder, spike_type, wrt_spike_type, section, elec_dist, sta_data[expected_sim_name][seed][key]['quantiles'])
                                    sta_data[cell_type][seed][key] = {
                                        'sta': observed_sta,
                                        'sta_binned': observed_sta_binned,
                                        'quantiles': quantiles,
                                        'EDs': EDs
                                    }
                                    r_squared = calculate_r_squared(observed_sta_binned, sta_data[expected_sim_name][seed][key]['sta_binned'])
                                    mae = calculate_mae(observed_sta_binned, sta_data[expected_sim_name][seed][key]['sta_binned'])
                                    mse = calculate_mae(observed_sta_binned, sta_data[expected_sim_name][seed][key]['sta_binned'])
                                    
                                    r_squareds[cell_type][seed][key] = r_squared
                                    maes[cell_type][seed][key] = mae
                                    mses[cell_type][seed][key] = mse
                                    
                                    # if r_squared is not None:
                                    #     logger.log(f"R-squared for {spike_type}, {wrt_spike_type}, {section}, {elec_dist} in {sim_folder}: {r_squared:.4f}")
                                    # else:
                                    #     logger.log(f"Unable to calculate R-squared for {spike_type}, {wrt_spike_type}, {section}, {elec_dist} due to shape mismatch: {observed_sta.shape, expected_stas[key].shape}")
                                except Exception as e:
                                    raise(e)
                                    logger.log(f"Failed to analyze or calculate R-squared for {spike_type}, {wrt_spike_type}, {section}, {elec_dist} due to {str(e)}")

    return sta_data, r_squareds, maes, mses

def calculate_r_squared(observed_sta, expected_sta):
    if observed_sta is None:
        return None
    try:
        assert observed_sta.shape == expected_sta.shape, "STAs must have the same shape"
        observed_values = observed_sta.flatten()
        expected_values = expected_sta.flatten()
        mean_observed = np.mean(observed_values)
        ss_tot = np.sum((observed_values - mean_observed) ** 2)
        ss_res = np.sum((observed_values - expected_values) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-15))
        return r_squared
    except Exception as e:
        raise(e)
    
def calculate_mse(observed_sta, expected_sta):
    if observed_sta is None:
        return None
    try:
        assert observed_sta.shape == expected_sta.shape, "STAs must have the same shape"
        observed_values = observed_sta.flatten()
        expected_values = expected_sta.flatten()
        mse = np.mean((observed_values - expected_values) ** 2)
        return mse
    except Exception as e:
        raise e
    
def calculate_mae(observed_sta, expected_sta):
    if observed_sta is None:
        return None
    try:
        assert observed_sta.shape == expected_sta.shape, "STAs must have the same shape"
        observed_values = observed_sta.flatten()
        expected_values = expected_sta.flatten()
        mae = np.mean(np.abs(observed_values - expected_values))
        return mae
    except Exception as e:
        raise e

def calculate_averages(data_dict, penalty = 1e15):
    """
    Calculate the average values for each cell in the data dictionary.

    Args:
        data_dict (dict): A dictionary with cells as keys and seed dictionaries as values.
                          Each seed dictionary contains parameters and their corresponding values.

    Returns:
        dict: A dictionary with cells as keys and dictionaries of averaged parameter values as values.
    """
    average_values = {}

    for cell, seeds in data_dict.items():
        cell_avg = {}
        num_seeds = len(seeds.keys())

        for seed, values in seeds.items():
            for parameter, value in values.items():
                if parameter not in cell_avg:
                    cell_avg[parameter] = 0
                if value is None:
                    cell_avg[parameter] += penalty # penalized for not having any spikes (cannot form sta)
                else:
                    cell_avg[parameter] += value

        for parameter, total_value in cell_avg.items():
            cell_avg[parameter] /= num_seeds

        average_values[cell] = cell_avg

    return average_values

# Print the average values for r_squareds, maes, and mses
def print_averages(data_dict, name):
    print(f"Average values for {name}:")
    for cell, avg_values in data_dict.items():
        print(cell)
        for parameter, value in avg_values.items():
            print(f"{parameter}: {value}")

def get_all_directories_within(directory):
    try:
        # List all items in the given directory
        items = os.listdir(directory)
        # Filter out only directories and include the original directory in the path
        directories = [os.path.join(directory, item) for item in items]
        return directories
    except Exception as e:
        raise(e)
    
def group_directories_by_cell_and_seed(directories):
    grouped_directories = {}
    
    for folder_name in directories:
        parts = folder_name.split('_')
        if len(parts) == 2:  # case where basename_seed
            celltype = parts[0].split('/')[-1]
            seed = parts[1]
            if celltype not in grouped_directories:
                grouped_directories[celltype] = {}
            if seed not in grouped_directories[celltype]:
                grouped_directories[celltype][seed] = folder_name
        else:
            raise ValueError("Folder name format is incorrect, expected 'basename_seed'")
    return grouped_directories

if __name__ == "__main__":

    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError

    # Save figures or just show them
    save = "-s" in sys.argv # (global)
    if "-s" in sys.argv:
        save_directory = sys.argv[sys.argv.index("-s") + 1]

    logger = Logger()
    
    sim_directories = get_all_directories_within(sim_directory)
    grouped_directories = group_directories_by_cell_and_seed(sim_directories)
    sta_data, r_squareds, maes, mses = analyze_simulation_directories(grouped_directories)

    # try:
    #     logger.log("Analyzing all spike relationships.")
    #     analyze_simulation_directories(sim_directory)
    # except Exception:
    #     print(traceback.format_exc())