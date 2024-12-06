# plot_sta adapted to compute R2 between two directories.

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

def _compute_sta_for_each_train_in_a_list(list_of_trains, spikes, win_length=60) -> np.ndarray:

    parameters = analysis.DataReader.load_parameters(sim_directory)

    stas = []
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
        stas.append(sta)

    stas = np.concatenate(stas)
    return stas

def get_quantiles_and_elec_dists(elec_dist_from, indexes, spikes, section, sim_directory):
    if section not in ["apic", "dend"]: raise ValueError
    if elec_dist_from not in ["soma", "nexus"]: raise ValueError

    elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{elec_dist_from}.csv"))
    morph = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    if indexes is not None:
        elec_dist = elec_dist.iloc[indexes, :]
        morph = morph.iloc[indexes, :]

    quantiles = analysis.SummaryStatistics.get_quantiles_based_on_elec_dist(
        morph = morph,
        elec_dist = elec_dist,
        spikes = spikes,
        section = section
    )
    return quantiles, elec_dist

def _map_stas_to_quantiles_and_plot(
        sta, 
        spikes, 
        section,
        elec_dist_from,
        title,
        xlabel_spike_type, 
        cbar_spike_type,
        sim_directory,
        indexes = None) -> None:

    quantiles, elec_dist = get_quantiles_and_elec_dists(elec_dist_from, indexes, spikes, section, sim_directory)

    sta_binned = analysis.SummaryStatistics.bin_matrix_to_quantiles(
        matrix = sta,
        quantiles = quantiles, 
        var_to_bin = elec_dist
    )

    return sta_binned, quantiles, elec_dist

def _analyze_spike_relationships(sim_directory, spike_type, wrt_spike_type, section, elec_dist):
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
                spike_times, _ = analysis.VoltageTrace.get_Na_spikes(v[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
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
                spike_times, _ = analysis.VoltageTrace.get_Na_spikes(v[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
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

        sta = _compute_sta_for_each_train_in_a_list(spikes, wrt_spikes)
        sta_binned, quantiles, EDs = _map_stas_to_quantiles_and_plot(
            sta=sta,
            spikes=spikes,
            section=section,
            elec_dist_from=elec_dist,
            # title=f"{sim_directory.split('/')[-2].split('_')[1]} model {section} {spike_type} spike rate around {wrt_spike_type} spiketimes",
            title=f"{sim_directory.split('/')[-1]} model {section} {spike_type} spike rate around {wrt_spike_type} spiketimes",
            xlabel_spike_type=wrt_spike_type,
            cbar_spike_type=spike_type,
            sim_directory=sim_directory,
            indexes=indexes
        )
        print(f"SUCCESS analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:")
        return sta, sta_binned, quantiles, EDs       
    except Exception as e:
        print(f"Error analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:")
        print(traceback.format_exc())
        print(str(e))
        return None, None, None, None

def analyze_all_spike_relationships(sim_directory):
    sta_data = {}
    sections = ["apic", "dend"]
    spike_types = ["Na", "Ca", "NMDA", "soma_spikes"]
    elec_dists = ["soma", "nexus"]
    for spike_type in spike_types:
        for wrt_spike_type in spike_types:
            for section in sections:
                for elec_dist in elec_dists:
                    if (elec_dist == "nexus") and ((spike_type != "Ca") or (wrt_spike_type != "Ca")):
                        continue  # skip this iteration because only Ca-related STAs need to be analyzed with nexus electrotonic distance
                    if (section == "dend") and ((spike_type == "Ca") or (wrt_spike_type == "Ca")):
                        continue  # skip this iteration because dend sections should not have Ca spikes
                    if spike_type == "soma_spikes":
                        continue  # special case for soma spikes; skipping for now
                    sta, sta_binned, quantiles, EDs = _analyze_spike_relationships(sim_directory, spike_type, wrt_spike_type, section, elec_dist)
                    key = f"{spike_type}_{wrt_spike_type}_{section}_{elec_dist}"
                    sta_data[key] = {
                        'sta': sta,
                        'sta_binned': sta_binned,
                        'quantiles': quantiles,
                        'EDs': EDs
                    }

    return sta_data

def compare_stas(expected_directory, observed_directory):
    expected_sta_data = analyze_all_spike_relationships(expected_directory)
    observed_sta_data = analyze_all_spike_relationships(observed_directory)
    r_squareds = {}
    for key in expected_sta_data.keys():
        quantiles = expected_sta_data[key]['quantiles']
        expected_sta = expected_sta_data[key]['sta_binned']
        # bin observed sta using expected sta quantiles
        if quantiles is not None:
            observed_sta = analysis.SummaryStatistics.bin_matrix_to_quantiles(matrix = observed_sta_data[key]['sta'], quantiles = quantiles, var_to_bin = observed_sta_data[key]['EDs'])
        else:
            continue
        r_squareds[key] = calculate_r_squared(observed_sta, expected_sta)
    return r_squareds

def calculate_r_squared(observed_sta, expected_sta):
    try:
        assert observed_sta.shape == expected_sta.shape, "STAs must have the same shape"
        observed_values = observed_sta.flatten()
        expected_values = expected_sta.flatten()
        mean_observed = np.mean(observed_values)
        ss_tot = np.sum((observed_values - mean_observed) ** 2)
        ss_res = np.sum((observed_values - expected_values) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    except:
        return None