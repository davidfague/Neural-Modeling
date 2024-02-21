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

# TODO: 
# (1) control for bursts

# https://github.com/dbheadley/InhibOnDendComp/blob/master/src/mean_dendevt.py
def _plot_sta(sta, quantiles, title, xlabel_spike_type, ylabel_ed_from, limit=True) -> plt.figure:
    x_ticks = np.arange(0, 100, 5)
    x_tick_labels = ['{}'.format(i) for i in np.arange(-50, 50, 5)]
    
    fig = plt.figure(figsize=(10, 5))
    
    if limit:
      # Calculate mean and standard deviation of sta
      mean_val = np.mean(sta)
      std_val = np.std(sta)
  
      # Calculate limits as within 95% of the mean
      lower_limit = mean_val - 0.99 * std_val
      upper_limit = mean_val + 0.99 * std_val
  
      # Set vmin and vmax to the calculated limits
      plt.imshow(sta, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=lower_limit, vmax=upper_limit)
    else:
      plt.imshow(sta, cmap = sns.color_palette("coolwarm", as_cmap = True))
  
    plt.title(title)
    plt.xticks(ticks=x_ticks - 0.5, labels=x_tick_labels)
    plt.xlabel(f'Time w.r.t. {xlabel_spike_type} spikes (ms)')
    plt.yticks(ticks=np.arange(11) - 0.5, labels=np.round(quantiles, 3))
    plt.ylabel(f"Elec. dist. quantile (from {ylabel_ed_from})")
    plt.colorbar(label="average of each binned segment's percent change from mean spike rate during entire interval")
    return fig

def _compute_sta_for_each_train_in_a_list(list_of_trains, spikes) -> np.ndarray:

    parameters = analysis.DataReader.load_parameters(sim_directory)

    stas = []
    for train in list_of_trains:
        if len(train) == 0: 
            stas.append(np.zeros((1, 100)))
            continue
        cont_train = np.zeros(parameters.h_tstop)
        cont_train[train] = 1

        # Skip spikes that are in the beginning of the trace
        cont_train[:parameters.skip] = 0
        sta = analysis.SummaryStatistics.spike_triggered_average(cont_train.reshape((1, -1)), spikes, 100)

        # Normalize by average
        sta = (sta - np.mean(cont_train)) / (np.mean(cont_train) + 1e-15) * 100 # percent change from mean
        stas.append(sta)

    stas = np.concatenate(stas)
    return stas

def _map_stas_to_quantiles_and_plot(
        sta, 
        spikes, 
        section,
        elec_dist_from,
        title,
        xlabel_spike_type, 
        indexes = None) -> None:
    
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

    sta_binned = analysis.SummaryStatistics.bin_matrix_to_quantiles(
        matrix = sta,
        quantiles = quantiles, 
        var_to_bin = elec_dist
    )

    fig = _plot_sta(sta_binned, quantiles, title, xlabel_spike_type, elec_dist_from)
    plt.show()
    if save:
        fig.savefig(os.path.join(sim_directory, f"{elec_dist_from}_{title}_wrt{xlabel_spike_type}.png"), dpi = fig.dpi)

def _analyze_Na():

    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    v = analysis.DataReader.read_data(sim_directory, "v")
    
    Na_spikes = []
    for i in range(len(gnaTa)):
        spikes, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], 0.001 / 1000, soma_spikes, 2, v[i], v[0])
        Na_spikes.append(spikes)

    sta = _compute_sta_for_each_train_in_a_list(Na_spikes, soma_spikes)
    #plt.imshow(sta)
    #plt.show()
    
    for section in ["apic", "dend"]:
        seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
        indexes = seg_data[seg_data["section"] == section].index
        ica = analysis.DataReader.read_data(sim_directory, "ica")
        for elec_dist in ["soma", "nexus"]:
            try:
                _map_stas_to_quantiles_and_plot(
                    sta = sta, 
                    spikes = Na_spikes, 
                    section = section,
                    elec_dist_from = elec_dist,
                    title = f"{section}-Na",
                    xlabel_spike_type = "soma")
            except:
                print(section, elec_dist)
                print(traceback.format_exc())
                continue
                
# ADDED NA TO NMDA AND Ca
            try:
                # Ca
                Ca_spikes = []
                for i in indexes:
                    left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                    Ca_spikes.extend(left_bounds)
                Ca_spikes = np.sort(np.unique(Ca_spikes))
    
                sta = _compute_sta_for_each_train_in_a_list(Na_spikes, Ca_spikes)
                for elec_dist in ["soma", "nexus"]:
                    try:
                        _map_stas_to_quantiles_and_plot(
                            sta = sta,
                            spikes = Na_spikes, 
                            section = section,
                            elec_dist_from = elec_dist,
                            title = f"{section}-Na",
                            xlabel_spike_type = "Ca",
                            indexes = indexes)
                    except:
                        print(section, elec_dist)
                        print(traceback.format_exc()) 
                        continue
            except:
                print(section)
                print(traceback.format_exc()) 
                pass
                
            try:
                # NMDA
                inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")
                NMDA_spikes = []
                for i in indexes:
                    left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                    NMDA_spikes.extend(left_bounds)
                NMDA_spikes = np.sort(np.unique(NMDA_spikes))
    
                sta = _compute_sta_for_each_train_in_a_list(Na_spikes, NMDA_spikes)
                for elec_dist in ["soma", "nexus"]:
                    try:
                        _map_stas_to_quantiles_and_plot(
                            sta = sta,
                            spikes = NMDA_spikes, 
                            section = section,
                            elec_dist_from = elec_dist,
                            title = f"{section}-Na",
                            xlabel_spike_type = "NMDA",
                            indexes = indexes)
                    except:
                        print(section, elec_dist)
                        print(traceback.format_exc()) 
                        continue
            except:
                print(section)
                print(traceback.format_exc()) 
                pass

def _analyze_Ca():

    lowery = 500
    uppery = 1500

    v = analysis.DataReader.read_data(sim_directory, "v")
    ica = analysis.DataReader.read_data(sim_directory, "ica")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")

    for section in ["apic", "dend"]:
        try:
            seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
            indexes = seg_data[(seg_data["section"] == section) & (seg_data["pc_1"] > lowery) & (seg_data["pc_1"] < uppery)].index

            Ca_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                Ca_spikes.append(left_bounds)
            
            sta = _compute_sta_for_each_train_in_a_list(Ca_spikes, soma_spikes)
            for elec_dist in ["soma", "nexus"]:
                try:
                    _map_stas_to_quantiles_and_plot(
                        sta = sta,
                        spikes = Ca_spikes, 
                        section = section,
                        elec_dist_from = elec_dist,
                        title = f"{section}-Ca",
                        xlabel_spike_type = "soma",
                        indexes = indexes)
                except:
                    print(section, elec_dist)
                    print(traceback.format_exc())
                    continue
        except:
            print(section)
            print(traceback.format_exc()) 
            continue

def _analyze_NMDA():

    v = analysis.DataReader.read_data(sim_directory, "v")
    inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")
    ica = analysis.DataReader.read_data(sim_directory, "ica")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")

    for section in ["apic", "dend"]:

        seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
        indexes = seg_data[seg_data["section"] == section].index

        try:
            NMDA_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
                NMDA_spikes.append(left_bounds)
        except:
            print(section)
            print(traceback.format_exc()) 
            continue # There are no NMDA spikes

        try:
            # Soma
            sta = _compute_sta_for_each_train_in_a_list(NMDA_spikes, soma_spikes)
            for elec_dist in ["soma", "nexus"]:
                try:
                    _map_stas_to_quantiles_and_plot(
                        sta = sta,
                        spikes = NMDA_spikes, 
                        section = section,
                        elec_dist_from = elec_dist,
                        title = f"{section}-NMDA",
                        xlabel_spike_type = "soma",
                        indexes = indexes)
                except:
                    print(section, elec_dist)
                    print(traceback.format_exc()) 
                    continue
        except:
            print(section, elec_dist)
            print(traceback.format_exc()) 
            pass
        
        try:
            # Ca
            Ca_spikes = []
            for i in indexes:
                left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
                Ca_spikes.extend(left_bounds)
            Ca_spikes = np.sort(np.unique(Ca_spikes))

            sta = _compute_sta_for_each_train_in_a_list(NMDA_spikes, Ca_spikes)
            for elec_dist in ["soma", "nexus"]:
                try:
                    _map_stas_to_quantiles_and_plot(
                        sta = sta,
                        spikes = NMDA_spikes, 
                        section = section,
                        elec_dist_from = elec_dist,
                        title = f"{section}-NMDA",
                        xlabel_spike_type = "Ca",
                        indexes = indexes)
                except:
                    print(section, elec_dist)
                    print(traceback.format_exc()) 
                    continue
        except:
            print(section)
            print(traceback.format_exc()) 
            pass

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

        sta = _compute_sta_for_each_train_in_a_list(spikes, wrt_spikes)
        _map_stas_to_quantiles_and_plot(
            sta=sta,
            spikes=spikes,
            section=section,
            elec_dist_from=elec_dist,
            title=f"{section}-{spike_type}",
            xlabel_spike_type=wrt_spike_type,
            indexes=indexes
        )
        print(f"SUCCESS analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:")        
    except Exception as e:
        print(f"Error analyzing {spike_type} spikes w.r.t. {wrt_spike_type} spikes in {section} section, {elec_dist} distance:")
        print(traceback.format_exc())
        print(str(e))

def analyze_all_spike_relationships(sim_directory):
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
                    _analyze_spike_relationships(sim_directory, spike_type, wrt_spike_type, section, elec_dist)



if __name__ == "__main__":

    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError

    # Save figures or just show them
    save = "-s" in sys.argv # (global)

    logger = Logger()

    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    parameters = analysis.DataReader.load_parameters(sim_directory)
    logger.log(f"Soma firing rate: {round(soma_spikes.shape[1] * 1000 / parameters.h_tstop, 2)} Hz")

    try:
        logger.log("Analyzing all spike relationships.")
        analyze_all_spike_relationships(sim_directory)
    except Exception:
        print(traceback.format_exc())

#    try:
#        logger.log("Analyzing Na.")
#        _analyze_Na()
#    except Exception:
#        print(traceback.format_exc())
#    
#    try:
#        logger.log("Analyzing Ca.")
#        _analyze_Ca()
#    except Exception:
#        print(traceback.format_exc())
#
#    try:
#        logger.log("Analyzing NMDA.")
#        _analyze_NMDA()
#    except Exception:
#        print(traceback.format_exc())

