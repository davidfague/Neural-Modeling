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

# https://github.com/dbheadley/InhibOnDendComp/blob/master/src/mean_dendevt.py
def _plot_sta(
          sta, 
          quantiles, 
          title,
          clipping_values = (1, 99)) -> plt.figure:
    
    x_ticks = np.arange(0, 50, 5)
    x_tick_labels = ['{}'.format(i) for i in np.arange(-50, 50, 10)]
     
    fig = plt.figure(figsize = (10, 5))
    plt.imshow(
            sta, 
            cmap = sns.color_palette("coolwarm", as_cmap = True), 
            vmin = clipping_values[0], 
            vmax = clipping_values[1])
    plt.title(title)
    plt.xticks(ticks = x_ticks - 0.5, labels = x_tick_labels)
    plt.xlabel('Time (ms)')
    plt.yticks(ticks = np.arange(11) - 0.5, labels = np.round(quantiles, 3))
    plt.ylabel("Edge Quantile")
    plt.colorbar(label = 'STA')
    return fig

def _compute_sta_for_each_train_in_a_list(list_of_trains) -> np.ndarray:
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    parameters = analysis.DataReader.load_parameters(sim_directory)

    stas = []
    for train in list_of_trains:
        if len(train) == 0: 
            stas.append(np.zeros((1, 50)))
            continue
        cont_train = np.zeros(parameters.h_tstop)
        cont_train[train] = 1
        sta = analysis.SummaryStatistics.spike_triggered_average(cont_train.reshape((1, -1)), soma_spikes, 50)
        stas.append(sta)

    stas = np.concatenate(stas)
    return stas

def _map_stas_to_quantiles_and_plot(sta, spikes, elec_dist_var, title, indexes = None) -> None:
    elec_dist = pd.read_csv(os.path.join(sim_directory, "elec_distance_nexus.csv"))
    morph = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    if indexes is not None:
        elec_dist = elec_dist.iloc[indexes, :]
        morph = morph.iloc[indexes, :]

    quantiles = analysis.SummaryStatistics.get_quantiles_based_on_elec_dist(
        morph = morph,
        elec_dist = elec_dist,
        spikes = spikes,
        elec_dist_var = elec_dist_var
    )

    sta_binned = analysis.SummaryStatistics.bin_matrix_to_quantiles(
        matrix = sta, 
        quantiles = quantiles, 
        var_to_bin = elec_dist
    )

    fig = _plot_sta(sta_binned, quantiles, title)
    plt.show()
    if save:
        fig.savefig(os.path.join(sim_directory, f"{title}.png"), dpi = fig.dpi)

def _analyze_Na():

    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    
    Na_spikes = []
    for i in range(len(gnaTa)):
        spikes, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], 0.001 / 1000, soma_spikes, 2)
        Na_spikes.append(spikes)

    sta = _compute_sta_for_each_train_in_a_list(Na_spikes)
    _map_stas_to_quantiles_and_plot(sta, Na_spikes, "dend", "Na-dend")
    _map_stas_to_quantiles_and_plot(sta, Na_spikes, "apic", "Na-apic")

def _analyze_Ca():

    lowery = 500
    uppery = 1500

    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    indexes = seg_data[(seg_data["section"] == "apic") & (seg_data["pc_1"] > lowery) & (seg_data["pc_1"] < uppery)].index

    v = analysis.DataReader.read_data(sim_directory, "v")
    ica = analysis.DataReader.read_data(sim_directory, "ica")

    Ca_spikes = []
    for i in indexes:
        left_bounds, _, _ = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
        Ca_spikes.append(left_bounds)
    
    sta = _compute_sta_for_each_train_in_a_list(Ca_spikes)
    _map_stas_to_quantiles_and_plot(sta, Ca_spikes, "apic", "Ca-apic", indexes)

def _analyze_NMDA():

    seg_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    indexes = seg_data[(seg_data["section"] == "apic") | (seg_data["section"] == "dend")].index

    v = analysis.DataReader.read_data(sim_directory, "v")
    inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")

    NMDA_spikes = []
    for i in indexes:
        left_bounds, _, _ = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
        NMDA_spikes.append(left_bounds)

    sta = _compute_sta_for_each_train_in_a_list(NMDA_spikes)
    _map_stas_to_quantiles_and_plot(sta, NMDA_spikes, "dend", "NMDA-dend", indexes)
    _map_stas_to_quantiles_and_plot(sta, NMDA_spikes, "apic", "NMDA-apic", indexes)

if __name__ == "__main__":

    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError

    # Save figures or just show them
    save = "-s" in sys.argv # (global)

    logger = Logger()

    try:
        logger.log("Analyzing Na.")
        _analyze_Na()
    except Exception:
        print(traceback.format_exc())
    
    try:
        logger.log("Analyzing Ca.")
        _analyze_Ca()
    except Exception:
        print(traceback.format_exc())

    try:
        logger.log("Analyzing NMDA.")
        _analyze_NMDA()
    except Exception:
        print(traceback.format_exc())


