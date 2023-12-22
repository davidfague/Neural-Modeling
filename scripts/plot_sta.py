import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_stas(
          sta, 
          quantiles, 
          title, 
          x_ticks, 
          x_tick_labels, 
          clipping_values = (1, 99)) -> plt.figure:
     
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
    # https://github.com/dbheadley/InhibOnDendComp/blob/master/src/mean_dendevt.py
    plt.colorbar(label = 'Percent Change from Mean')
    return fig

if __name__ == "__main__":

    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
    else:
        raise RuntimeError
    
    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes.h5")

    Na_spikes = []
    for i in range(len(gnaTa)):
        spikes, backpropAP = analysis.get_Na_spikes(gnaTa[i], 0.001 / 1000, soma_spikes, 2)
        Na_spikes.append(spikes)

    elec_dist = pd.read_csv(os.path.join(sim_directory, "elec_distance_nexus.csv"))
    morph = pd.read_csv(os.path.join(sim_directory, "segments_by_morphology.csv"))

    quantiles_dend = analysis.get_quantiles_based_on_elec_dist(
        morph = morph,
        elec_dist = elec_dist,
        spikes = Na_spikes,
        elec_dist_var = "dend"
    )

    quantiles_apic = analysis.get_quantiles_based_on_elec_dist(
        morph = morph,
        elec_dist = elec_dist,
        spikes = Na_spikes,
        elec_dist_var = "apic"
    )

    stas = []
    for train in Na_spikes:
        if len(train) == 0: 
            stas.append(np.zeros((1, 50)))
            continue
        cont_train = np.zeros(2000)
        cont_train[train] = 1
        sta = analysis.spike_triggered_average(cont_train.reshape((1, -1)), soma_spikes, 50)
        stas.append(sta)

    stas = np.concatenate(stas)
    
    sta_dend = analysis.bin_matrix_to_quantiles(matrix = stas, quantiles = quantiles_dend, var_to_bin = elec_dist)
    sta_apic = analysis.bin_matrix_to_quantiles(matrix = stas, quantiles = quantiles_apic, var_to_bin = elec_dist)

    x_ticks = np.arange(0, 50, 5)
    x_tick_labels = ['{}'.format(i) for i in np.arange(-50, 50, 10)]

    sta_mean = np.mean(sta_dend, axis = 1, keepdims = True)
    normed_sta_dend = (sta_dend - sta_mean) / (np.abs(sta_mean) + 1e-10)

    fig = plot_stas(sta_apic, quantiles_dend, "img", x_ticks, x_tick_labels)
    plt.show()

    # fig.savefig(f'{save_to}', dpi = fig.dpi)