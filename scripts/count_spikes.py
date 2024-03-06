import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def _analyze_Na():
    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    v = analysis.DataReader.read_data(sim_directory, "v")
    # ina = -1 * analysis.DataReader.read_data(sim_directory, "ina_NaTa_t.h5")
    segment_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    
    threshold = 0.001 / 1000
    durations = []
    Na_spikes = []
    for i in range(len(gnaTa)):
        spikes, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], threshold, soma_spikes, 2, v[i], v[0])
        if len(spikes) < 0:
            durations.append(0)
            continue
        _, downward_crossing = analysis.VoltageTrace.get_crossings(gnaTa[i], threshold)
        durations.append(analysis.VoltageTrace.get_duration(spikes, downward_crossing))
        Na_spikes.append(spikes)

    total_spikes = 0
    spikes_per_segment_per_micron = []
    for i in range(len(Na_spikes)):
        total_spikes += len(Na_spikes[i])
        spikes_per_segment_per_micron.append(len(Na_spikes[i]) / segment_data.loc[i, "L"])

    print(f"Avg spikes per segment: {total_spikes / len(Na_spikes)}")
    print(f"Avg spikes per segment per ms: {total_spikes / len(Na_spikes) / len(gnaTa[0])}")
    print(f"Avg spikes per micron: {np.mean(spikes_per_segment_per_micron)}")
    print(f"Avg spikes per segment per ms: {np.mean(spikes_per_segment_per_micron) / len(gnaTa[0])}")

    out = []
    for i in range(len(gnaTa)):
        # charges = []
        # for idx, spike in enumerate(Na_spikes[i]):
            # charges.append(np.sum(ina[i][int(spike) : int(spike) + int(durations[i][idx])]))
        
        out.append(
            [
                gnaTa[i][Na_spikes[i].flatten().astype(int)].flatten().tolist(), 
                durations[i], 
                [len(Na_spikes[i]) / total_spikes * 100] * len(Na_spikes)
            ]
        )
    gnas = []
    durs = []
    vs = []
    for i in range(len(out)):
        gnas.extend(out[i][0])
        durs.extend(out[i][1])
        vs.extend(out[i][2])

    gnaTa_quantiles = np.quantile(gnaTa.flatten()[~np.isnan(gnaTa.flatten())], np.arange(0, 1.05, 0.05))
    duration_quantiles = np.quantile([item for row in durations for item in row], np.arange(0, 1.05, 0.05))
    
    matrix = np.zeros((20, 20))
    for i in range(len(gnaTa_quantiles) - 1):
        for j in range(len(duration_quantiles) - 1):
            inds = np.where((gnas > gnaTa_quantiles[i]) & 
                            (gnas < gnaTa_quantiles[i + 1]) &
                            (durs > duration_quantiles[j]) & 
                            (durs < duration_quantiles[j + 1]))[0]
            matrix[i, j] = np.mean(np.array(vs)[inds])
    plt.imshow(matrix.T, origin = 'lower')
    plt.xlabel("Duration (ms)")
    plt.ylabel("Conductance (?)")
    plt.colorbar(label = 'Percentage of events')
    plt.show()

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError
    
    _analyze_Na()