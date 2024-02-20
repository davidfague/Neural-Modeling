import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _analyze_Na():
    gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
    soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
    v = analysis.DataReader.read_data(sim_directory, "v")
    
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

    out = []
    for i in range(len(gnaTa)):
        out.append(
            [
                gnaTa[i][Na_spikes[i].flatten().astype(int)].flatten().tolist(), 
                durations[i], 
                v[i][Na_spikes[i].flatten().astype(int)].flatten().tolist()
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
    plt.imshow(matrix)
    plt.show()

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError
    
    _analyze_Na()