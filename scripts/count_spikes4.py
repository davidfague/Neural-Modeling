import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py

dt = 0.1

def _compute_spike_statistics(spikes, T, row_name, ben=False):
    if ben:
        segment_data = pd.read_csv(os.path.join(sim_directory, "Segments.csv"))
    else:
        segment_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    total_spikes = 0
    spikes_per_segment_per_micron = []
    for i in range(len(spikes)):
        total_spikes += len(spikes[i])
        if ben:
            spikes_per_segment_per_micron.append(len(spikes[i]) / segment_data.loc[i, "Section_L"])
        else:
            spikes_per_segment_per_micron.append(len(spikes[i]) / segment_data.loc[i, "L"])

    spike_stats = pd.DataFrame(
        {
            "Total spikes": total_spikes,
            "Avg spikes per seg": np.round(total_spikes / len(spikes), 4),
            "Avg spikes per seg per ms": np.round(total_spikes / len(spikes) / T, 4),
            "Avg spikes per um": np.round(np.mean(spikes_per_segment_per_micron), 4),
            "Avg spikes per um per ms": np.round(np.mean(spikes_per_segment_per_micron) / T, 6)
        },
        index=[row_name]
    )
    return spike_stats

def _analyze_Na(ben):
    if ben:
        with h5py.File(os.path.join(sim_directory, "NaTa_t.gNaTa_t_report.h5"), 'r') as file:
            gnaTa = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
        with h5py.File(os.path.join(sim_directory, "spikes.h5"), 'r') as file:
            soma_spikes = np.array(file["spikes"]["biophysical"]["timestamps"])
        with h5py.File(os.path.join(sim_directory, "v_report.h5"), 'r') as file:
            v = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
    else:
        gnaTa = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t")
        soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
        v = analysis.DataReader.read_data(sim_directory, "v")

    threshold = 0.003 / 1000 # TODO: CHeck without division by 1000
    durations = []
    Na_spikes = []

    for i in range(len(gnaTa)):
        spikes, spike_ends, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], threshold, soma_spikes, 2, v[i], v[0])
        if len(spikes) < 0:
            durations.append(0)
            continue
        _, downward_crossing = analysis.VoltageTrace.get_crossings(gnaTa[i], threshold)
        dur = analysis.VoltageTrace.get_duration(spikes, downward_crossing)
        durations.append(dur)
        Na_spikes.append(spikes)

    all_g = []; all_duration = []
    for i in range(len(gnaTa)):
        for spike_start, duration in zip(Na_spikes[i], durations[i]):
            g = gnaTa[i][int(spike_start[0]):int(spike_start[0] + duration)]
            all_g.append(np.sum(g) * 1000)
            all_duration.append(duration)

    total_events = len(all_g)
    median_g = np.median(all_g)
    median_duration = np.median(all_duration)

    duration_bins = np.arange(0.15,1.05,0.1)
    mag_bins = 2*np.logspace(-3,-2,num=15)

    H, xedges, yedges = np.histogram2d(all_duration, all_g, bins=[mag_bins, duration_bins])
    events_in_bins = np.sum(H)
    H = H / total_events * 100  # Normalize by total events\
    percentage_included = events_in_bins / total_events * 100

    spike_stats = _compute_spike_statistics(Na_spikes, len(gnaTa[0]), "Na", ben)
    spike_stats["Median duration"] = median_duration
    spike_stats["Median charge or g"] = round(median_g, 4)
    spike_stats["'%' events plotted"] = round(percentage_included, 1)
    # if events_in_bins <1:import pdb;pdb.set_trace()
    return H, xedges, yedges, spike_stats

def _analyze_Ca(ben):
    if ben:
        with h5py.File(os.path.join(sim_directory, "v_report.h5"), 'r') as file:
            v = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
        with h5py.File(os.path.join(sim_directory, "Ca_HVA.ica_report.h5"), 'r') as file:
            ica = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
    else:
        v = analysis.DataReader.read_data(sim_directory, "v")
        ica = analysis.DataReader.read_data(sim_directory, "ica")

    charge = []
    durations = []
    Ca_spikes = []
    for i in range(len(v)):
        left_bounds, right_bounds, sum_current = analysis.VoltageTrace.get_Ca_spikes(v[i], -40, ica[i])
        Ca_spikes.append(left_bounds)
        dur = np.array(right_bounds) - np.array(left_bounds)
        durations.extend(dur.flatten().tolist())
        charge.extend(sum_current)

    durations = np.array(durations)
    charge = -np.array(charge)

    total_events = len(charge)
    median_charge = np.median(charge)
    median_duration = np.median(durations)

    duration_bins = 2*np.logspace(1.1,1.5,num=15)
    mag_bins = np.linspace(0.1,1.4,num=15)

    H, yedges, xedges = np.histogram2d(charge, durations, bins=[mag_bins, duration_bins])
    events_in_bins = np.sum(H)
    H = H / total_events * 100  # Normalize by total events
    percentage_included = events_in_bins / total_events * 100

    spike_stats = _compute_spike_statistics(Ca_spikes, len(ica[0]), "Ca", ben)
    spike_stats["Median duration"] = median_duration
    spike_stats["Median charge or g"] = round(median_charge, 4)
    spike_stats["'%' events plotted"] = round(percentage_included, 1)
    # if events_in_bins <1:import pdb;pdb.set_trace()
    return H, yedges, xedges, spike_stats

def _analyze_NMDA(ben):
    if ben:
        with h5py.File(os.path.join(sim_directory, "v_report.h5"), 'r') as file:
            v = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
        with h5py.File(os.path.join(sim_directory, "inmda_report.h5"), 'r') as file:
            inmda = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
    else:
        v = analysis.DataReader.read_data(sim_directory, "v")
        inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")

    charge = []
    durations = []
    NMDA_spikes = []
    for i in range(len(v)):
        left_bounds, right_bounds, sum_current = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
        NMDA_spikes.append(left_bounds)
        # print(len(right_bounds), len(left_bounds))
        dur = np.array(right_bounds) - np.array(left_bounds)
        durations.extend(dur.flatten().tolist())
        charge.extend(sum_current)

    durations = np.array(durations)
    charge = -np.array(charge)

    total_events = len(charge)

    median_charge = np.median(charge)
    median_duration = np.median(durations)


    duration_bins = 2*np.logspace(1.1,1.8,num=15)
    mag_bins = np.logspace(-1,0.5,num=15)

    H, yedges, xedges = np.histogram2d(charge, durations, bins=[mag_bins, duration_bins])
    events_in_bins = np.sum(H)
    H = H / total_events * 100  # Normalize by total events
    percentage_included = events_in_bins / total_events * 100

    spike_stats = _compute_spike_statistics(NMDA_spikes, len(inmda[0]), "NMDA", ben)
    spike_stats["Median duration"] = median_duration
    spike_stats["Median charge or g"] = round(median_charge, 4)
    spike_stats["'%' events plotted"] = round(percentage_included, 1)

    return H, yedges, xedges, spike_stats

def _analyze_all_and_plot(ben=False):

    results = [
        _analyze_Na(ben),
        _analyze_NMDA(ben),
        _analyze_Ca(ben)
    ]

    stats_df = pd.concat([results[i][3] for i in range(3)], axis=0)

    fig, ax = plt.subplot_mosaic(
        [['left', 'center', 'right'],
         ['bottom', 'bottom', 'bottom']],
        constrained_layout=False)

    for i, axname in enumerate(['left', 'center', 'right']):
        im = ax[axname].pcolormesh(results[i][2], results[i][1], results[i][0])
        cbar = fig.colorbar(im, ax=ax[axname])
        if i == 2:
            cbar.set_label("Percentage of events")
        ax[axname].set_xlabel("Duration (ms)")

        if i == 0:
            ax[axname].set_title("Na spike properties")
            ax[axname].set_ylabel("Conductance (mS / cm2)")

        if i == 1:
            ax[axname].set_title("NMDA spike properties")
            ax[axname].set_ylabel("Charge (nA ms)")

        if i == 2:
            ax[axname].set_title("Ca spike properties")
            ax[axname].set_ylabel("Charge (nA ms)")

    from pandas.plotting import table
    ax["bottom"].axis('off')
    tab = table(ax['bottom'], stats_df, loc='center', fontsize=50)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)

    plt.show()

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
    else:
        raise RuntimeError
    
    _analyze_all_and_plot(ben=False)