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

def _compute_spike_statistics(spikes, T, row_name, ben = False):
    if ben == True:
        segment_data = pd.read_csv(os.path.join(sim_directory, "Segments.csv"))
    else:
        segment_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))

    # Get spike statistics
    total_spikes = 0
    spikes_per_segment_per_micron = []
    for i in range(len(spikes)):
        total_spikes += len(spikes[i])
        if ben == True:
            spikes_per_segment_per_micron.append(len(spikes[i]) / segment_data.loc[i, "Section_L"])
        else:
            spikes_per_segment_per_micron.append(len(spikes[i]) / segment_data.loc[i, "L"])

    spike_stats = pd.DataFrame(
        {
            "Total spikes": total_spikes,
            "Avg spikes per segment": np.round(total_spikes / len(spikes), 4),
            "Avg spikes per segment per ms": np.round(total_spikes / len(spikes) / T, 4),
            "Avg spikes per micron": np.round(np.mean(spikes_per_segment_per_micron), 4),
            "Avg spikes per micron per ms": np.round(np.mean(spikes_per_segment_per_micron) / T, 6)
        },
        index = [row_name]
    )
    return spike_stats


def _analyze_Na(ben):

    # Read data
    if ben == True:
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

    print(len(v[0]))
    
    # Compute spikes
    threshold = 0.001 / 1000
    durations = []
    Na_spikes = []
    for i in range(len(gnaTa)):
        spikes, _ = analysis.VoltageTrace.get_Na_spikes(gnaTa[i], threshold, soma_spikes, 2, v[i], v[0])
        if len(spikes) < 0:
            durations.append(0)
            continue
        _, downward_crossing = analysis.VoltageTrace.get_crossings(gnaTa[i], threshold)
        dur = analysis.VoltageTrace.get_duration(spikes, downward_crossing)
        durations.append(dur)
        Na_spikes.append(spikes)

    # Make a duration vs conductance hist
    all_g = []; all_duration = []
    for i in range(len(gnaTa)):
        for spike_start, duration in zip(Na_spikes[i], durations[i]):
            g = gnaTa[i][int(spike_start[0]) : int(spike_start[0] + duration)]
            all_g.append(np.sum(g) * 1000) # convert to mS / cm2
            all_duration.append(duration)


    # Filter g and duration
    # lt_g = np.median(all_g) - np.std(all_g); ut_g = np.median(all_g) + np.std(all_g)
    # lt_dur = np.median(all_duration) - np.std(all_duration); ut_dur = np.median(all_duration) + np.std(all_duration)
    lt_g = 0; ut_g = 14
    lt_dur = 0; ut_dur = 10

    filtered_g = []
    filtered_dur = []
    for i in range(len(all_g)):
        if (all_g[i] > lt_g) and (all_g[i] < ut_g) and (all_duration[i] > lt_dur) and (all_duration[i] < ut_dur):
            filtered_g.append(all_g[i])
            filtered_dur.append(all_duration[i])

    H, yedges, xedges = np.histogram2d(filtered_g, filtered_dur, bins = 20)
    H = H / np.sum(H) * 100
    
    spike_stats = _compute_spike_statistics(Na_spikes, len(gnaTa[0]), "Na", ben)

    return H, yedges, xedges, spike_stats

def _analyze_Ca(ben):

    # Read data
    if ben == True:
        with h5py.File(os.path.join(sim_directory, "v_report.h5"), 'r') as file:
            v = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
        with h5py.File(os.path.join(sim_directory, "Ca_HVA.ica_report.h5"), 'r') as file:
            ica = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
    else:
        v = analysis.DataReader.read_data(sim_directory, "v")
        ica = analysis.DataReader.read_data(sim_directory, "ica")

    # Compute spikes
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

    # Filter charge and duration
    # lt_charge = np.median(charge) - np.std(charge); ut_charge = np.median(charge) + np.std(charge)
    # lt_dur = np.median(durations) - np.std(durations); ut_dur = np.median(durations) + np.std(durations)
    lt_charge = 0.1; ut_charge = 2
    lt_dur = 20; ut_dur = 70

    filtered_charge = []
    filtered_durations = []
    for i in range(len(durations)):
        if (charge[i] > lt_charge) and (charge[i] < ut_charge) and (durations[i] > lt_dur) and (durations[i] < ut_dur):
            filtered_charge.append(charge[i])
            filtered_durations.append(durations[i])

    H, yedges, xedges = np.histogram2d(filtered_charge, filtered_durations, bins = 20)
    H = H / np.sum(H) * 100
    
    spike_stats = _compute_spike_statistics(Ca_spikes, len(ica[0]), "Ca", ben)

    return H, yedges, xedges, spike_stats

def _analyze_NMDA(ben):
    # Read data
    if ben == True:
        with h5py.File(os.path.join(sim_directory, "v_report.h5"), 'r') as file:
            v = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
        with h5py.File(os.path.join(sim_directory, "inmda_report.h5"), 'r') as file:
            inmda = np.array(file["report"]["biophysical"]["data"]).T[:, ::int(1/dt)]
    else:
        v = analysis.DataReader.read_data(sim_directory, "v")
        inmda = analysis.DataReader.read_data(sim_directory, "i_NMDA")

    # Compute spikes
    charge = []
    durations = []
    NMDA_spikes = []
    for i in range(len(v)):
        left_bounds, right_bounds, sum_current = analysis.VoltageTrace.get_NMDA_spikes(v[i], -40, inmda[i])
        NMDA_spikes.append(left_bounds)
        dur = np.array(right_bounds) - np.array(left_bounds)
        durations.extend(dur.flatten().tolist())
        charge.extend(sum_current)

    durations = np.array(durations)
    charge = -np.array(charge)

    # Filter charge and duration
    # lt_charge = np.median(charge) - np.std(charge); ut_charge = np.median(charge) + np.std(charge)
    # lt_dur = np.median(durations) - np.std(durations); ut_dur = np.median(durations) + np.std(durations)
    lt_charge = 0.1; ut_charge = 2
    lt_dur = 20; ut_dur = 150

    filtered_charge = []
    filtered_durations = []
    for i in range(len(durations)):
        if (charge[i] > lt_charge) and (charge[i] < ut_charge) and (durations[i] > lt_dur) and (durations[i] < ut_dur):
            filtered_charge.append(charge[i])
            filtered_durations.append(durations[i])

    H, yedges, xedges = np.histogram2d(filtered_charge, filtered_durations, bins = 20)
    H = H / np.sum(H) * 100
    
    spike_stats = _compute_spike_statistics(NMDA_spikes, len(inmda[0]), "NMDA", ben)
    
    return H, yedges, xedges, spike_stats

def _analyze_all_and_plot(ben = False):
    results = [out for out in [_analyze_Na(ben), _analyze_NMDA(ben), _analyze_Ca(ben)]]
    stats_df = pd.concat([results[i][3] for i in range(3)], axis = 0)

    fig, ax = plt.subplot_mosaic(
        [['left', 'center', 'right'],
         ['bottom', 'bottom', 'bottom']],
        constrained_layout = False)

    for i, axname in enumerate(['left', 'center', 'right']):
        im = ax[axname].pcolormesh(results[i][2], results[i][1], results[i][0], vmin = 0, vmax = 5)
        cbar = fig.colorbar(im, ax = ax[axname])
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
    tab = table(ax['bottom'], stats_df, loc = 'center', fontsize = 50)
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)

    plt.show()

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError
    
    _analyze_all_and_plot(ben = True)