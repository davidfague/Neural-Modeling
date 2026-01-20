import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from Modules.cell_model import plot_morphology
from Modules.cell_model.adjacency import find_branching_seg_with_most_branching_descendants_in_subset_y

def find_nexus_seg(seg_data, adj_matrix):
    apical_segment_indices = []
    y_coords = []
    for i, seg in seg_data.iterrows():
        y_coord = seg.pc_1
        y_coords.append(y_coord)
        if 'apic' in seg.seg:
            apical_segment_indices.append(i)
    nexus_index_in_all_list, _ = find_branching_seg_with_most_branching_descendants_in_subset_y(
        adj_matrix, apical_segment_indices, y_coords)
    return nexus_index_in_all_list

def find_relatives(seg_id, adj_matrix):
    descendant_seg_ids = np.where(adj_matrix[seg_id, :] == 1)[0]
    ascendant_seg_ids = np.where(adj_matrix[:, seg_id] == 1)[0]
    return descendant_seg_ids, ascendant_seg_ids

def compute_axial_currents_btwn_segments(this_seg_id, other_seg_id, v, seg_data):
    # axial current between segments from other_seg to this_seg
    return -(v[this_seg_id] - v[other_seg_id]) / (
        seg_data.loc[this_seg_id, "seg_half_seg_RA"] +
        seg_data.loc[other_seg_id, "seg_half_seg_RA"])

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_and_save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=300)
    plt.close(fig)

def plot_special_segments_and_relatives(seg_data, segments_of_interest, save_dir):
    target_color = 'r*'
    descendant_color = 'b*'
    ascendant_color = 'g*'
    special_indices = []
    special_colors = []
    # Target segments
    for seg_name in segments_of_interest.keys():
        special_indices.append(segments_of_interest[seg_name]['seg_index'])
        special_colors.append(target_color)
    # Descendants
    descendent_seg_ids = np.concatenate([segments_of_interest[seg_name]['descendant_ids'] for seg_name in segments_of_interest.keys()])
    special_indices.extend(descendent_seg_ids)
    special_colors.extend([descendant_color] * len(descendent_seg_ids))
    # Ascendants
    ascendent_seg_ids = np.concatenate([segments_of_interest[seg_name]['ascendant_ids'] for seg_name in segments_of_interest.keys()])
    special_indices.extend(ascendent_seg_ids)
    special_colors.extend([ascendant_color] * len(ascendent_seg_ids))

    fig = plot_morphology.plot_special_segments(seg_data, special_indices, special_colors, title_suffix=" (relatives)")
    plot_and_save(fig, save_dir, "special_segments_with_relatives")

def plot_axial_currents(t, segments_of_interest, v, seg_data, save_dir, sum_basal_for_soma=True):
    for seg_interest_type, seg_interest_data in segments_of_interest.items():
        is_soma = seg_interest_type == 'soma'
        fig, ax = plt.subplots()
        ax.set_title(f"Axial currents from segments into {seg_interest_type}")
        for relative_type in ['ascendant', 'descendant']:
            cmap = plt.cm.Blues if relative_type == 'ascendant' else plt.cm.Reds
            num_segments = len(seg_interest_data[f'{relative_type}_ids'])
            fractions = np.linspace(0.2, 0.9, num_segments) if num_segments > 1 else [0.55]
            if sum_basal_for_soma and is_soma:
                if seg_interest_data[f'{relative_type}_names']:
                    ax.plot(t, (v[0, 100:] / 70) - np.mean(v[0, 100:] / 70), label='soma voltage (adjusted)')
                    indices_to_sum = np.array(['dend' in item for item in seg_interest_data[f'{relative_type}_names']])
                    if indices_to_sum.any():
                        summed_basal = np.sum(np.array(seg_interest_data[f'{relative_type}_currents'])[indices_to_sum], axis=0)[100:]
                        summed_all = np.sum(np.array(seg_interest_data[f'{relative_type}_currents']), axis=0)[100:]
                        ax.plot(t, summed_basal, label='sum basal segments')
                        ax.axhline(np.mean(summed_basal), color='k', linestyle='--', alpha=0.7, label='mean (basal sum)')
                        ax.plot(t, summed_all, label='sum all segments')
                        ax.axhline(np.mean(summed_all), color='gray', linestyle='--', alpha=0.7, label='mean (all sum)')
                    for i, seg_id in enumerate(np.array(seg_interest_data[f'{relative_type}_ids'])):
                        if not indices_to_sum[i]:
                            current_trace = seg_interest_data[f'{relative_type}_currents'][i][100:]
                            label = np.array(seg_interest_data[f'{relative_type}_names'])[i]
                            ax.plot(t, current_trace, label=label)
                            ax.axhline(np.mean(current_trace), color='k', linestyle='--', alpha=0.5)
                    ax.set_ylim([-2, 2])
            else:
                for i, seg_id in enumerate(seg_interest_data[f'{relative_type}_ids']):
                    color = cmap(fractions[i])
                    current_trace = seg_interest_data[f'{relative_type}_currents'][i][100:]
                    label = seg_interest_data[f'{relative_type}_names'][i]
                    ax.plot(t, current_trace, label=label, color=color)
                    ax.axhline(np.mean(current_trace), color=color, linestyle='--', alpha=0.7)
        ax.set_xlim(4750, 4800)
        ax.set_xlabel('Time(ms)')
        ax.set_ylabel('axial current (pA)')
        ax.legend()
        fig.tight_layout()
        plot_and_save(fig, save_dir, f"axial_currents_{seg_interest_type}")

def plot_channel_currents(parameters, segments_of_interest, currents, save_dir):
    for seg_name, seg_info in segments_of_interest.items():
        for current_name, current in currents.items():
            current_to_plot = current[seg_info['seg_index'], :]
            fig, ax = plt.subplots()
            ax.plot(current_to_plot)
            ax.set_title(f"{current_name} {seg_name}")
            ax.set_xlabel("Timesteps")
            fig.tight_layout()
            plot_and_save(fig, save_dir, f"{current_name}_{seg_name}")

def main():
    SIM_DIR = "/home/drfrbc/Neural-Modeling/simulations/2025-06-13-17-10-constant-poisson_no-delay-inh_no-clustering_fi/tuning_Complex_Np5000_amp0.0"  # <--- EDIT THIS LINE
    OUT_DIR = os.path.join(SIM_DIR, "axial_currents")
    ensure_dir(OUT_DIR)

    # Load data
    v = analysis.DataReader.read_data(SIM_DIR, "v")
    seg_data = pd.read_csv(os.path.join(SIM_DIR, "segment_data.csv"))
    parameters = analysis.DataReader.load_parameters(SIM_DIR)
    adj_matrix = np.loadtxt(os.path.join(SIM_DIR, "adj_matrix.txt"))

    # Find segments of interest
    segments_of_interest = {
        'soma': {'seg_index': 0},
        'nexus': {'seg_index': find_nexus_seg(seg_data, adj_matrix)}
    }

    # Print segment names
    for seg_interest_type in segments_of_interest.keys():
        print(f"{seg_interest_type}: {seg_data.iloc[segments_of_interest[seg_interest_type]['seg_index']].seg}")

    # Find relatives
    for seg_interest_type in segments_of_interest.keys():
        descendent_seg_ids, ascendant_seg_ids = find_relatives(segments_of_interest[seg_interest_type]['seg_index'], adj_matrix)
        segments_of_interest[seg_interest_type]['descendant_ids'] = descendent_seg_ids
        segments_of_interest[seg_interest_type]['descendant_names'] = [seg_data.iloc[seg_id].seg for seg_id in descendent_seg_ids]
        segments_of_interest[seg_interest_type]['ascendant_ids'] = ascendant_seg_ids
        segments_of_interest[seg_interest_type]['ascendant_names'] = [seg_data.iloc[seg_id].seg for seg_id in ascendant_seg_ids]

    # Plot special segments and relatives on morphology
    plot_special_segments_and_relatives(seg_data, segments_of_interest, OUT_DIR)

    # Compute axial currents
    for seg_interest_type in segments_of_interest.keys():
        this_seg_id = segments_of_interest[seg_interest_type]['seg_index']
        segments_of_interest[seg_interest_type]['ascendant_currents'] = [
            compute_axial_currents_btwn_segments(this_seg_id, other_seg_id, v, seg_data)
            for other_seg_id in segments_of_interest[seg_interest_type]['ascendant_ids']]
        segments_of_interest[seg_interest_type]['descendant_currents'] = [
            compute_axial_currents_btwn_segments(this_seg_id, other_seg_id, v, seg_data)
            for other_seg_id in segments_of_interest[seg_interest_type]['descendant_ids']]

    # Prepare t axis
    t = np.arange(0, parameters.h_tstop + parameters.h_dt, parameters.h_dt)
    t = t[100:]

    # Plot axial currents
    plot_axial_currents(t, segments_of_interest, v, seg_data, OUT_DIR)

    # Plot channel currents
    currents = {
        'ica': analysis.DataReader.read_data(SIM_DIR, 'ica'),
        'Vm': analysis.DataReader.read_data(SIM_DIR, 'v')
    }
    plot_channel_currents(parameters, segments_of_interest, currents, OUT_DIR)

if __name__ == "__main__":
    main()
