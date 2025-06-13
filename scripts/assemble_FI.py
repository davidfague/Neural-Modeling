# gather spike data and CI amplitudes and plot fi.

import sys
sys.path.append('..')
sys.path.append('../Modules')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Modules.analysis as analysis


def gather_spike_tables(sim_dirs):
    # Accumulate data in a list of rows
    all_rows = []

    for sim_dir in sim_dirs:
        # Read dSpike_table.csv
        dSpike_path = os.path.join(sim_dir, "dSpike_table.csv")
        if not os.path.exists(dSpike_path):
            print(f"Skipping {sim_dir}, file not found.")
            continue
        df = pd.read_csv(dSpike_path)
        params = analysis.DataReader.load_parameters(sim_dir)

        CI_amplitude = params.h_i_amplitude
        numpy_random_state = params.numpy_random_state

        # Add sim_dir and CI_amplitude columns
        df['sim_dir'] = sim_dir
        df['CI_amplitude'] = CI_amplitude
        df['numpy_random_state'] = numpy_random_state
        
        # Append to master list
        all_rows.append(df)

    # Combine all rows into one DataFrame
    all_data = pd.concat(all_rows, ignore_index=True)
    return all_data

def plot_fi(data):
        # Scatter plot: CI_amplitude vs. Soma_Spike_Rate
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(data['CI_amplitude'], data['Soma_Spike_Rate'])
    plt.scatter(data['CI_amplitude'], data['Soma_Spike_Rate'])
    plt.xlabel('CI_amplitude')
    plt.ylabel('Soma_Spike_Rate')
    plt.title('Soma_Spike_Rate vs. CI_amplitude')
    # plt.savefig(os.path.join(sims_dir, 'Soma FR/I.png')) # don't need because it is one figure.

    # Scatter plot: CI_amplitude vs. Total_CA_Spikes
    plt.subplot(1, 2, 2)
    plt.plot(data['CI_amplitude'], data['Total_CA_Spikes'])
    plt.scatter(data['CI_amplitude'], data['Total_CA_Spikes'])
    plt.xlabel('CI_amplitude')
    plt.ylabel('Total_CA_Spikes')
    plt.title('Apical Total_CA_Spikes vs. CI_amplitude')

    plt.suptitle(sims_dir.split('-')[-1].replace('_', ' '), fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(sims_dir, 'Soma and Ca2+ FR_I.png'))
    # plt.show()

def get_spike_rates_and_plot_fi(sim_dirs):
    all_data = gather_spike_tables(sim_dirs)
    apical_data = all_data[all_data['Segment_Type'] == 'apic']
    plot_fi(apical_data)

if __name__ == "__main__":

    sims_dir = '/home/drfrbc/Neural-Modeling/simulations/2025-06-13-12-26-pink_and_delay_no_clustering_fi' # folder containing simulation folders

    sim_dirs = [os.path.join(sims_dir, sim_dir) for sim_dir in os.listdir(sims_dir)] # every directory in sims_dir

    get_spike_rates_and_plot_fi(sim_dirs)

    print(f'Finished all simulations')