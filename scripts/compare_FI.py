import sys
sys.path.append('..')
sys.path.append('../Modules')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Modules.analysis as analysis

def gather_spike_tables(sim_dirs):
    all_rows = []
    for sim_dir in sim_dirs:
        dSpike_path = os.path.join(sim_dir, "dSpike_table.csv")
        if not os.path.exists(dSpike_path):
            print(f"Skipping {sim_dir}, file not found.")
            continue
        df = pd.read_csv(dSpike_path)
        params = analysis.DataReader.load_parameters(sim_dir)
        CI_amplitude = params.h_i_amplitude
        numpy_random_state = params.numpy_random_state
        df['sim_dir'] = sim_dir
        df['CI_amplitude'] = CI_amplitude
        df['numpy_random_state'] = numpy_random_state
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

def plot_fi_multi(data_dict):
    """Plot FI curves for multiple data sets, each corresponding to a sims_dir label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = plt.cm.tab10.colors  # Up to 10, add more if needed

    for idx, (label, data) in enumerate(data_dict.items()):
        grouped = data.groupby('CI_amplitude')
        ci_values = grouped['CI_amplitude'].first().values
        soma_mean = grouped['Soma_Spike_Rate'].mean().values
        soma_std = grouped['Soma_Spike_Rate'].std().values
        ca_mean = grouped['Total_CA_Spikes'].mean().values
        ca_std = grouped['Total_CA_Spikes'].std().values
        color = colors[idx % len(colors)]

        axes[0].errorbar(ci_values, soma_mean, yerr=soma_std, fmt='-o', capsize=3, label=label, color=color)
        axes[1].errorbar(ci_values, ca_mean, yerr=ca_std, fmt='-o', capsize=3, label=label, color=color)

    axes[0].set_xlabel('CI amplitude')
    axes[0].set_ylabel('Soma Spike Rate')
    axes[0].set_title('Soma Spike Rate vs. CI amplitude')
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    axes[1].set_xlabel('CI amplitude')
    axes[1].set_ylabel('Total CA Spikes')
    axes[1].set_title('Apical Total CA Spikes vs. CI amplitude')
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    plt.suptitle('FI Curves Comparison')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('FI_curves_comparison.png')
    plt.show()

def get_spike_rates_and_plot_fi_multi(sims_dirs):
    data_dict = {}
    for sims_dir in sims_dirs:
        # Find all subdirectories in this sims_dir
        sim_dirs = [
            os.path.join(sims_dir, sim_dir) 
            for sim_dir in os.listdir(sims_dir)
            if os.path.isdir(os.path.join(sims_dir, sim_dir))
        ]
        all_data = gather_spike_tables(sim_dirs)
        if all_data.empty:
            print(f"No data found in {sims_dir}")
            continue
        apical_data = all_data[all_data['Segment_Type'] == 'apic']
        # Use the last part of the path as label
        label = os.path.basename(sims_dir)#.split('-')[-1].replace('_', ' ')  # Adjust this to get a meaningful label
        data_dict[label] = apical_data

    if not data_dict:
        print("No valid data to plot.")
        return
    plot_fi_multi(data_dict)

if __name__ == "__main__":
    # Define as a LIST of parent dirs (could be one, could be several)
    sims_dirs = [
        # '/home/drfrbc/Neural-Modeling/simulations/2025-06-13-17-10-constant-poisson_no-delay-inh_no-clustering_fi',
        # '/home/drfrbc/Neural-Modeling/simulations/2025-06-13-20-21-pink-poisson_delay-inh_no-clustering_fi',
        # '/home/drfrbc/Neural-Modeling/simulations/2025-06-13-23-09-pink-poisson_delay-inh_WITH_tuft-exc-clustering_fi',
        # '/home/drfrbc/Neural-Modeling/simulations/2025-06-16-15-59-tuft_rhythmic_inh',
        # '/home/drfrbc/Neural-Modeling/simulations/2025-06-16-14-39-fr_shifts_negative',
        '/home/drfrbc/Neural-Modeling/simulations/2025-06-17-15-55-fr_shifts_-1',
        '/home/drfrbc/Neural-Modeling/simulations/2025-06-17-16-16-fr_shifts_-2',
        '/home/drfrbc/Neural-Modeling/simulations/2025-06-18-10-26-fr_shifts_-0'
        # '/path/to/other/parent_dir',
        # Add more as needed
    ]
    get_spike_rates_and_plot_fi_multi(sims_dirs)
    print(f'Finished all simulations')
