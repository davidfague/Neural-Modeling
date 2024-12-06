import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
import os
import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt
import seaborn as sns

def extract_peak_info(corr, lags):
    peak_index = np.argmax(np.abs(corr))
    peak_value = corr[peak_index]
    peak_lag = lags[peak_index]
    return peak_value, peak_lag

def compute_cross_correlations(seg_voltages, mappings):
    correlations = {}
    for mapping, (source_model, target_model) in mappings.items():
        source_voltages = seg_voltages[source_model]
        target_voltages = seg_voltages[target_model]
        
        for source_seg in source_voltages:
            for target_seg in target_voltages:
                source_voltage = source_voltages[source_seg]
                target_voltage = target_voltages[target_seg]
                corr = correlate(source_voltage - np.mean(source_voltage), 
                                 target_voltage - np.mean(target_voltage), mode='full')
                norm_corr = corr / (np.std(source_voltage) * np.std(target_voltage) * len(source_voltage))
                correlations[f"{source_seg} to {target_seg}"] = norm_corr
    return correlations

def create_heatmaps(correlations, title, save=False, sim_directory=None):
    sources = sorted(set(key.split(' to ')[0] for key in correlations.keys()))
    targets = sorted(set(key.split(' to ')[1] for key in correlations.keys()))

    peak_values = np.zeros((len(targets), len(sources)))
    time_lags = np.zeros((len(targets), len(sources)))

    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            key = f"{source} to {target}"
            if key in correlations:
                corr = correlations[key]
                lags = np.arange(-len(corr)//2, len(corr)//2 + 1)
                peak_value, peak_lag = extract_peak_info(corr, lags)
                peak_values[j, i] = peak_value
                time_lags[j, i] = peak_lag

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    sns.heatmap(peak_values, ax=axes[0], annot=True)
    axes[0].set_title('Peak Correlation Values')
    sns.heatmap(time_lags, ax=axes[1], annot=True)
    axes[1].set_title('Correlation Peak Lag Times')
    plt.suptitle(title)

    if save and sim_directory:
        plt.savefig(os.path.join(sim_directory, f"{title.replace(' ', '_')}_heatmap.png"))
    plt.show()

if __name__ == "__main__":
    sim_directory = sys.argv[1] if len(sys.argv) > 1 else "./"
    save = "-s" in sys.argv

    seg_voltages = {
        # Load your segment voltage data for 'detailed', 'nr', and 'ce' models
        'detailed': {}, # Placeholder: Replace with your actual data loading logic
        'nr': {},
        'ce': {}
    }

    # Define mappings between models
    mappings = {
        'detailed_to_nr': ('detailed', 'nr'),
        'detailed_to_ce': ('detailed', 'ce'),
        'nr_to_ce': ('nr', 'ce')
    }

    # Compute correlations
    correlations = compute_cross_correlations(seg_voltages, mappings)

    # Create heatmaps for each mapping
    for mapping in mappings:
        create_heatmaps(correlations, mapping, save=save, sim_directory=sim_directory)
