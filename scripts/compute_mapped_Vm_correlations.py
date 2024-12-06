import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
import os
import numpy as np
import pandas as pd
import glob
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean, cosine
from scipy.signal import correlate
import matplotlib.pyplot as plt
import seaborn as sns


def extract_peak_info(corr, lags):
    peak_index = np.argmax(np.abs(corr))  # Index of the peak correlation value
    peak_value = corr[peak_index]         # Peak correlation value
    peak_lag = lags[peak_index]           # Corresponding lag time
    return peak_value, peak_lag

def create_heatmaps(results, save=False, sim_directory=None):
    # Determine unique sources and targets from the results
    sources = sorted(set(key.split(' to ')[0] for key in results))
    targets = sorted(set(key.split(' to ')[1] for key in results))

    # If there are too many segments, consider strategies for reduction or aggregation
    #if len(sources) * len(targets) > 1000:  # example threshold
    #    print("Too many segments to display clearly on a heatmap.")
    #    return

    # Initialize matrices for peak values and time lags
    peak_values = np.full((len(targets), len(sources)), np.nan)
    time_lags = np.full((len(targets), len(sources)), np.nan)

    # Populate the matrices with peak correlation values and their corresponding lags
    for key, corr in results.items():
        source_seg, target_seg = key.split(' to ')
        source_index = sources.index(source_seg)
        target_index = targets.index(target_seg)
        len_corr = len(corr)
        lags = np.arange(-len_corr // 2, len_corr // 2 + len_corr % 2)
        peak_value, peak_lag = extract_peak_info(corr, lags)
        peak_values[target_index, source_index] = peak_value
        time_lags[target_index, source_index] = peak_lag

    # Define limits for the color bars based on percentiles
    vmax_peak = np.nanpercentile(peak_values, 99)
    vmin_peak = np.nanpercentile(peak_values, 1)
    vmax_lag = np.nanpercentile(time_lags, 99)
    vmin_lag = np.nanpercentile(time_lags, 1)

    # Create the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    sns.heatmap(peak_values, ax=axes[0], cmap='viridis', vmax=vmax_peak, vmin=vmin_peak,
                xticklabels=50, yticklabels=50)  # Only show some labels to avoid clutter
    axes[0].set_title('Peak Correlation Values')

    sns.heatmap(time_lags, ax=axes[1], cmap='coolwarm', vmax=vmax_lag, vmin=vmin_lag,
                xticklabels=50, yticklabels=50)  # Only show some labels to avoid clutter
    axes[1].set_title('Correlation Peak Lag Times')

    plt.tight_layout()

    # Save the figure if the save flag is set and a directory is provided
    if save and sim_directory is not None:
        filepath = os.path.join(sim_directory, "cross_correlations_heatmap.png")
        fig.savefig(filepath, dpi=fig.dpi)

    plt.show()

def visualize_correlations(correlations):
    fig = plt.figure(figsize=(15, 5))
    for pair, corr in correlations.items():
        len_source = len(correlations[pair])
        lags = np.arange(-len_source // 2, len_source // 2 + (1 if len_source % 2 == 0 else 0))
        plt.plot(lags, corr, label=pair)
    plt.legend()
    plt.title('Cross-Correlations between Segments')
    plt.xlabel('Lag')
    plt.ylabel('Cross-Correlation')
    #plt.show()
    if save:
        fig.savefig(os.path.join(sim_directory, f"cross_correlations.png"), dpi = fig.dpi)
    plt.close()

def calculate_correlations(sim_directories, base_name):
    #print(f"Analyzing {base_name} with directories: {sim_directories}")
    amplitudes=[]
    firing_rates=[]
    for sim_directory in sim_directories:
      soma_spikes = analysis.DataReader.read_data(sim_directory, "soma_spikes")
      parameters = analysis.DataReader.load_parameters(sim_directory)
      amplitudes.append(parameters.h_i_amplitude)
      firing_rates.append(analyze_and_log(soma_spikes, parameters, base_name))
      v = analysis.DataReader.read_data(sim_directory, "v")
      segment_data = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))['seg']
      segment_names = segment_data["seg"]
      mapping = read_segment_mapping(sim_directory, "")
      segment_names = read_segment_names(names_csv_path)
      signal_pairs = prepare_signals(your_signal_data, mapping, segment_names)
      for signal1, signal2 in signal_pairs:
        print(calculate_pearson(signal1, signal2))
      # And so on for the other functions

def read_seg_to_seg_csv(directory, seg_to_segs, seg_voltages, model_morphs):
    """Finds 'seg_to_seg.csv' files in each immediate subdirectory, reads them, and stores them in a dictionary with prefixes as keys."""
    found=False
    for file_name in os.listdir(directory):
        print(f"checking {file_name} in {directory}")
        if file_name.endswith("seg_to_seg.csv"):
            prefix = file_name.replace("_seg_to_seg.csv", "")  # Extract prefix
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                print(f"Successfully read {file_path}")
                seg_to_segs[prefix] = df  # Store the DataFrame in the dictionary with the prefix as the key
                seg_voltages[prefix] = analysis.DataReader.read_data(directory, "v")
                model_morphs[prefix] = pd.read_csv(os.path.join(directory, "segment_data.csv"))['seg']
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
            found=True
            break  # Stop searching this directory after finding the first match
    if not found: # detailed has no mapping file
        prefix = 'detailed'  # Extract prefix
        try:
            seg_voltages[prefix] = analysis.DataReader.read_data(directory, "v")
            model_morphs[prefix] = pd.read_csv(os.path.join(directory, "segment_data.csv"))['seg']
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


def analyze_directory(directory, seg_to_segs, seg_voltages, model_morphs):
    """Analyzes a single directory."""
    try:
        print(f"Reading seg_to_seg {directory}")
        # Directly modify and use the passed dictionary, no need to reassign
        read_seg_to_seg_csv(directory, seg_to_segs, seg_voltages, model_morphs)
    except Exception as e:
        print(f"Error processing {directory}: {e}")

def extract_section_and_position(segment_name):
    try:
        # Assuming segment_name format is like "L5PCtemplate[0].dend[6](0.1)"
        # Split the segment name to isolate the anatomical section and position
        section_part = segment_name.split('[')[-1].split(']')[0]
        position_str = segment_name.split('(')[-1].rstrip(')]')  # Remove trailing ")]"
        position = float(position_str)
        return section_part, position
    except ValueError as e:
        print(f"Error parsing segment name '{segment_name}': {e}")
        return None, None

def find_closest_segment(segment_name, segments_list):
    section, position = extract_section_and_position(segment_name)
    if section is None:
        return None

    closest_segment = None
    min_distance = float('inf')
    for candidate_name in segments_list:
        candidate_section, candidate_position = extract_section_and_position(candidate_name)
        # Proceed only if the section matches and position was successfully parsed
        if candidate_section == section and candidate_position is not None:
            distance = abs(position - candidate_position)
            if distance < min_distance:
                min_distance = distance
                closest_segment = candidate_name

    return closest_segment
    
def find_segment_index(segment_name, model_name, model_morphs):
    segments_list = model_morphs[model_name].tolist()  # Convert Series to list
    indices = []
    # If segment_name is a list, iterate through it
    if isinstance(segment_name, list):
        for seg in segment_name:
            try:
                index = segments_list.index(seg)
                indices.append(index)
            except ValueError:
                closest_segment = find_closest_segment(seg, segments_list)
                index = segments_list.index(closest_segment)
                indices.append(index)
    else:
        try:
            index = segments_list.index(segment_name)
            indices.append(index)
        except ValueError:
            closest_segment = find_closest_segment(segment_name, segments_list)
            index = segments_list.index(closest_segment)
            indices.append(index)
    return indices

def compute_cross_correlations(seg_to_segs, seg_voltages, model_morphs):
    # this function uses the seg_to_seg mapping (a dictioary with key 'nr' that maps the names of 'detailed' segments to the names of 'nr' segments and key 'ce' that maps the names of 'nr' segments to the names of 'ce' segments), model_morphs (a dictionary with keys 'nr', 'detailed', 'ce' that has the ordered segment names of each of the models), and seg_voltages (a dictionary with keys 'nr', 'detailed', 'ce' that has the voltage traces for each model's segments. The voltage index within a model corresponds to the segment index for the ordered segment names of the model's "model_morph".
    # this function computes the cross correlation of segments that are mapped together.
    results = {}

    for model_name, mapping_df in seg_to_segs.items():
        source_col = 'neuron_reduce' if model_name == 'nr' else 'cable_expander'
        target_col = 'detailed' if model_name == 'nr' else 'neuron_reduce'

        for _, row in mapping_df.iterrows():
            source_seg = row[source_col]  # Source segment name
            target_seg = row[target_col]  # Target segment name

            source_indices = find_segment_index(source_seg, model_name, model_morphs)
            target_indices = find_segment_index(target_seg, 'detailed' if model_name == 'nr' else 'nr', model_morphs)

            if not isinstance(source_indices, list):
                source_indices = [source_indices]
            if not isinstance(target_indices, list):
                target_indices = [target_indices]

            correlations = []
            for s_index in source_indices:
                for t_index in target_indices:
                    source_voltage = seg_voltages[model_name][s_index]
                    target_voltage = seg_voltages['detailed' if model_name == 'nr' else 'nr'][t_index]

                    cross_corr = np.correlate(source_voltage - np.mean(source_voltage),
                                              target_voltage - np.mean(target_voltage), mode='full')
                    norm_cross_corr = cross_corr / (np.std(source_voltage) * np.std(target_voltage) * len(source_voltage))
                    correlations.append(norm_cross_corr)

            # Average the cross-correlation values if there are multiple
            if correlations:
                avg_corr = np.mean(correlations, axis=0)
                results[f"{source_seg} to {target_seg}"] = avg_corr
            else:
                raise ValueError(f"No valid indices for source {source_seg} or target {target_seg}.")

    return results

def compute_correlations_by_section(seg_to_segs, seg_voltages, model_morphs):
    """
    This function is an extension of compute_cross_correlations that includes correlations
    between cable_expander and detailed through neuron_reduce, and it splits correlations
    by anatomical sections.
    """
    sections = ['dend', 'apic', 'soma']
    results = {
        'ce_to_nr': {sec: {} for sec in sections},
        'nr_to_detailed': {sec: {} for sec in sections},
        'ce_to_detailed': {sec: {} for sec in sections}
    }

    for model_name, mapping_df in seg_to_segs.items():
        for section in sections:
            source_col = 'neuron_reduce' if model_name == 'nr' else 'cable_expander'
            target_col = 'detailed' if model_name == 'nr' else 'neuron_reduce'
            
            for _, row in mapping_df.iterrows():
                source_seg = row[source_col]
                target_seg = row[target_col]
                
                # Filter by section if the segment name contains it
                if section in source_seg.lower() or section in target_seg.lower():
                    source_indices = find_segment_index(source_seg, model_name, model_morphs)
                    target_indices = find_segment_index(target_seg, 'detailed' if model_name == 'nr' else 'nr', model_morphs)
        
                    if not isinstance(source_indices, list):
                        source_indices = [source_indices]
                    if not isinstance(target_indices, list):
                        target_indices = [target_indices]
        
                    correlations = []
                    for s_index in source_indices:
                        for t_index in target_indices:
                            source_voltage = seg_voltages[model_name][s_index]
                            target_voltage = seg_voltages['detailed' if model_name == 'nr' else 'nr'][t_index]
        
                            cross_corr = np.correlate(source_voltage - np.mean(source_voltage),
                                                      target_voltage - np.mean(target_voltage), mode='full')
                            norm_cross_corr = cross_corr / (np.std(source_voltage) * np.std(target_voltage) * len(source_voltage))
                            correlations.append(norm_cross_corr)
        
                    # Average the cross-correlation values if there are multiple
                    if correlations:
                        avg_corr = np.mean(correlations, axis=0)
                        results[f"{source_seg} to {target_seg}"] = avg_corr
                    else:
                        raise ValueError(f"No valid indices for source {source_seg} or target {target_seg}.")

                    # Store results in the dictionary under the appropriate section
                    results_key = f'{model_name}_to_{target_col if model_name == "ce" else "detailed"}'
                    section_key = section  # section is determined by your filtering logic earlier in the loop
                    correlation_key = f"{source_seg} to {target_seg}"
                    if section_key in results[results_key]:  # Check if the section key exists
                        results[results_key][section_key][correlation_key] = avg_corr
                    else:
                        raise ValueError(f"Section key {section_key} not found in results for {results_key}")

    return results

def create_section_heatmaps(correlation_results_by_section, save=False, sim_directory=None):
    """
    Create and save heatmaps for each anatomical section.
    """
    for model_key, section_data in correlation_results_by_section.items():
        for section, results in section_data.items():
                # Determine unique sources and targets from the results
          sources = sorted(set(key.split(' to ')[0] for key in results))
          targets = sorted(set(key.split(' to ')[1] for key in results))

          # If there are too many segments, consider strategies for reduction or aggregation
          #if len(sources) * len(targets) > 1000:  # example threshold
          #    print("Too many segments to display clearly on a heatmap.")
          #    return

          # Initialize matrices for peak values and time lags
          peak_values = np.full((len(targets), len(sources)), np.nan)
          time_lags = np.full((len(targets), len(sources)), np.nan)
      
          # Populate the matrices with peak correlation values and their corresponding lags
          for key, corr in results.items():
              source_seg, target_seg = key.split(' to ')
              source_index = sources.index(source_seg)
              target_index = targets.index(target_seg)
              len_corr = len(corr)
              lags = np.arange(-len_corr // 2, len_corr // 2 + len_corr % 2)
              peak_value, peak_lag = extract_peak_info(corr, lags)
              peak_values[target_index, source_index] = peak_value
              time_lags[target_index, source_index] = peak_lag
      
          # Define limits for the color bars based on percentiles
          vmax_peak = np.nanpercentile(peak_values, 99)
          vmin_peak = np.nanpercentile(peak_values, 1)
          vmax_lag = np.nanpercentile(time_lags, 99)
          vmin_lag = np.nanpercentile(time_lags, 1)
      
          # Create the heatmaps
          fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
      
          sns.heatmap(peak_values, ax=axes[0], cmap='viridis', vmax=vmax_peak, vmin=vmin_peak,
                      xticklabels=50, yticklabels=50)  # Only show some labels to avoid clutter
          axes[0].set_title('Peak Correlation Values')
      
          sns.heatmap(time_lags, ax=axes[1], cmap='coolwarm', vmax=vmax_lag, vmin=vmin_lag,
                      xticklabels=50, yticklabels=50)  # Only show some labels to avoid clutter
          axes[1].set_title('Correlation Peak Lag Times')
      
          plt.tight_layout()

          # Customize title to include model_key and section
          fig.suptitle(f'Cross-Correlation Heatmaps for {model_key} - {section.capitalize()}')

          # Save the figure if the save flag is set and a directory is provided
          if save and sim_directory is not None:
              filename = f"cross_correlations_heatmap_{model_key}_{section}.png"
              filepath = os.path.join(sim_directory, filename)
              fig.savefig(filepath, dpi=fig.dpi)

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv  # Global flag for saving

    if not os.path.isdir(sim_directory):
        raise RuntimeError(f"Specified path is not a directory: {sim_directory}")

    seg_to_segs = {}  # Initialize an empty dictionary to store a DataFrame for each model
    seg_voltages = {} # initialize an empty dictionary to store seg voltages for each model
    model_morphs = {} # initialize an empty dictionary to store seg morphology
    for folder_name in os.listdir(sim_directory):
        folder_path = os.path.join(sim_directory, folder_name)
        if os.path.isdir(folder_path):
            analyze_directory(folder_path, seg_to_segs, seg_voltages, model_morphs)
    
    results = compute_cross_correlations(seg_to_segs, seg_voltages, model_morphs)
    visualize_correlations(results)
    
    #correlation_results_by_section = compute_correlations_by_section(seg_to_segs, seg_voltages, model_morphs)
    #for model_key, section_data in correlation_results_by_section.items():
    #    create_section_heatmaps({model_key: section_data}, save=save, sim_directory=sim_directory)
    create_heatmaps(results)
