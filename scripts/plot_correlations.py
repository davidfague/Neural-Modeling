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

import os
import pandas as pd

import ast
from collections import defaultdict
import re

#get_mappings(sim_directory):
#  mappings = {'detailed to neuron_reduce':{}, 'detailed to cable_expander':{}, 'neuron_reduce to cable_expander':{}}
#  mapping_files = [os.path.join(dir_path, file) for dir_path in [os.path.join(sim_directory, d) for d in os.listdir(sim_directory) if os.path.isdir(os.path.join(sim_directory, d))] for file in os.listdir(dir_path) if file.endswith("seg_to_seg.csv")]
#  #print(f"mapping_files: {mapping_files}")
#  for mapping_file in mapping_files:
#     if mapping_file.split('/')[-1].startswith("nr"):
#       mappings['detailed to neuron_reduce'] = pd.read_csv(mapping_file)
#     elif mapping_file.split('/')[-1].startswith("ce"):
#       mappings['neuron_reduce to cable_expander'] = pd.read_csv(mapping_file)

#def get_mappings(sim_directory):
#    mappings = {'detailed to neuron_reduce':{}, 'detailed to cable_expander':{}, 'neuron_reduce to cable_expander':{}}
#    mapping_files = [os.path.join(dir_path, file) 
#                     for dir_path in [os.path.join(sim_directory, d) for d in os.listdir(sim_directory) if os.path.isdir(os.path.join(sim_directory, d))] 
#                     for file in os.listdir(dir_path) if file.endswith("seg_to_seg.csv")]
#
#    for mapping_file in mapping_files:
#        if mapping_file.split('/')[-1].startswith("nr"):
#            df = pd.read_csv(mapping_file)
#            # Convert the DataFrame into the desired dictionary format
#            detailed_to_neuron_reduce = {row['detailed']: row['neuron_reduce'] for index, row in df.iterrows()}
#            mappings['detailed to neuron_reduce'] = detailed_to_neuron_reduce
#        elif mapping_file.split('/')[-1].startswith("ce"):
#            df = pd.read_csv(mapping_file)
#            # Assuming a similar structure for cable_expander mappings, convert those as well
#            neuron_reduce_to_cable_expander = {row['neuron_reduce']: row['cable_expander'] for index, row in df.iterrows()} if 'cable_expander' in df.columns else {}
#            mappings['neuron_reduce to cable_expander'] = neuron_reduce_to_cable_expander
#
#    return mappings

def get_mappings(sim_directory):
    mappings = {'detailed to neuron_reduce': {}, 'detailed to cable_expander': {}, 'neuron_reduce to cable_expander': {}}
    mapping_files = [os.path.join(dir_path, file)
                     for dir_path in [os.path.join(sim_directory, d) for d in os.listdir(sim_directory) if os.path.isdir(os.path.join(sim_directory, d))]
                     for file in os.listdir(dir_path) if file.endswith("seg_to_seg.csv")]

    for mapping_file in mapping_files:
        if mapping_file.split('/')[-1].startswith("nr"):
            df = pd.read_csv(mapping_file)
            detailed_to_neuron_reduce = {row['detailed']: row['neuron_reduce'] for index, row in df.iterrows()}
            mappings['detailed to neuron_reduce'] = detailed_to_neuron_reduce
        elif mapping_file.split('/')[-1].startswith("ce"):
            df = pd.read_csv(mapping_file)
            neuron_reduce_to_cable_expander = defaultdict(list)
            if 'cable_expander' in df.columns:
                # Define a regex pattern to match the model segments
                pattern = r"model\[\d+\]\.\w+\[\d+\]\(\d+\.\d+\)"
                for index, row in df.iterrows():
                    # Find all matches of the pattern in the cable_expander string
                    cable_list = re.findall(pattern, row['cable_expander'])
                    # Extend the list for this neuron_reduce key with the cables
                    neuron_reduce_to_cable_expander[row['neuron_reduce']].extend(cable_list)
            mappings['neuron_reduce to cable_expander'] = dict(neuron_reduce_to_cable_expander)
            
    #mappings['neuron_reduce to cable_expander'] = invert_mappings(mappings) # for neuron_reduce to cable_expander

    return mappings

#def invert_mappings(mappings):
#    inverted_mappings = defaultdict(list)
#    for neuron_reduce, cable_expanders in mappings['neuron_reduce to cable_expander'].items():
#        for cable_expander in cable_expanders:
#            inverted_mappings[cable_expander].append(neuron_reduce)
#    return dict(inverted_mappings)

def get_voltages(sim_directory):
    seg_voltages = {}
    model_directories = [os.path.join(sim_directory, d) for d in os.listdir(sim_directory) if os.path.isdir(os.path.join(sim_directory, d))]
    
    for model_folder in model_directories:
        model_segment_names = pd.read_csv(os.path.join(model_folder, "segment_data.csv"))['seg']
        model_name = model_folder.split('/')[-1]  # Assuming this function correctly extracts the model name
        v = analysis.DataReader.read_data(model_folder, "v")
        
        # Initialize the model_name key if it doesn't exist
        if model_name not in seg_voltages:
            seg_voltages[model_name] = {}
        
        for index, model_segment_name in enumerate(model_segment_names):
            # Assign voltage values to the corresponding segment name under the model name key
            seg_voltages[model_name][model_segment_name] = v[index]
    
    return seg_voltages

def get_voltage_mappings_and_compute_correlations(mappings, voltages, from_model, to_model):
    correlations = {}
    for from_seg_name in voltages[from_model].keys():
        from_v = get_voltage_from_seg_name(voltages, from_model, from_seg_name)
        to_seg_name = get_seg_mapped_to(mappings, from_model, to_model, from_seg_name, voltages)
        to_v = get_voltage_from_seg_name(voltages, to_model, to_seg_name)
        correlations[f"{from_seg_name} to {to_seg_name}"] = correlate(from_v, to_v)
    return correlations


def get_seg_mapped_to(mappings, from_model, to_model, from_seg_name, voltages):
    section, position, sec_index = extract_section_and_position(from_seg_name)
    #print(f"section: {section}")
    #print(list(voltages[to_model].keys()))
    if 'soma' == section:
      to_seg_options = [to_seg_option for to_seg_option in list(voltages[to_model].keys())
                          if 'soma' == extract_section_and_position(to_seg_option)[0]]
      if len(to_seg_options) == 1:
        to_seg_name = to_seg_options[0]
      else:
        raise ValueError(f"ERROR: got too many seg_names when mapping {from_seg_name} to {to_seg_options}.")
      #print(f"mapped {from_seg_name} to {to_seg_name}")
      return to_seg_name
    from_model_to_model_key = get_from_model_to_model_key(from_model, to_model)
    try:
        to_seg_name = mappings[from_model_to_model_key][from_seg_name]
    except KeyError:
        # If direct mapping is not found, find the closest segment
        closest_seg_name = get_closest_seg_name(voltages[to_model].keys(), from_seg_name)
        if closest_seg_name:
            print(f"Exact match not found for {from_seg_name} using {closest_seg_name}")
            return closest_seg_name
        else:
            raise ValueError(f"ERROR: Cannot find a mapping or closest segment for {from_seg_name}")

    return to_seg_name
  
def get_from_model_to_model_key(from_model, to_model):
  if from_model == 'detailed' and to_model == 'neuron_reduce':
    return 'detailed to neuron_reduce'
  elif from_model == 'detailed' and to_model == 'cable_expander':
    return 'detailed to cable_expander'
  elif from_model == 'neuron_reduce' and to_model == 'cable_expander':
    return 'neuron_reduce to cable_expander'
  else:
    raise(ValueError(f"{from_model} to {to_model} is not supported in get_from_model_to_model_key()"))

def get_voltage_from_seg_name(voltages, model_name, seg_name):
  try:
    v = voltages[model_name][seg_name]
    #print(f"len(v):{len(v)}")
    return v
  except:
    v = search_for_best_v(voltages, model_name, seg_name)
    return v

def get_closest_seg_name(segment_names, target_seg_name):
    # need to update to use the exact sec index.
    section, position, sec_index = extract_section_and_position(target_seg_name)
    #print(f"section in search_for_best_v: {target_seg_name,section}")
    if section is None:
        raise(ValueError("section is None in search_for_best_v"))
    
    closest_segment = None
    min_distance = float('inf')
    for candidate_name in segment_names:
        candidate_section, candidate_position, candidate_section_index = extract_section_and_position(candidate_name)
        # Proceed only if the section matches and position was successfully parsed
        if candidate_section == section and candidate_section_index == sec_index and candidate_position is not None:
            distance = abs(position - candidate_position)
            if distance < min_distance:
                min_distance = distance
                closest_segment = candidate_name
    return closest_segment

def search_for_best_v(voltages, model_name, seg_name):
    #print(f"voltages[{model_name}].keys(): {voltages[model_name].keys()}")
    closest_segment = get_closest_seg_name(segment_names = voltages[model_name].keys(), target_seg_name=seg_name)
    v = voltages[model_name][closest_segment]
    print(f"voltage not found for {seg_name} using {closest_segment}")
    return v
    
def extract_section_and_position(segment_name):
    try:
        # Split the segment name by '.' to separate into components
        parts = segment_name.split('.')
        # The section name is in the second-to-last part, after splitting by '.'
        # and before the last '[' character
        section_part_with_index = parts[-2]  # Gets the part with "soma[0]" or similar
        section_part = section_part_with_index.split('[')[0]  # Isolates "soma" from that part
        section_index = section_part_with_index.split('[')[1].split(']')[0]

        # Extracting the position, which is the float value inside the parentheses
        position_str = segment_name.split('(')[-1].rstrip(')')  # Isolates "0.5" or similar, by removing the closing parenthesis
        position = float(position_str)

        #print(f"extracted section_part, position: {section_part}, {position} from {segment_name}")
        return section_part, position, section_index
    except ValueError as e:
        raise ValueError(f"Error parsing segment name '{segment_name}': {e}")


if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv  # Global flag for saving

    if not os.path.isdir(sim_directory):
        raise RuntimeError(f"Specified path is not a directory: {sim_directory}")

    mappings = get_mappings(sim_directory)
    #print(f"mappings: {mappings}")
    #print(f"mappings['detailed to neuron_reduce']: {mappings['detailed to neuron_reduce']}")
    #print(f"mappings['detailed to cable_expander']: {mappings['detailed to cable_expander']}")
    print(f"mappings['neuron_reduce to cable_expander']: {mappings['neuron_reduce to cable_expander']}")
    voltages = get_voltages(sim_directory)
    #print(voltages['neuron_reduce'].keys())
    #segs= voltages['detailed'].keys()
    #print(f"len(segs): {len(segs)}")
    #print(f"voltages['Detailed']: {voltages['Detailed'].keys()}")
    #print(f"voltages['Reduced']: {voltages['Reduced'].keys()}")
    #print(f"voltages['Expanded']: {voltages['Expanded'].keys()}")

    #detailed_to_reduced_correlations = get_voltage_mappings_and_compute_correlations(mappings, voltages, from_model='detailed', to_model='neuron_reduce')
    #print(detailed_to_reduced_correlations)
    #detailed_to_expanded_correlations = get_voltage_mappings_and_compute_correlations(mappings, voltages, from_model='detailed', to_model='cable_expander')
    #print(detailed_to_expanded_correlations)
    #reduced_to_expanded_correlations = get_voltage_mappings_and_compute_correlations(mappings, voltages, from_model='neuron_reduce', to_model='cable_expander')
    #print(reduced_to_expanded_correlations)