import sys
sys.path.append("../")
sys.path.append("../Modules/")

import analysis
from logger import Logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback

import re

def read_nexus_seg_txt(sim_directory):
    # Construct the path to the nexus_seg_index.txt file
    file_path = os.path.join(sim_directory, 'nexus_seg_index.txt')
    
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read all lines in the file
            lines = file.readlines()
            
            # Iterate through each line
            for line in lines:
                # Check if the line contains "Seg Index:"
                if "Seg Index:" in line:
                    # Extract the number following "Seg Index:" and convert it to an integer
                    seg_index = int(line.split("Seg Index:")[1].strip())
                    return seg_index
                    
        # If "Seg Index:" is not found in any line, raise an exception
        raise ValueError(f"'Seg Index:' not found in {file_path}")
    
    except Exception as e:
        # Handle exceptions such as file not found or unable to read file
        print(f"An error occurred: {e}")
        return None

def plot_dvsoma_dvdend_by_distance(sim_directories, parameters):
    #parameters = analysis.DataReader.load_parameters(sim_directory)
    #parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration
    v = analysis.DataReader.read_data(sim_directories[0], "v", parameters=parameters)
    print(sim_directories[0],v)
    #steady_state_init_time_index = int((parameters.h_i_delay/2))#/parameters.h_dt)
    #steady_state_final_time_index = int((parameters.h_i_delay+parameters.h_i_duration*3/4))#/parameters.h_dt)
    #print(f"len(v[0]): {len(v[0])}")
    #print(f"steady_state_init_time_index: {steady_state_init_time_index}")
    #print(f"steady_state_final_time_index: {steady_state_final_time_index}")
    #if use_nexus:
    #  index_to_use = read_nexus_seg_txt(sim_directory)
    #else:
    #  index_to_use = 0 # soma
    #steady_state_v_init = v[index_to_use][steady_state_init_time_index]
    #steady_state_v_final = v[index_to_use][steady_state_final_time_index]
    #print(f"steady_state_v_init: {steady_state_v_init}")
    #print(f"steady_state_v_final: {steady_state_v_final}")
    #input_resistance = (steady_state_v_final - steady_state_v_init) / parameters.h_i_amplitude
    #print(f"input_resistance: {input_resistance}")
    # need to get the voltage from the -1 amp simulations. Then calculate the difference in voltage from (t = 0+delay/2) and (t = delay+(duration*3/4)) and divide by -1 nA

def group_directories_by_prefix(directory_path):
    """
    Groups directories by the prefix "Detailed_EPSPs", separating the base folder from those with trailing integers.

    Args:
    directory_path (str): The path to the directory containing the folders.

    Returns:
    dict: A dictionary with two keys, 'base' for the base folder, and 'numbered' for folders with trailing integers.
          The values are lists of directory paths.
    """
    # Initialize the dictionary with two keys: 'base' for the base folder and 'numbered' for those with integers.
    grouped_directories = {'base': [], 'numbered': []}

    # Compile a regex pattern to identify folders with 'Detailed_EPSPs' followed by zero or more digits.
    pattern = re.compile(r'^Detailed_EPSPs(\d*)$')

    for folder_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(full_path):
            # Use regex to check if the folder name matches the pattern
            match = pattern.match(folder_name)
            if match:
                # If the captured group (digits) is empty, it's the base folder; otherwise, it's a numbered folder.
                if match.group(1) == '':
                    grouped_directories['base'].append(full_path)
                else:
                    grouped_directories['numbered'].append(full_path)
    
    return grouped_directories


if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # Fixed variable name to match usage
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv # (global)

    # New logic to group directories and analyze them
    grouped_directories = group_directories_by_prefix(sim_directory)
    #print(grouped_directories)
    parameters = analysis.DataReader.load_parameters(grouped_directories['base'][0])
        #try:
    #print("Analyzing", base_name)
    plot_dvsoma_dvdend_by_distance(grouped_directories['numbered'], parameters) # Pass the list of directories and base name
        #except Exception as e:
        #    print(f"Error processing {base_name}: {e}")
