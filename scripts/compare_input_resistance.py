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

def calc_time_constant(sim_directory, index_to_use=0):
    print(f"calculating time constant")
    parameters = analysis.DataReader.load_parameters(sim_directory)
    v = analysis.DataReader.read_data(sim_directory, "v")
    
    # if use_nexus:
    #     index_to_use = read_nexus_seg_txt(sim_directory)
    # else:
    #     index_to_use = 0  # Assuming soma if not specified
    
    # Check the array length and ensure indices are within bounds
    array_length = len(v[index_to_use])
    #print(f"Array length: {array_length}")

    steady_state_init_time_index = int((parameters.h_i_delay/2))#/parameters.h_dt)
    steady_state_final_time_index = int((parameters.h_i_delay+parameters.h_i_duration*3/4))#/parameters.h_dt)
    # print(f"h_i_delay_index: {h_i_delay_index}")
    # print(f"h_i_end_index: {h_i_end_index}")
    
    # plt.plot(v[0])
    # plt.show()

    # Ensure indices are within the array bounds
    # steady_state_init_time_index = min(h_i_delay_index, array_length - 1)
    # steady_state_final_time_index = min(h_i_end_index, array_length - 1)
    
    print(f"steady_state_init_time_index: {steady_state_init_time_index}")
    print(f"steady_state_final_time_index: {steady_state_final_time_index}")

    steady_state_v_init = v[index_to_use][steady_state_init_time_index]
    steady_state_v_during = v[index_to_use][steady_state_final_time_index]

    target_voltage = steady_state_v_init + 0.68 * (steady_state_v_during - steady_state_v_init)
    
    print(f"steady_state_v_init: {steady_state_v_init}")
    print(f"target_voltage: {target_voltage}")

    # Search for the time constant within the valid range of the array
    for i in range(steady_state_init_time_index, steady_state_final_time_index + 1):
        if v[index_to_use][i] <= target_voltage:
            time_constant = i - parameters.h_i_delay
            print(f"Time constant: {time_constant} ms")
            return time_constant

    print("Time constant not found within the given range.")
    return None



def calc_input_resistance(sim_directory, index_to_use=0):
    print(f"calculating input resistance")
    parameters = analysis.DataReader.load_parameters(sim_directory)
    parameters.h_i_delay, parameters.h_i_delay + parameters.h_i_duration
    v = analysis.DataReader.read_data(sim_directory, "v")
    print(f"parameters.h_i_delay: {parameters.h_i_delay}")
    print(f"parameters.h_i_duration: {parameters.h_i_duration}")
    steady_state_init_time_index = int((parameters.h_i_delay/2))#/parameters.h_dt)
    steady_state_final_time_index = int((parameters.h_i_delay+parameters.h_i_duration*3/4))#/parameters.h_dt)
    print(f"len(v[0]): {len(v[0])}")
    print(f"steady_state_init_time_index: {steady_state_init_time_index}")
    print(f"steady_state_final_time_index: {steady_state_final_time_index}")
    # if use_nexus:
    #   index_to_use = read_nexus_seg_txt(sim_directory)
    # else:
    #   index_to_use = 0 # soma
    steady_state_v_init = v[index_to_use][steady_state_init_time_index]
    steady_state_v_final = v[index_to_use][steady_state_final_time_index]
    print(f"steady_state_v_init: {steady_state_v_init}")
    print(f"steady_state_v_final: {steady_state_v_final}")
    print(f"amplitude: {parameters.h_i_amplitude}")
    input_resistance = abs((steady_state_v_final - steady_state_v_init) / parameters.h_i_amplitude)
    print(f"input_resistance: {input_resistance} MOhms")
    return input_resistance
    # need to get the voltage from the -1 amp simulations. Then calculate the difference in voltage from (t = 0+delay/2) and (t = delay+(duration*3/4)) and divide by -1 nA

def group_directories_by_prefix(directory_path):
    """
    Groups directories by the prefix before '_amp' in their names and includes the full path.

    Args:
    directory_path (str): The path to the directory containing the folders.

    Returns:
    dict: A dictionary with keys as prefixes and values as lists of directory paths.
    """
    grouped_directories = {}
    for folder_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, folder_name)
        if os.path.isdir(full_path):
            # Extract the part of the folder name before '_amp'
            prefix = folder_name
            if prefix not in grouped_directories:
                grouped_directories[prefix] = []
            grouped_directories[prefix].append(full_path)
    return grouped_directories


if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # Fixed variable name to match usage
    else:
        raise RuntimeError("Directory not specified")

    save = "-s" in sys.argv # (global)
    
    if 'nexus' in sys.argv:
      use_nexus=True
    else:
      use_nexus=False

    # New logic to group directories and analyze them
    grouped_directories = group_directories_by_prefix(sim_directory)
    for base_name, directories in grouped_directories.items():
        #try:
            print("Analyzing Input Resistance for", base_name)
            calc_input_resistance(directories[0], use_nexus) # Pass the list of directories and base name
            calc_time_constant(directories[0])
        #except Exception as e:
        #    print(f"Error processing {base_name}: {e}")
