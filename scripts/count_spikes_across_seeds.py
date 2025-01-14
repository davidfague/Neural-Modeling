# notes:
# there's two subsettings of ca_df and the second is used
import os
import subprocess
import argparse

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../Modules"))
from Modules import analysis

DUR_TO_USE = 10 # seconds of simulation

def count_events(sim_path):
    segs_na_df, segs_nmda_df, segs_ca_df = get_dfs_from_path(sim_path)

    # Initialize an empty list to collect rows before converting them into a DataFrame
    rows = []

    # Define the spike types and their corresponding DataFrames and columns
    spike_types = {
        'num_nmda_spikes': ('Total_NMDA_Spikes', segs_nmda_df),
        'num_na_spikes': ('Total_NA_Spikes', segs_na_df),
        'num_ca_spikes': ('Total_CA_Spikes', segs_ca_df)
    }

    # Calculate the total number of spikes for each segment type and spike type
    for seg_type in ['apic', 'dend']:
        row = {'Segment_Type': seg_type}
        for spike_field, (column_name, df) in spike_types.items():
            total_spikes = df[df.Type == seg_type][spike_field].sum()
            row[column_name] = round(total_spikes / DUR_TO_USE, 1)
        rows.append(row)

    # Convert the list of rows into a DataFrame
    spike_table = pd.DataFrame(rows)

    # Display the table
    return spike_table

def read_segs(sim_directory):
        segs = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
        # make same
        segs['Sec ID'] = segs['idx_in_section_type']
        segs['Type'] = segs['section']
        segs['Coord X'] = segs['pc_0']
        segs['Coord Y'] = segs['pc_1']
        segs['Coord Z'] = segs['pc_2']
        elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{'soma'}.csv"))
        segs['Elec_distance'] = elec_dist['25_active']
        elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{'nexus'}.csv"))
        segs['Elec_distance_nexus'] = elec_dist['25_active']
        Xs = []
        for seg in segs['seg']:
            Xs.append(seg.split('(')[-1].split(')')[0])
        segs['X'] = Xs

        # continue
        segs['segmentID'] = segs.index

        segs['Sec ID'] = segs['Sec ID'].astype(int)
        segs['X'] = segs['X'].astype(float)
        segs['Elec_distanceQ'] = 'None'

        segs.loc[segs.Type=='dend','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='dend','Elec_distance'], 10, labels=False)
        segs.loc[segs.Type=='apic','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='apic','Elec_distance'], 10, labels=False)
        return segs

def get_dfs_from_path(sim_path):
    segs = read_segs(sim_path)
    spks = analysis.DataReader.read_data(sim_path, "soma_spikes")
    spktimes = spks[0][:]
    spkinds = np.sort((spktimes*10).astype(int))
    na = analysis.DataReader.read_data(sim_path, "gNaTa_t_NaTa_t").T

    # na
    na_df = pd.read_csv(os.path.join(sim_path, 'na.csv'))

    max_retries = 1000  # Number of retries
    for _ in range(max_retries):
        try:
            for i in np.random.choice(na_df[(na_df.na_lower_bound > 20) & (na_df.na_lower_bound < 1400000)].index, 10000):
                seg = na_df.loc[i, 'segmentID']
                if not pd.isnull(na_df.loc[i, 'na_lower_bound']):
                    spkt = int(na_df.loc[i, 'na_lower_bound'])
                    # Ensure trace slicing does not go out of bounds
                    trace_start = max(0, spkt - 10)
                    trace_end = min(na.shape[0], spkt + 10)
                    trace = na[trace_start:trace_end, seg]
                    
                    if len(trace) == 20:  # Ensure the trace is the expected length
                        peak_value = np.max(trace)
                        half_peak = peak_value / 2
                        duration = np.arange(trace_start - (spkt - 10), trace_end - (spkt - 10))[trace > half_peak] + trace_start - (spkt - 10)
                        na_df.loc[i, 'duration_low'] = duration[0]
                        na_df.loc[i, 'duration_high'] = duration[-1]
                        na_df.loc[i, 'peak_value'] = peak_value
                    else:
                        raise ValueError(f"Trace length is {len(trace)} not 20, retrying...")
                else:
                    raise ValueError("Invalid na_lower_bound, retrying...")
            break  # Exit the retry loop if no errors occur
        except Exception as e:
            print(f"Retry {_ + 1}/{max_retries} failed: {e}")
    else:
        print("Maximum retries reached. Unable to process.")

    na_df['duration'] = (na_df['duration_high'] - na_df['duration_low'] + 1)/10
    seg_na_df = na_df.groupby('segmentID')['na_lower_bound'].count().reset_index().rename(columns={'na_lower_bound':'num_na_spikes'})
    segs_na_df = segs.set_index('segmentID').join(seg_na_df.set_index('segmentID'))
    segs_na_df.loc[segs_na_df.num_na_spikes>1000,'num_na_spikes'] = 1000

    # ca
    ca_df = pd.read_csv(os.path.join(sim_path, 'ca.csv'))
    ca_df['dist_from_soma_spike'] = ca_df['ca_lower_bound'].apply(lambda x: np.min(np.abs(x-spkinds)))
    ca_df['duration'] = (ca_df['ca_upper_bound'] - ca_df['ca_lower_bound'])/10
    ca_df['mag_dur'] = ca_df['mag']/ca_df['duration']
    ca_df = ca_df[(ca_df.mag<-0.1)&
                           (ca_df.duration<250)&
                           (ca_df.duration>26)&
                           (ca_df.dist_from_soma_spike>50)&
                           (ca_df.mag_dur<-0.006)]
    ca_df = ca_df[(ca_df.mag<-0.1)&
                           (ca_df.duration<250)&
                           (ca_df.duration>26)&
                           (ca_df.dist_from_soma_spike>50)]
    seg_ca_df = ca_df.groupby('segmentID')['ca_lower_bound'].count().reset_index().rename(columns={'ca_lower_bound':'num_ca_spikes'})
    segs_ca_df = segs.set_index('segmentID').join(seg_ca_df.set_index('segmentID')).reset_index()

    # nmda
    nmda_df = pd.read_csv(os.path.join(sim_path, 'nmda.csv'))
    nmda_df['duration'] = (nmda_df['nmda_upper_bound'] - nmda_df['nmda_lower_bound'])/10
    nmda_df['log_duration'] = np.log(nmda_df['duration'])
    nmda_df['log_mag'] = np.log(np.abs(nmda_df['mag']))
    seg_nmda_df = nmda_df.groupby('segmentID')['nmda_lower_bound'].count().reset_index().rename(columns={'nmda_lower_bound':'num_nmda_spikes'})
    segs_nmda_df = segs.set_index('segmentID').join(seg_nmda_df.set_index('segmentID'))

    return segs_na_df, segs_nmda_df, segs_ca_df


def main():
    '''Deprecating'''
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run find_events_ben.py for each simulation directory.")
    parser.add_argument("-d", "--directory", required=True, help="Path to the directory containing simulation folders.")
    args = parser.parse_args()

    # Get the path to the root directory
    root_dir = args.directory

    # Ensure the directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: The specified directory '{root_dir}' does not exist.")
        return

    # Iterate over subdirectories
    for sim_dir in os.listdir(root_dir):
        sim_path = os.path.join(root_dir, sim_dir)

        # Check if the path is a directory
        if os.path.isdir(sim_path):
            print(f"Processing simulation directory: {sim_path}")
            try:
                # Call find_events_ben.py with the current simulation directory
                subprocess.run(
                    ["python", "find_events_ben.py", "-d", sim_path],
                    check=True
                )
                spike_table = count_events(sim_path)
                spike_table.to_csv(os.path.join(sim_path, "spike_table.csv"))

            except subprocess.CalledProcessError as e:
                print(f"Error running find_events_ben.py for {sim_path}: {e}")

# if __name__ == "__main__":
    # main()
    #####
    # ben = False
    # if "-d" in sys.argv: # arg is a simulation directory
    #     sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    #     spike_table = count_events(sim_directory)
    #     spike_table.to_csv(os.path.join(sim_directory, "spike_table.csv"))
    # elif "-f" in sys.argv: # arg is a simulations folder of simulation directories
    #     simulations_directory = sys.argv[sys.argv.index("-f") + 1]
    #     print(f"simulations_directory: {simulations_directory}")
    #     for sim_directory in os.listdir(simulations_directory):
    #         sim_path = os.path.join(simulations_directory, sim_directory)
    #         spike_table = count_events(sim_path)
    #         spike_table.to_csv(os.path.join(sim_path, "spike_table.csv"))
    # else:
    #     raise RuntimeError

def aggregate_simulation_data(simulations_directory):
    combined_table = []
    
    for sim_directory in os.listdir(simulations_directory):
        sim_path = os.path.join(simulations_directory, sim_directory)
        if not os.path.isdir(sim_path):
            continue  # Skip non-directory files
            
        spike_table = count_events(sim_path)
        spike_table['simulation_directory'] = sim_directory  # Add directory name to the table
        combined_table.append(spike_table)
    
    # Combine all DataFrames into one
    if combined_table:
        combined_table = pd.concat(combined_table, ignore_index=True)
    else:
        combined_table = pd.DataFrame()
    
    return combined_table

if __name__ == "__main__":
    if "-d" in sys.argv:  # Single simulation directory
        sim_directory = sys.argv[sys.argv.index("-d") + 1]
        spike_table = count_events(sim_directory)
        spike_table.to_csv(os.path.join(sim_directory, "spike_table.csv"), index=False)
        print(spike_table)
    elif "-f" in sys.argv:  # Folder containing multiple simulation directories
        simulations_directory = sys.argv[sys.argv.index("-f") + 1]
        print(f"simulations_directory: {simulations_directory}")
        
        combined_table = aggregate_simulation_data(simulations_directory)
        
        if not combined_table.empty:
            combined_csv_path = os.path.join(simulations_directory, "combined_spike_table.csv")
            combined_table.to_csv(combined_csv_path, index=False)
            print(f"Combined table saved to {combined_csv_path}")
            print(combined_table)
        else:
            print("No valid simulation data found in the directory.")
    else:
        raise RuntimeError("No valid arguments provided. Use -d for a single directory or -f for a folder of directories.")

#TODO: store spike_table for each simulation, compute mean,std across simulations for each spike type in the table.