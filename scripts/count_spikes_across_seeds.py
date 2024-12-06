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

DUR_TO_USE = 150 # seconds of simulation

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

def get_dfs_from_path(sim_path):
    segs = read_segs(sim_path)
    import analysis
    spks = analysis.DataReader.read_data(sim_path, "soma_spikes")
    spktimes = spks[0][:]
    spkinds = np.sort((spktimes*10).astype(int))

    # na
    na_df = pd.read_csv(os.path.join(sim_path, 'nmda.csv'))
    for i in np.random.choice(na_df[(na_df.na_lower_bound>20) & (na_df.na_lower_bound<1400000)].index,10000):
        seg = na_df.loc[i,'segmentID']
        if not pd.isnull(na_df.loc[i,'na_lower_bound']):
            spkt = int(na_df.loc[i,'na_lower_bound'])
            trace = na[spkt-10:spkt+10,seg]#['report']['biophysical']['data'][spkt-10:spkt+10,seg]
            peak_value = np.max(trace)
            half_peak = peak_value/2
            duration = np.arange(0,20)[trace>half_peak] + spkt - 10
            na_df.loc[i,'duration_low'] = duration[0]
            na_df.loc[i,'duration_high'] = duration[-1]
            na_df.loc[i,'peak_value'] = peak_value
        else:
            na_df.loc[i,'duration_low'] = np.nan
            na_df.loc[i,'duration_high'] = np.nan
            na_df.loc[i,'peak_value'] = np.nan    
    na_df['duration'] = (na_df['duration_high'] - na_df['duration_low'] + 1)/10
    seg_na_df = na_df.groupby('segmentID')['na_lower_bound'].count().reset_index().rename(columns={'na_lower_bound':'num_na_spikes'})
    segs_na_df = segs.set_index('segmentID').join(seg_na_df.set_index('segmentID'))
    segs_na_df.loc[segs_na_df.num_na_spikes>1000,'num_na_spikes'] = 1000

    # ca
    ca_df = pd.read_csv(os.path.join(sim_path, 'nmda.csv'))
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

if __name__ == "__main__":
    main()

#TODO: store spike_table for each simulation, compute mean,std across simulations for each spike type in the table.