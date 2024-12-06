import os
import subprocess
import argparse
import pandas as pd
import numpy as np

DUR_TO_USE = 150  # seconds of simulation

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

def count_events(sim_path):
    segs_na_df, segs_nmda_df, segs_ca_df = get_dfs_from_path(sim_path)

    rows = []
    spike_types = {
        'num_nmda_spikes': ('Total_NMDA_Spikes', segs_nmda_df),
        'num_na_spikes': ('Total_NA_Spikes', segs_na_df),
        'num_ca_spikes': ('Total_CA_Spikes', segs_ca_df),
    }

    for seg_type in ['apic', 'dend']:
        row = {'Segment_Type': seg_type}
        for spike_field, (column_name, df) in spike_types.items():
            total_spikes = df[df.Type == seg_type][spike_field].sum()
            row[column_name] = round(total_spikes / DUR_TO_USE, 1)
        rows.append(row)

    return pd.DataFrame(rows)

def gather_and_save_tables(root_dir, destination_path):
    all_tables = []
    directories = []

    # Process each simulation directory
    for sim_dir in os.listdir(root_dir):
        sim_path = os.path.join(root_dir, sim_dir)

        if os.path.isdir(sim_path):
            print(f"Processing simulation directory: {sim_path}")
            try:
                # Run the simulation processing script
                subprocess.run(
                    ["python", "find_events_ben.py", "-d", sim_path],
                    check=True
                )
                
                # Get the spike table
                spike_table = count_events(sim_path)
                spike_table['Directory'] = sim_path  # Add directory information
                all_tables.append(spike_table)
                directories.append(sim_path)

            except subprocess.CalledProcessError as e:
                print(f"Error running find_events_ben.py for {sim_path}: {e}")

    if all_tables:
        # Combine all spike tables
        combined_table = pd.concat(all_tables, ignore_index=True)
        
        # Compute mean and standard deviation
        summary_stats = combined_table.groupby("Segment_Type").agg(
            {
                "Total_NMDA_Spikes": ["mean", "std"],
                "Total_NA_Spikes": ["mean", "std"],
                "Total_CA_Spikes": ["mean", "std"],
            }
        ).reset_index()
        summary_stats.columns = [
            "Segment_Type",
            "NMDA_Mean",
            "NMDA_Std",
            "NA_Mean",
            "NA_Std",
            "CA_Mean",
            "CA_Std",
        ]
        summary_stats["Directory"] = "Summary Statistics"
        
        # Append summary stats to combined table
        combined_table = pd.concat([combined_table, summary_stats], ignore_index=True)

        # Save to CSV
        combined_table.to_csv(destination_path, index=False)
        print(f"Saved combined spike tables to {destination_path}")
    else:
        print("No valid spike tables found.")

def main():
    parser = argparse.ArgumentParser(description="Process simulation directories and save spike data.")
    parser.add_argument("-d", "--directory", required=True, help="Root directory containing simulation folders.")
    parser.add_argument("-o", "--output", required=True, help="Output path for the combined CSV file.")
    args = parser.parse_args()

    root_dir = args.directory
    destination_path = args.output

    if not os.path.isdir(root_dir):
        print(f"Error: The specified directory '{root_dir}' does not exist.")
        return

    gather_and_save_tables(root_dir, destination_path)

if __name__ == "__main__":
    main()
