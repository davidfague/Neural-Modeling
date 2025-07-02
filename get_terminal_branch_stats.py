import os
import pandas as pd
import numpy as np

def get_terminal_branch_statistics(segment_csv_path):
    df = pd.read_csv(segment_csv_path)
    
    # Identify terminal segments: seg that is not listed as a pseg
    segs = set(df['seg'])
    pseg_set = set(df['pseg'].dropna())
    terminal_mask = ~df['seg'].isin(pseg_set)
    terminal_df = df[terminal_mask].copy()
    
    # Prepare output dictionary
    branch_stats = {}
    for sec_type in terminal_df['sec_type_precise'].unique():
        sec_df = terminal_df[terminal_df['sec_type_precise'] == sec_type]
        sec_stats = {}
        for _, row in sec_df.iterrows():
            seg_id = int(row['seg_id'])
            # The terminal branch in this case is just this segment, but
            # you could expand this logic if you want to group segments that form a "branch"
            total_length = float(row['length'])
            center_coords = tuple(round(float(row[c]), 2) for c in ['pc_0', 'pc_1', 'pc_2'])
            sec_stats[seg_id] = {
                'total_length': round(total_length, 2),
                'center_coords': center_coords
            }
        branch_stats[sec_type] = sec_stats
    return branch_stats

if __name__ == "__main__":
    sim_dir = "/home/drfrbc/Neural-Modeling/simulations/2025-07-01-09-59-high_baseline_fr/complex"
    segment_csv_path = os.path.join(sim_dir, "segment_data.csv")
    assert os.path.isfile(segment_csv_path), f"segment_data.csv not found in {sim_dir}"
    branch_stats = get_terminal_branch_statistics(segment_csv_path)
    
    # Write to a .py file
    output_py = os.path.join(sim_dir, "terminal_branch_statistics.py")
    with open(output_py, "w") as f:
        f.write("# Auto-generated terminal branch statistics\n")
        f.write("branch_stats = ")
        import pprint
        pprint.pprint(branch_stats, stream=f, width=120)
    print(f"Wrote: {output_py}")

    # Optional: Print preview
    for sec_type, segs in branch_stats.items():
        print(f"\n{sec_type}:")
        for seg_id, data in segs.items():
            print(f"  Seg {seg_id}:")
            print(f"    Total length: {data['total_length']}")
            print(f"    Center coordinates: {data['center_coords']}")
