#!/usr/bin/env python3
"""
Find the simulation directory with the closest dSpike_table.csv to a specified target.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_distance(target_df, candidate_df, weights=None):
    """
    Calculate the distance between target and candidate dSpike tables.
    Uses normalized Euclidean distance across numeric columns.
    
    Parameters:
    -----------
    target_df : DataFrame
        Target dSpike table
    candidate_df : DataFrame
        Candidate dSpike table
    weights : dict, optional
        Dictionary mapping column names to weight values.
        Higher weight = more importance in distance calculation.
        Example: {'Total_NMDA_Spikes': 2.0, 'Total_NA_Spikes': 0.5}
    """
    # Merge on Segment_Type to align rows
    merged = target_df.merge(candidate_df, on='Segment_Type', suffixes=('_target', '_candidate'))
    
    if len(merged) == 0:
        return float('inf')  # No matching segment types
    
    # Get numeric columns (exclude Segment_Type)
    numeric_cols = [col for col in target_df.columns if col != 'Segment_Type']
    
    # Default weights to 1.0 if not specified
    if weights is None:
        weights = {col: 1.0 for col in numeric_cols}
    
    # Calculate weighted squared differences for each numeric column
    total_distance = 0
    for col in numeric_cols:
        target_vals = merged[f'{col}_target'].values
        candidate_vals = merged[f'{col}_candidate'].values
        
        # Normalize by target values to make comparison scale-invariant
        # Add small epsilon to avoid division by zero
        normalized_diff = (target_vals - candidate_vals) / (target_vals + 1e-6)
        
        # Apply weight (default to 1.0 if not specified)
        weight = weights.get(col, 1.0)
        total_distance += weight * np.sum(normalized_diff ** 2)
    
    return np.sqrt(total_distance)


def find_closest_dspike(target_dspike_dict, sims_dir, weights=None):
    """
    Find the simulation directory with the closest dSpike_table.csv.
    
    Parameters:
    -----------
    target_dspike_dict : dict
        Dictionary defining the target dSpike table
    sims_dir : str or Path
        Directory containing simulation subdirectories
    weights : dict, optional
        Dictionary mapping column names to weight values.
        Higher weight = more importance in distance calculation.
        Example: {'Total_NMDA_Spikes': 2.0, 'Total_NA_Spikes': 0.5}
    
    Returns:
    --------
    str : Path to the closest matching simulation directory
    """
    # Convert target dictionary to DataFrame
    target_df = pd.DataFrame(target_dspike_dict)
    
    # Search for all dSpike_table.csv files
    sims_path = Path(sims_dir)
    dspike_files = list(sims_path.glob('*/dSpike_table.csv'))
    
    if not dspike_files:
        print(f"No dSpike_table.csv files found in {sims_dir}")
        return None
    
    print(f"Found {len(dspike_files)} dSpike_table.csv files")
    
    # Calculate distance for each file
    min_distance = float('inf')
    closest_dir = None
    
    for dspike_file in dspike_files:
        try:
            candidate_df = pd.read_csv(dspike_file, index_col=0)
            distance = calculate_distance(target_df, candidate_df, weights)
            
            print(f"  {dspike_file.parent.name}: distance = {distance:.4f}")
            
            if distance < min_distance:
                min_distance = distance
                closest_dir = dspike_file.parent
        except Exception as e:
            print(f"  Error reading {dspike_file}: {e}")
    
    return closest_dir


if __name__ == "__main__":
    # Define target dSpike table
    target_dspike = {
        'Segment_Type': ['apic', 'dend'],
        'Total_NMDA_Spikes': [3500.0, 3500.0],
        'Total_NA_Spikes': [5000.0, 5000.1],
        'Total_CA_Spikes': [500, 0.0],
        'Soma_Spike_Rate': [5, 5]
    }
    
    # Define weights for each metric
    # Higher weight = more important in matching
    # Set NA weight lower since you don't care as much about it
    weights = {
        'Total_NMDA_Spikes': 3.0,   # High priority
        'Total_CA_Spikes': 3.0,      # High priority
        'Soma_Spike_Rate': 1.5,      # Medium-high priority
        'Total_NA_Spikes': 0.1       # Low priority (don't care as much)
    }
    
    # Define simulations directory
    sims_dir = "/home/drfrbc/Neural-Modeling/simulations/2025-12-10-19-57-tuning_perisomatic_inh_increased_nexus_excitation"
    
    # Find closest match
    print(f"\nSearching for closest match to target dSpike table...")
    print(f"Target:\n{pd.DataFrame(target_dspike)}\n")
    print(f"Weights: {weights}\n")
    
    closest_dir = find_closest_dspike(target_dspike, sims_dir, weights)
    
    if closest_dir:
        print(f"\n{'='*60}")
        print(f"Closest match found:")
        print(f"{closest_dir}")
        print(f"{'='*60}")
    else:
        print("No matching directory found.")


#/home/drfrbc/Neural-Modeling/simulations/2025-12-10-19-57-tuning_perisomatic_inh_increased_nexus_excitation/sta_Complex_NexInhDen0.1_PeriInhDen0.22_DistBasInhDen0.1_TuftInhDen0.2_TuftDistExcDen7_DistBasLocL5ExcDen3.1_Np5000