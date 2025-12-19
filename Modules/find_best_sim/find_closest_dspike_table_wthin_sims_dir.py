#!/usr/bin/env python3
"""
Find the simulation directory with the closest dSpike_table.csv to a specified target.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_distance(target_df, candidate_df, weights=None, normalization_factors=None):
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
    normalization_factors : dict, optional
        Dictionary mapping column names to normalization values (typically std or range).
        Used to normalize differences to a common scale across columns.
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
    
    # Default normalization factors to 1.0 if not specified
    if normalization_factors is None:
        normalization_factors = {col: 1.0 for col in numeric_cols}
    
    # Calculate weighted squared differences for each numeric column
    total_distance = 0
    for col in numeric_cols:
        target_vals = merged[f'{col}_target'].values
        candidate_vals = merged[f'{col}_candidate'].values
        
        # Normalize by the standard deviation/range across all candidates
        # This puts all columns on a comparable scale
        norm_factor = normalization_factors.get(col, 1.0)
        normalized_diff = (target_vals - candidate_vals) / norm_factor
        
        # Apply weight (default to 1.0 if not specified)
        weight = weights.get(col, 1.0)
        total_distance += weight * np.sum(normalized_diff ** 2)
    # Optionally consider mean_v and std_v if present in target_df and candidate_df
    for voltage_col in ['mean_v', 'std_v']:
        if voltage_col in target_df.columns and voltage_col in candidate_df.columns:
            target_vals = merged[f'{voltage_col}_target'].values
            candidate_vals = merged[f'{voltage_col}_candidate'].values
            norm_factor = normalization_factors.get(voltage_col, 1.0)
            normalized_diff = (target_vals - candidate_vals) / norm_factor
            weight = weights.get(voltage_col, 1.0)
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
    # First pass: collect all candidate data to compute normalization factors
    all_candidates = []
    # Consider mean_v and std_v columns if present in target_df
    numeric_cols = [col for col in target_df.columns if col != 'Segment_Type']
    for voltage_col in ['mean_v', 'std_v']:
        if voltage_col in target_df.columns:
            numeric_cols.append(voltage_col)
    for dspike_file in dspike_files:
        try:
            candidate_df = pd.read_csv(dspike_file, index_col=0)
            # Try to load segment_data.csv from the same directory
            seg_data_path = dspike_file.parent / 'segment_data.csv'
            if seg_data_path.exists():
                seg_df = pd.read_csv(seg_data_path)
                # Only keep columns we care about
                seg_df = seg_df.rename(columns={
                    'sec_type_precise': 'Segment_Type',
                    'mean_v': 'mean_v',
                    'std_v': 'std_v'
                })
                # Only keep relevant columns
                seg_df = seg_df[['Segment_Type', 'mean_v', 'std_v']].copy()
                # Group by Segment_Type and take mean (in case multiple rows per type)
                seg_df = seg_df.groupby('Segment_Type').mean().reset_index()
                # Merge mean_v and std_v into candidate_df on Segment_Type
                candidate_df = candidate_df.merge(seg_df, on='Segment_Type', how='left')
            all_candidates.append(candidate_df)
        except Exception as e:
            print(f"  Error reading {dspike_file}: {e}")
    if not all_candidates:
        print("No valid candidate files found")
        return None
    # Compute normalization factors (standard deviation across all candidates for each column)
    normalization_factors = {}
    for col in numeric_cols:
        all_values = []
        for candidate_df in all_candidates:
            merged = target_df.merge(candidate_df, on='Segment_Type')
            if col in merged.columns:
                all_values.extend(merged[col].values)
        if all_values:
            std = np.std(all_values)
            # Use std if it's not too small, otherwise use mean absolute value
            normalization_factors[col] = std if std > 1e-6 else (np.mean(np.abs(all_values)) + 1e-6)
        else:
            normalization_factors[col] = 1.0
    print(f"Normalization factors: {normalization_factors}\n")
    
    # Second pass: calculate distance for each file using normalization factors
    min_distance = float('inf')
    closest_dir = None
    for dspike_file, candidate_df in zip(dspike_files, all_candidates):
        try:
            distance = calculate_distance(target_df, candidate_df, weights, normalization_factors)
            print(f"  {dspike_file.parent.name}: distance = {distance:.4f}")
            if distance < min_distance:
                min_distance = distance
                closest_dir = dspike_file.parent
        except Exception as e:
            print(f"  Error processing {dspike_file}: {e}")
    return closest_dir


if __name__ == "__main__":
    # Define target dSpike table
    target_dspike = {
        'Segment_Type': ['apic', 'dend'],
        'Total_NMDA_Spikes': [4000, 4000.0],
        'Total_NA_Spikes': [2000.0, 2000.1],
        'Total_CA_Spikes': [300, 0.0],
        'Soma_Spike_Rate': [3, 3],
        'mean_v': [-65, -65],   # Example desired mean voltages
        'std_v': [25, 25]         # Example desired std voltages
    }
    # Define weights for each metric
    # Higher weight = more important in matching
    # Set NA weight lower since you don't care as much about it
    weights = {
        'Total_NMDA_Spikes': 3.0,   # High priority
        'Total_CA_Spikes': 3.0,      # High priority
        'Soma_Spike_Rate': 1.00,     # Medium-high priority
        'Total_NA_Spikes': 0.1,      # Low priority
        'mean_v': 2.0,               # User can set this
        'std_v': 0.0                 # User can set this
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