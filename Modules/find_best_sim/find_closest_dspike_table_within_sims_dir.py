#!/usr/bin/env python3
"""
find_closest_dspike_table_wthin_sims_dir.py
Find the simulation directory with the closest dSpike_table.csv to a specified target.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_distance(target_df, candidate_df, weights=None, error_params=None, candidate_segment_df=None, use_raw_differences=False):
    """
    Calculate the distance between target and candidate dSpike tables.
    Uses z-scored absolute errors: computes |target - candidate|, then z-scores within each metric.
    
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
    error_params : dict, optional
        Dictionary mapping column names to {'mean': float, 'std': float}.
        These are the mean and std of absolute errors across all candidates.
    use_raw_differences : bool, optional
        Deprecated parameter, kept for backward compatibility.
    
    Returns:
    --------
    tuple : (float, dict) - Distance and dictionary of component contributions
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
    
    # Default error params if not specified
    if error_params is None:
        error_params = {col: {'mean': 0, 'std': 1} for col in numeric_cols}
    
    # Calculate weighted squared z-scored errors for each numeric column (excluding mean_v, std_v)
    total_distance = 0
    distance_components = {}  # Track contribution from each metric
    
    for col in numeric_cols:
        if col in ['mean_v', 'std_v']:
            continue
        target_vals = merged[f'{col}_target'].values
        candidate_vals = merged[f'{col}_candidate'].values
        
        # Get error statistics
        params = error_params.get(col, {'mean': 0, 'std': 1})
        mean_error = params['mean']
        std_error = params['std']
        
        # Compute absolute error, then z-score it
        abs_errors = np.abs(target_vals - candidate_vals)
        z_scored_errors = (abs_errors - mean_error) / std_error
        
        # Distance is the weighted squared z-scored error
        weight = weights.get(col, 1.0)
        col_distance = weight * np.sum(z_scored_errors ** 2)
        total_distance += col_distance
        distance_components[col] = col_distance
        
        # Debug: track mean absolute error and mean z-scored error
        distance_components[f'{col}_abs_error'] = np.mean(abs_errors)
        distance_components[f'{col}_z_error'] = np.mean(np.abs(z_scored_errors))
    # Now, compare mean_v and std_v by segment (if present in candidate_segment_df)
    # candidate_segment_df is now a direct argument
    if candidate_segment_df is not None:
        for voltage_col in ['mean_v', 'std_v']:
            if voltage_col in candidate_segment_df.columns and voltage_col in target_df.columns:
                voltage_distance = 0
                abs_errors_list = []
                z_errors_list = []
                n_comparisons = 0
                for section_type in target_df['Segment_Type']:
                    seg_mask = candidate_segment_df['section'] == section_type
                    seg_vals = candidate_segment_df.loc[seg_mask, voltage_col].values
                    if seg_vals.size == 0:
                        continue
                    idx = list(target_df['Segment_Type']).index(section_type)
                    target_val = target_df[voltage_col].iloc[idx]
                    
                    # Compute absolute error, then z-score it
                    params = error_params.get(voltage_col, {'mean': 0, 'std': 1})
                    mean_error = params['mean']
                    std_error = params['std']
                    
                    abs_errors = np.abs(target_val - seg_vals)
                    z_scored_errors = (abs_errors - mean_error) / std_error
                    abs_errors_list.extend(abs_errors)
                    z_errors_list.extend(z_scored_errors)
                    
                    # Average the squared z-scored errors
                    voltage_distance += np.mean(z_scored_errors ** 2)
                    n_comparisons += 1
                
                if n_comparisons > 0:
                    weight = weights.get(voltage_col, 1.0)
                    col_distance = weight * voltage_distance
                    total_distance += col_distance
                    distance_components[voltage_col] = col_distance
                    distance_components[f'{voltage_col}_abs_error'] = np.mean(abs_errors_list)
                    distance_components[f'{voltage_col}_z_error'] = np.mean(np.abs(z_errors_list))
    
    dist = np.sqrt(total_distance)
    if np.isnan(dist):
        return float('inf'), {}
    
    # Return both distance and breakdown for debugging
    return dist, distance_components


def find_closest_dspike(target_dspike_dict, sims_dir, weights=None, top_n=1):
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
    top_n : int, optional
        Number of top matches to return. Default is 1. If greater than number
        of simulations, returns all simulations.
    
    Returns:
    --------
    list : List of (distance, Path) tuples for top N matches, sorted best to worst.
           If top_n=1, returns single Path (for backward compatibility).
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
    segment_dfs = []
    for dspike_file in dspike_files:
        try:
            candidate_df = pd.read_csv(dspike_file, index_col=0)
            seg_data_path = dspike_file.parent / 'segment_data.csv'
            candidate_segment_df = None
            if seg_data_path.exists():
                seg_df = pd.read_csv(seg_data_path)
                candidate_segment_df = seg_df[['section', 'mean_v', 'std_v']].copy()
            all_candidates.append(candidate_df)
            segment_dfs.append(candidate_segment_df)
        except Exception as e:
            print(f"  Error reading {dspike_file}: {e}")
    if not all_candidates:
        print("No valid candidate files found")
        return None
    # Compute error statistics (mean and std of absolute errors across all candidates)
    error_params = {}
    for col in numeric_cols:
        all_errors = []
        for candidate_df, candidate_segment_df in zip(all_candidates, segment_dfs):
            merged = target_df.merge(candidate_df, on='Segment_Type', suffixes=('_target', '_candidate'))
            if col in ['mean_v', 'std_v'] and candidate_segment_df is not None:
                # For voltage columns, compute errors per segment
                for section_type in target_df['Segment_Type']:
                    seg_mask = candidate_segment_df['section'] == section_type
                    seg_vals = candidate_segment_df.loc[seg_mask, col].values
                    if seg_vals.size > 0:
                        idx = list(target_df['Segment_Type']).index(section_type)
                        target_val = target_df[col].iloc[idx]
                        abs_errors = np.abs(target_val - seg_vals)
                        all_errors.extend(abs_errors)
            elif f'{col}_target' in merged.columns and f'{col}_candidate' in merged.columns:
                # For non-voltage columns, compute errors from table
                target_vals = merged[f'{col}_target'].values
                candidate_vals = merged[f'{col}_candidate'].values
                abs_errors = np.abs(target_vals - candidate_vals)
                all_errors.extend(abs_errors)
        if all_errors:
            mean_error = np.mean(all_errors)
            std_error = np.std(all_errors, ddof=1)  # Use ddof=1 for sample std
            # Use std if it's not too small, otherwise use 1.0 to avoid division by zero
            error_params[col] = {'mean': mean_error, 'std': std_error if std_error > 1e-6 else 1.0}
        else:
            error_params[col] = {'mean': 0, 'std': 1.0}
    print(f"Error statistics (mean and std of absolute errors): {error_params}\n")
    
    # Second pass: calculate distance for each file using error statistics
    results = []  # List of (distance, path) tuples
    n_filtered = 0
    for dspike_file, candidate_df, candidate_segment_df in zip(dspike_files, all_candidates, segment_dfs):
        try:
            # Apply threshold filtering if specified
            if thresholds is not None:
                skip = False
                for col, (min_val, max_val) in thresholds.items():
                    if col in candidate_df.columns:
                        # Check all rows for this column
                        col_vals = candidate_df[col].values
                        if np.any(col_vals < min_val) or np.any(col_vals > max_val):
                            skip = True
                            n_filtered += 1
                            break
                if skip:
                    continue
            
            distance, components = calculate_distance(target_df, candidate_df, weights, error_params, candidate_segment_df)
            # Print distance with component breakdown (show sqrt of components for interpretability)
            component_str = ", ".join([f"{k}: {np.sqrt(v):.2f}" for k, v in sorted(components.items()) if not (k.endswith('_abs_error') or k.endswith('_z_error'))])
            error_str = ", ".join([f"{k.replace('_abs_error', '')}: {v:.2f}" for k, v in sorted(components.items()) if k.endswith('_abs_error')])
            z_error_str = ", ".join([f"{k.replace('_z_error', '')}: {v:.2f}σ" for k, v in sorted(components.items()) if k.endswith('_z_error')])
            print(f"  {dspike_file.parent.name}: distance = {distance:.4f}")
            print(f"    √Components: {component_str}")
            if error_str:
                print(f"    Abs-errors: {error_str}")
            if z_error_str:
                print(f"    Z-errors: {z_error_str}")
            results.append((distance, dspike_file.parent))
        except Exception as e:
            print(f"  Error processing {dspike_file}: {e}")
    
    if thresholds and n_filtered > 0:
        print(f"\nFiltered out {n_filtered} simulations based on threshold criteria")
    
    if not results:
        return None
    
    # Sort by distance (ascending) and get top N
    results.sort(key=lambda x: x[0])
    n_results = min(top_n, len(results))
    top_results = results[:n_results]
    
    # Return single path for backward compatibility if top_n=1
    if top_n == 1:
        return top_results[0][1]
    
    return top_results


if __name__ == "__main__":
    # Define target dSpike table
    target_dspike = {
        'Segment_Type': ['apic', 'dend'],
        'Total_NMDA_Spikes': [4152.1, 3385.1],
        'Total_NA_Spikes': [2427.4, 2629.0],
        'Total_CA_Spikes': [59.0, 0.0],
        'Soma_Spike_Rate': [4.1, 4.1],
        'mean_v': [-50.07, -46.61],
        'std_v': [15.76, 17.18]
    }
    # Define weights for each metric
    # Higher weight = more important in matching
    # NOTE: Weights alone may not be enough if no simulations meet all criteria
    # Consider using thresholds (below) to filter unacceptable simulations
    weights = {
        'Total_NMDA_Spikes': 3.00,   
        'Total_CA_Spikes': 3.0,      
        'Soma_Spike_Rate': 100.00,   # Very high weight for soma rate
        'Total_NA_Spikes': 0.1,      
        'mean_v': 1.0,               
        'std_v': 0.1                 
    }
    sims_dir = "/home/drfrbc/Neural-Modeling/simulations/2026-01-15-11-31-Tuning_dSpikes_with_covary_params_smaller_sweep"
    
    # Number of top matches to display
    top_n = 5
    
    # Optional: Set hard thresholds to filter simulations
    # This ensures simulations meet minimum criteria before distance calculation
    # Recommended if weights alone don't prioritize critical metrics enough
    thresholds = {
        'Soma_Spike_Rate': (3.0, 7.0),  # Only consider 2-7 Hz (target is ~3-5 Hz)
    }
    # Uncomment to disable filtering:
    # thresholds = None
    print(f"\nSearching for top {top_n} closest matches to target dSpike table...")
    print(f"Target:\n{pd.DataFrame(target_dspike)}\n")
    print(f"Weights: {weights}\n")
    results = find_closest_dspike(target_dspike, sims_dir, weights, top_n=top_n)
    
    if results:
        if isinstance(results, Path):
            # Single result (backward compatibility)
            print(f"\n{'='*60}")
            print(f"Closest match found:")
            print(f"{results}")
            print(f"{'='*60}")
        else:
            # Multiple results - print worst to best (best last)
            print(f"\n{'='*80}")
            print(f"Top {len(results)} matches (best match last):")
            print(f"{'='*80}")
            for i, (distance, sim_dir) in enumerate(reversed(results), 1):
                rank = len(results) - i + 1
                print(f"\nRank {rank}: distance = {distance:.4f}")
                print(f"{sim_dir}")
            print(f"\n{'='*80}")
    else:
        print("No matching directory found.")


#/home/drfrbc/Neural-Modeling/simulations/2025-12-10-19-57-tuning_perisomatic_inh_increased_nexus_excitation/sta_Complex_NexInhDen0.1_PeriInhDen0.22_DistBasInhDen0.1_TuftInhDen0.2_TuftDistExcDen7_DistBasLocL5ExcDen3.1_Np5000