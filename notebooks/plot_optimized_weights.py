#!/usr/bin/env python3

"""
Plot Optimized Weight Distributions

This script loads optimization results and plots the weight distributions
for each synapse type and location combination.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from datetime import datetime

def load_optimization_results(results_dir):
    """Load optimization histories from the results directory."""
    # Find the most recent results directory if multiple exist
    if not os.path.isdir(results_dir):
        # Look for directories matching the pattern
        base_dir = os.path.dirname(results_dir)
        matching_dirs = [d for d in os.listdir(base_dir) 
                        if d.startswith('AA_PSC_tuning_results_')]
        if not matching_dirs:
            raise FileNotFoundError(f"No results directory found in {base_dir}")
        # Sort by timestamp and get the most recent
        results_dir = os.path.join(base_dir, sorted(matching_dirs)[-1])
    
    # Load optimization histories
    history_path = os.path.join(results_dir, 'optimization_histories.pkl')
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"No optimization histories found at {history_path}")
    
    with open(history_path, 'rb') as f:
        optimization_histories = pickle.load(f)
    
    return optimization_histories, results_dir

def plot_weight_distribution(weights, synapse_type, location_type, save_dir):
    """Plot the weight distribution with both histogram and fitted PDF."""
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(weights, bins=50, density=True, alpha=0.7, 
                              label='Histogram')
    
    # Fit normal distribution
    mu, std = stats.norm.fit(weights)
    x = np.linspace(min(weights), max(weights), 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', lw=2, label=f'Normal fit (μ={mu:.3f}, σ={std:.3f})')
    
    # Fit log-normal distribution
    shape, loc, scale = stats.lognorm.fit(weights)
    p_lognorm = stats.lognorm.pdf(x, shape, loc, scale)
    plt.plot(x, p_lognorm, 'g--', lw=2, 
            label=f'Log-normal fit (μ={np.log(scale):.3f}, σ={shape:.3f})')
    
    plt.xlabel('Synapse Weight')
    plt.ylabel('Density')
    plt.title(f'Weight Distribution for {synapse_type} in {location_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot QQ plot
    plt.subplot(1, 2, 2)
    stats.probplot(weights, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal Distribution)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'weight_distribution_{synapse_type}_{location_type}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved weight distribution plot to {plot_path}")
    
    # Return distribution parameters
    return {
        'mean': np.mean(weights),
        'std': np.std(weights),
        'normal_fit': {'mu': mu, 'std': std},
        'lognorm_fit': {'shape': shape, 'loc': loc, 'scale': scale}
    }

def plot_all_weight_distributions(optimization_histories, results_dir):
    """Plot weight distributions for all synapse types and locations."""
    # Create directory for weight distribution plots
    weight_plots_dir = os.path.join(results_dir, 'weight_distributions')
    os.makedirs(weight_plots_dir, exist_ok=True)
    
    # Store distribution parameters
    distribution_params = {}
    
    # Plot for each synapse type and location
    for (synapse_type, location_type), history in optimization_histories.items():
        print(f"\nProcessing {synapse_type} in {location_type}...")
        
        # Get the final optimization result (last entry in history)
        final_result = history[-1]
        weights = final_result['weights']
        
        # Plot weight distribution
        params = plot_weight_distribution(weights, synapse_type, location_type, 
                                        weight_plots_dir)
        
        # Store parameters
        distribution_params[f"{synapse_type}_{location_type}"] = params
    
    # Save distribution parameters
    params_df = pd.DataFrame(distribution_params).T
    params_df.to_csv(os.path.join(weight_plots_dir, 'weight_distribution_parameters.csv'))
    print(f"\nSaved distribution parameters to {weight_plots_dir}/weight_distribution_parameters.csv")

def main():
    # Get the most recent results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, 'AA_PSC_tuning_results')
    
    try:
        # Load optimization results
        optimization_histories, results_dir = load_optimization_results(results_dir)
        
        # Plot all weight distributions
        plot_all_weight_distributions(optimization_histories, results_dir)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 