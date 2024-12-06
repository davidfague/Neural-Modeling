# computes the correlation matrix for LFP between models
import sys
sys.path.append("../")
sys.path.append("../Modules/")
sys.path.append("../cell_inference/")

from config import params

import os
import analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import warnings

from ecp import ECP

def remove_nans_and_report_indices(data_list):
    # Find indices of nan values in the list
    nan_indices = [i for i, value in enumerate(data_list) if np.isnan(value)]
    
    # Check if the list contains any nan values
    if nan_indices:
        # Issue a warning with the indices of nan values
        warnings.warn(f"nan values found at indices {nan_indices}. They will be removed.", UserWarning)
        
        # Remove nan values from the list
        cleaned_list = [value for value in data_list if not np.isnan(value)]
    else:
        # If no nan values, return the original list
        cleaned_list = data_list
    
    return cleaned_list

def load_lfp(sim_directory):
  '''loads lfp from transmembrane current, cell morphology, and electrode positions'''
  # load transmembrane current
  i_membrane = analysis.DataReader.read_data(sim_directory, "i_membrane_")
  
  # get cell morphology
  morph = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
  
  # organize segment cordinates
  pc = morph[['pc_0', 'pc_1', 'pc_2']].to_numpy()
  dl = morph[['dl_0', 'dl_1', 'dl_2']].to_numpy()
  r = morph['r'].to_numpy()
  morph = {
        'pc': pc,
        'dl': dl,
        'r': r
  }
  print(morph)
  # get electrode coordinates
  elec_pos = params.ELECTRODE_POSITION
  
  # create ECP object
  ecp = ECP(i_membrane, seg_coords=morph, min_distance=params.MIN_DISTANCE)
  
  # set ecp obj electrode positions
  elec_pos=params.ELECTRODE_POSITION
  ecp.set_electrode_positions(elec_pos)
  
  # calculate lfp from electrodes
  lfp = ecp.calc_ecp().T  # Unit: mV
  
  return lfp
  
def compute_correlations(lfp1, lfp2): # deprecated
  # given two sets of n lfp traces s1_n(t) and s2_n(t), compute the correlation between [s1_1(t) and s2_1(t), s1_2(t) and s2_2(t), ... , s1_n(t) and s2_n(t)]
  # return a correlation array of length n.
  # expecting lfp = np.array of np.arrays of floats
  
  # Initialize an empty list to store correlation values
  correlations = []
  
  # Iterate over pairs of LFP traces
  for trace1, trace2 in zip(lfp1, lfp2):
      # Compute correlation for the current pair of traces
      correlation = np.corrcoef(trace1, trace2)[0, 1]
      correlations.append(correlation)
  
  correlations = remove_nans_and_report_indices(correlations)
  
  return correlations
  
def compute_cross_correlation(lfp1, lfp2):
    '''given two sets of n lfp traces s1_n(t) and s2_n(t), compute the correlation between [s1_1(t) and s2_1(t), s1_2(t) and s2_2(t), ... , s1_n(t) and s2_n(t)]
    return a correlation array of length n.
    expecting lfp = np.array of np.arrays of floats '''
    # make sure they are sized the same
    if len(lfp1) != len(lfp2):
      raise(ValueError(f"len(lfp1) != len(lfp2): {len(lfp1)}, {len(lfp2)}. The number of electrodes does not match."))
      
    if len(lfp1[0]) != len(lfp2[0]):
      raise(ValueError(f"len(lfp1[0]) != len(lfp2[0]): {len(lfp1[0])}, {len(lfp2[0])}. The durations of recording do not match."))
      
    # Initialize lists to store correlation values
    pearson_correlations = []
    cross_correlation_results = []
    full_cross_correlations = []
    
    # Iterate over pairs of LFP traces
    for trace1, trace2 in zip(lfp1, lfp2): # trace1 and trace2 have the same index in their respective lfp set.
        # Compute Pearson correlation
        pearson_corr = np.corrcoef(trace1, trace2)[0, 1]
        pearson_correlations.append(pearson_corr)
        
        # Compute cross-correlation
        cross_corr = np.correlate(trace1 - np.mean(trace1), trace2 - np.mean(trace2), mode='full')
        lag_indices = np.arange(-len(trace1) + 1, len(trace2))
        
        # Normalize the cross-correlation
        cross_corr /= np.sqrt(np.sum((trace1 - np.mean(trace1))**2) * np.sum((trace2 - np.mean(trace2))**2))
        
        # Find the peak correlation and its corresponding lag
        max_corr_index = np.argmax(cross_corr)
        max_correlation = cross_corr[max_corr_index]
        best_lag = lag_indices[max_corr_index]
        full_cross_correlations.append((lag_indices, cross_corr))
        
        cross_correlation_results.append((max_correlation, best_lag))
        
    pearson_correlations = remove_nans_and_report_indices(pearson_correlations)
    
    return pearson_correlations, cross_correlation_results, full_cross_correlations

def plot_correlations(correlations, sim_directory1, sim_directory2): # deprecated
    # Generate a sequence of trace indices (1-based for readability)
    trace_indices = range(1, len(correlations) + 1)
    
    # Create a scatter plot of correlation values
    plt.figure(figsize=(10, 6))
    plt.scatter(trace_indices, correlations, color='blue')
    
    # Adding title and labels
    plt.title(f"Correlation between Corresponding LFP of {sim_directory1.split('/')[-1]} and {sim_directory2.split('/')[-1]}")
    plt.xlabel('Electrode Index')
    plt.ylabel('Correlation Coefficient')
    
    # Show grid for better readability
    plt.grid(True)
    
    # Display the plot
    plt.show()
   
def plot_cross_correlation_results(cross_correlation_results, sim_directory1, sim_directory2):
    # Extract peak correlations and lags
    peak_correlations = [result[0] for result in cross_correlation_results]
    lags = [result[1] for result in cross_correlation_results]
    
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot peak correlations
    axs[0].scatter(range(1, len(peak_correlations) + 1), peak_correlations, color='blue')
    axs[0].set_title(f"Peak Cross-Correlation between {sim_directory1.split('/')[-1]} and {sim_directory2.split('/')[-1]}")
    axs[0].set_xlabel('Electrode Index')
    axs[0].set_ylabel('Peak Cross-Correlation')
    axs[0].grid(True)
    
    # Plot lags
    axs[1].scatter(range(1, len(lags) + 1), lags, color='red')
    axs[1].set_title(f"Corresponding Lags for Peak Cross-Correlation")
    axs[1].set_xlabel('Electrode Index')
    axs[1].set_ylabel('Lag (samples)')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_last_cross_correlations(full_cross_correlations, num_plots=3):
    """
    Plot the cross-correlation functions for the last few LFP trace pairs.

    Parameters:
    - full_cross_correlations: List of tuples containing (lag_indices, cross_corr) for each trace pair.
    - num_plots: Number of last trace pairs to plot.
    """
    # Ensure we don't try to plot more pairs than available
    num_plots = min(num_plots, len(full_cross_correlations))
    
    # Select the last few cross-correlation results
    last_few = full_cross_correlations[-num_plots:]
    
    # Plot each selected cross-correlation
    for i, (lags, cross_corr) in enumerate(last_few, start=1):
        plt.figure(figsize=(8, 4))
        plt.plot(lags, cross_corr)
        plt.title(f"Cross-Correlation for Trace Pair {len(full_cross_correlations) - num_plots + i}")
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.grid(True)
        plt.show()
  
def compare_simulations(sim_directory1, sim_directory2):
  # load lfp from simulations
  lfp1 = load_lfp(sim_directory1)
  lfp2 = load_lfp(sim_directory2)
  
  # compute the correlations of the lfps
  pearons_correlations, cross_correlation_results, full_cross_correlations = compute_cross_correlation(lfp1, lfp2)
  
  # perform other metrics on pearson correlations
  #print(f"pearson_correlations mean: {np.mean(pearson_correlations)}")
  #print(f"pearson_correlations variance: {np.var(pearson_correlations)}")
  # plot the correlations of the lfps
  #plot_correlations(correlations, sim_directory1, sim_directory2)
  
  #print(cross_correlation_results)
  plot_cross_correlation_results(cross_correlation_results, sim_directory1, sim_directory2)
  plot_last_cross_correlations(full_cross_correlations)
  print(lfp1)
  print(type(lfp1))
  print(type(lfp1[0]))
  print(type(lfp1[0][0]))
  

if __name__ == "__main__":
    # Extract directories from command line arguments
    sim_directories = []
    for arg in sys.argv[1:]:  # Skip the first argument, which is the script name
        if arg.startswith('-'):  # Skip any argument that is an option
            continue
        sim_directories.append(arg)
    
    # Generate all combinations of the directories in pairs and compare them
    for sim_dir1, sim_dir2 in combinations(sim_directories, 2):
        compare_simulations(sim_dir1, sim_dir2)
