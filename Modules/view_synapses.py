# Module for visualizing the generated synaptic inputs
import os
import sys
sys.path.append("../")
sys.path.append("../Modules/")
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from Modules import analysis

# read synapse_data.h5 from our simulations.
def read_synapse_distribution_file(sim_directory):
    """
    Reads the synapse_data.h5 file and loads its datasets into a Pandas DataFrame.

    Parameters:
        sim_directory (str): Path to the directory containing synapse_data.h5.

    Returns:
        pd.DataFrame: A DataFrame where each column corresponds to a dataset in the HDF5 file.
    """
    # Construct the full file path
    synapse_file_path = os.path.join(sim_directory, 'synapse_data.h5')

    # Check if the file exists
    if not os.path.exists(synapse_file_path):
        raise FileNotFoundError(f"File not found: {synapse_file_path}")

    # Dictionary to temporarily store the data
    synapse_data = {}

    # Read the HDF5 file
    with h5py.File(synapse_file_path, 'r') as h5f:
        # Load all datasets into the dictionary
        for key in h5f.keys():
            synapse_data[key] = h5f[key][()]  # Load dataset into memory as NumPy array

    # Convert the dictionary into a Pandas DataFrame
    synapse_df = pd.DataFrame(synapse_data)

    return synapse_df

def read_transfer_impedance_file(sim_directory, loc='soma'):
    # loc can be nexus
    imp_file = os.path.join(sim_directory, f"elec_distance_{loc}.csv")
    impedance_data = pd.read_csv(imp_file)
    return impedance_data

def add_seg_info_to_syn_data(syn_data, seg_data):
    """
    Adds segment information (Distance and section) from seg_data to syn_data
    using the seg_id as a lookup.

    Parameters:
        syn_data (pd.DataFrame): DataFrame containing synapse data with 'seg_id'.
        seg_data (pd.DataFrame): DataFrame containing segment data.

    Returns:
        pd.DataFrame: Updated syn_data with 'Distance' and 'section' columns added.
    """
    # Set 'Unnamed: 0' as the index in seg_data for easy lookup
    seg_data_indexed = seg_data.set_index('Unnamed: 0')

    # Use .loc to map the segment information to syn_data based on seg_id
    syn_data['Distance'] = syn_data['seg_id'].map(seg_data_indexed['Distance'])
    syn_data['section'] = syn_data['seg_id'].map(seg_data_indexed['section'])
    syn_data['soma_trans_imp'] = syn_data['seg_id'].map(seg_data_indexed['soma_trans_imp'])
    syn_data['seg_L'] = syn_data['seg_id'].map(seg_data_indexed['L'])

    # Convert the 'synapse_type' column from bytes to strings
    syn_data['synapse_type'] = syn_data['synapse_type'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return syn_data

def build_synapse_counts_dict(syn_data, seg_id_column):
    """
    Builds a dictionary of synapse counts for each segment ID.

    Parameters:
        syn_data (DataFrame): Data containing synapse information.
        seg_id_column (str): Column name for segment IDs.

    Returns:
        dict: A dictionary mapping segment IDs to synapse counts.
    """
    synapse_counts = Counter(syn_data[seg_id_column])
    return dict(synapse_counts)

def build_synapse_density_dict(syn_data, seg_id_column):
    """
    Builds a dictionary of synapse counts for each segment ID.

    Parameters:
        syn_data (DataFrame): Data containing synapse information.
        seg_id_column (str): Column name for segment IDs.

    Returns:
        dict: A dictionary mapping segment IDs to synapse counts.
    """
    synapse_densities = pd.DataFrame(dict(Counter(syn_data['seg_id'].values)), index = [0]).transpose().reset_index()

    return synapse_densities

def plot_morphology(seg_data, seg_id_column, variable_dict, ax, elevation=20, azimuth=-100, radius_scale=1.0, title='', clim_max=30, cbar_label='Synapse Count'):
    """
    Plots a 3D morphology of segments colored based on a given variable.

    Parameters:
        seg_data (DataFrame): Data containing segment positions and radii.
        seg_id_column (str): The column name for segment IDs.
        variable_dict (dict): Dictionary containing variable values for each segment.
        ax (Axes3D): The 3D axes to plot on.
        elevation (int): Elevation angle for the 3D plot.
        azimuth (int): Azimuth angle for the 3D plot.
        radius_scale (float): Scaling factor for segment radius.
        title (str): Title for the plot.
        clim_max (float): Maximum limit for color normalization.
    """
    if clim_max is None:
        values = np.array(list(variable_dict.values()))
        clim_max = values.mean() + 2 * values.std()

    custom_vmin = 0
    custom_vmax = clim_max
    norm = plt.Normalize(vmin=custom_vmin, vmax=custom_vmax)


    for seg_id, seg in seg_data.iterrows():
        if seg_id != 0:
            x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
            y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
            z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]

            radius = seg['r'] * radius_scale
            value = variable_dict.get(seg_id, 0)
            color = plt.cm.jet(norm(value))

            ax.plot(x_points, z_points, y_points, linewidth=radius, color=color)
    ax.set_title(title)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    # norm = plt.Normalize(vmin=0, vmax=clim_max)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap='jet')#'viridis')
    plt.colorbar(mappable, ax=ax, shrink=0.6, label=cbar_label)