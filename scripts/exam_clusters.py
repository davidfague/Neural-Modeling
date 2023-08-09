import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from cell_inference.config import params
import constants
from Modules.segment import Segment


output_folder = "output/2023-08-08_16-39-13_seeds_123_87L5PCtemplate[0]_196nseg_108nbranch_59156NCs_29578nsyn"


constants.show_electrodes = False
if constants.show_electrodes:
  elec_pos = params.ELECTRODE_POSITION
else:
  elec_pos = None

# Default view
elev, azim = 90, -90#
  
# Set up the plotting_modes and cluster_types to iterate over
plotting_modes = ['functional_groups', 'presynaptic_cells']
cluster_types = ['exc', 'inh_distributed'] #'inh_soma']  

def load_data(cluster_types, output_folder):
  data = {}
  data["detailed_seg_info"] = pd.read_csv(os.path.join(output_folder, "detailed_seg_info.csv"))
  data["functional_groups"] = {}
  data["presynaptic_cells"] = {}
  
  for cluster_type in cluster_types:
      data["functional_groups"][cluster_type] = pd.read_csv(os.path.join(output_folder, cluster_type + "_functional_groups.csv"))
      data["presynaptic_cells"][cluster_type] = pd.read_csv(os.path.join(output_folder, cluster_type + "_presynaptic_cells.csv"))
  
  return data

def reset_segment_assignments(segments):
    """Reset the assignment attributes of all segments."""
    for seg in segments:
        seg.functional_group_names = []
        seg.functional_group_indices = []
        seg.presynaptic_cell_names = []
        seg.presynaptic_cell_indices = []

def assign_funcgroups_and_precells_to_segments(cluster_type, plotting_mode, segments, data):
    """Assign FuncGroups and PreCells to segments based on the given cluster_type."""
    max_dists = [] # list for cluster spans
    if plotting_mode == 'functional_groups':
      # Iterate over functional groups and assign them to segments
      for _, row in data["functional_groups"][cluster_type].iterrows():
          fg_name = row["name"]
          fg_index = row["functional_group_index"]
          cleaned_strings = row["target_segment_indices"].replace('[', '').replace(']', '').split(',')
          target_indices = [int(x) for x in cleaned_strings]
          max_dists.append(max_distance_for_segments(segments))
          for target_seg_index in target_indices:
              seg = segments[target_seg_index]
              seg.functional_group_names.append(fg_name)
              seg.functional_group_indices.append(fg_index)
    # For presynaptic cells
    elif plotting_mode == 'presynaptic_cells':
      # Iterate over presynaptic cells and assign them to segments
      for _, row in data["presynaptic_cells"][cluster_type].iterrows():
          pc_name = row["name"]
          pc_index = row["presynaptic_cell_index"]
          cleaned_strings = row["target_segment_indices"].replace('[', '').replace(']', '').split(',')
          target_indices = [int(x) for x in cleaned_strings]
          max_dists.append(max_distance_for_segments(segments))
          for target_seg_index in target_indices:
              seg = segments[target_seg_index]
              seg.presynaptic_cell_names.append(pc_name)
              seg.presynaptic_cell_indices.append(pc_index)
            
    mean_distance = np.mean(max_dists)
    std_distance = np.std(max_dists)
    
    print(f"'{cluster_type}' '{plotting_mode}':")
    print(f"Mean of maximum distances: {mean_distance}")
    print(f"Standard deviation of maximum distances: {std_distance}\n")


  # Function to compute pairwise distance between segment endpoints
def pairwise_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + 
                   (point1[1] - point2[1])**2 + 
                   (point1[2] - point2[2])**2)

# Function to compute the maximum distance between two segments
def max_distance_between_segments(seg1, seg2):
    seg1_points = [
        (seg1.p0_x3d, seg1.p0_y3d, seg1.p0_z3d),
        (seg1.p0_5_x3d, seg1.p0_5_y3d, seg1.p0_5_z3d),
        (seg1.p1_x3d, seg1.p1_y3d, seg1.p1_z3d)
    ]
    
    seg2_points = [
        (seg2.p0_x3d, seg2.p0_y3d, seg2.p0_z3d),
        (seg2.p0_5_x3d, seg2.p0_5_y3d, seg2.p0_5_z3d),
        (seg2.p1_x3d, seg2.p1_y3d, seg2.p1_z3d)
    ]
    
    # compute all pairwise distances and return the maximum
    return max(pairwise_distance(p1, p2) for p1 in seg1_points for p2 in seg2_points)

# find the max distance between clustered segments
def max_distance_for_segments(segments):
    n = len(segments)
    if n <= 1:
        return 0
    max_distance = 0
    for i in range(n):
        for j in range(i+1, n):
            dist = max_distance_between_segments(segments[i], segments[j])
            max_distance = max(max_distance, dist)
    return max_distance
      
def plot_segments(segments, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for segment in segments:
        p0 = (segment.p0_x3d, segment.p0_y3d, segment.p0_z3d)
        p0_5 = (segment.p0_5_x3d, segment.p0_5_y3d, segment.p0_5_z3d)
        p1 = (segment.p1_x3d, segment.p1_y3d, segment.p1_z3d)
        color = segment.color
        ax.plot([p0[0], p0_5[0]], [p0[1], p0_5[1]], [p0[2], p0_5[2]], color=color)
        ax.plot([p0_5[0], p1[0]], [p0_5[1], p1[1]], [p0_5[2], p1[2]], color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Set the y-axis to be vertical
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect([1, 3, 1])  # x, y, z
    plt.savefig(save_name)
    plt.close()

def main(cluster_types, plotting_modes, output_folder):
  # read detailed seg info and clustering csvs
  data = load_data(cluster_types=cluster_types, output_folder=output_folder)
  save_path = os.path.join(output_folder, "Analysis Clusters")
  if os.path.exists(save_path):
    print('Directory already exists:',save_path)
  else:
    print('Creating Directory:',save_path)
    os.mkdir(save_path)
  
  num_segments = len(data["detailed_seg_info"])
  print("NUM detailed seg", num_segments)
  detailed_segments=[]
  for i in range(num_segments):
      # Build seg_data
      seg_data = {} # placeholder
      seg = Segment(seg_info = data["detailed_seg_info"].iloc[i], seg_data = {})
      detailed_segments.append(seg)
  
  for cluster_type in cluster_types:
    for plotting_mode in plotting_modes:
      
      # Reset segment assignments
      reset_segment_assignments(detailed_segments)
      print(f'analyzing {cluster_type} {plotting_mode}')
      # Assign segments to FuncGroups or PreCells based on the current cluster_type and plotting mode
      assign_funcgroups_and_precells_to_segments(cluster_type, plotting_mode, detailed_segments, data)
    
      # make color maps for clustering assignment
      num_groups = len(data[plotting_mode][cluster_type])
      group_colors = plt.cm.get_cmap('tab20', num_functional_groups)
      print(f"number of '{cluster_type}' '{plotting_mode}': '{num_groups}'")
      
      for seg in detailed_segments:
        if plotting_mode == 'functional_group' and seg.functional_group_indices:
          seg.color = functional_group_colors(seg.functional_group_indices[0])
        elif plotting_mode == 'presynaptic_cell' and seg.presynaptic_cell_indices:
          seg.color = presynaptic_cell_colors(seg.presynaptic_cell_indices[0])
      
      # plot
      print(f'plotting {cluster_type} {plotting_mode}')
      save_name = os.path.join(save_path, f'{cluster_type}_{plotting_mode}.png')
      plot_segments(detailed_segments, save_name)
  
if __name__ == "__main__":
    main(cluster_types, plotting_modes, output_folder)
    
    