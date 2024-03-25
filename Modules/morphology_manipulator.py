'''
Will start by making specific SA and electrotonic distance calculations
Eventually will manipulate model sections similar to a new version of cable_expander and neuron_reduce
'''
'''
TODO
report deviation from d(3/2) rule at branching points.
'''
from electrotonic_distance import *
from surface_area import record_model_SA_and_L, calc_SA_and_L_beyond_seg
from adjacency import *

class MorphologyManipulator():

  def __init__(self):
    pass
    
  def run(self, cell_model):
    adjacency_matrix = cell_model.compute_directed_adjacency_matrix()
    #print(adjacency_matrix.shape[0])
    #print(f"type(adjacency_matrix): {type(adjacency_matrix)}")
    nexus_seg_index = self.find_nexus_seg(cell_model, adjacency_matrix)
    #print(f"nexus_seg_index: {nexus_seg_index}")
    SA_df, L_df = self.report_SA_and_L(cell_model, nexus_seg_index, adjacency_matrix)
    elec_L_of_tufts = self.report_electrotonic_lengths(cell_model, nexus_seg_index, adjacency_matrix)
    return nexus_seg_index, SA_df, L_df, elec_L_of_tufts
  
  def report_SA_and_L(self, cell_model, nexus_seg, adjacency_matrix):
    SA_df, L_df = record_model_SA_and_L(cell_model)
    tufts_SA, tufts_L = calc_SA_and_L_beyond_seg(cell_model, nexus_seg, adjacency_matrix)
    SA_df['tufts'] = tufts_SA
    L_df['tufts'] = tufts_L
    
    return SA_df, L_df
  
  def report_electrotonic_lengths(self, cell_model, nexus_seg_index, adjacency_matrix):
    terminal_tuft_segment_indices_in_all_list = find_terminal_descendants(adjacency_matrix, nexus_seg_index)
    segments, _ = cell_model.get_segments(['all'])
    #print(f"terminal descendants of nexus: {terminal_tuft_segment_indices_in_all_list}")
    terminal_segments = [segments[i] for i in terminal_tuft_segment_indices_in_all_list]
    #print(f"terminal_segments: {terminal_segments}")
    elec_L_of_tufts = []
    for terminating_segment_id in terminal_tuft_segment_indices_in_all_list:
      elec_L_of_tufts.append(calculate_electrotonic_length_of_dendrite_path(cell_model, nexus_seg_index, terminating_segment_id, adjacency_matrix))
    return elec_L_of_tufts
    
  def find_nexus_seg(self, cell_model, adjacency_matrix):
    all_seg_list, seg_data = cell_model.get_segments(['all'])
    #print(f"seg_data[379].coords['p1_1']: {seg_data[379].coords['p1_1']}")
    y_coords = []
    for i, seg in enumerate(all_seg_list):
      y_coord = seg_data[i].coords["p1_1"].iloc[0] if not seg_data[i].coords["p1_1"].empty else None
      y_coords.append(y_coord)
    #print(f"y_coords: {y_coords}")
    #print(f"all_seg_list: {all_seg_list}")
    apical_segment_indices = [i for i, seg in enumerate(all_seg_list) if 'apic' in str(seg)]
    #print(f"apical_segment_indices: {apical_segment_indices}")
    nexus_index_in_all_list, _ = find_branching_seg_with_most_branching_descendants_in_subset_y(adjacency_matrix, apical_segment_indices, y_coords)
    #print(f"The found apical nexus segment is: {all_seg_list[nexus_index_in_all_list]}")
    return nexus_index_in_all_list
    
  def update_reduced_model_tuft_lengths(self, cell_model):
    detailed_elec_L = [0.6000434984975269, 0.9621854139951519, 0.9251444865227809, 1.1075742022172554, 0.5550602230975482, 0.5714039725117774, 0.7191907854252777, 0.6601947162572058, 0.7156152071735141, 0.7408438237635562, 0.7483024996467823, 0.7392476472595797, 1.1746837683045173, 1.0719345236372346, 1.1039025303214562, 1.2530362169269815, 1.0396898631919669, 1.1108487066324562, 1.1577674524065849, 1.206312304860035, 1.2473527898807388, 1.0342912031404228]
    for i, sec in enumerate(cell_model.apic):
      if i == 0:
        continue # skip trunk
      length_constant = calc_length_constant_in_microns(sec)
      sec.L = detailed_elec_L[i-1]*length_constant
      
  def report_length_constants(self, sections):
    for sec in sections:
      calc_length_constant_in_microns(sec)
      
  def record_MM(self, cell, MM, save_path, expand_cable): # need to separate recording nexus_seg_index from this and create constants to control
    nexus_seg_index, SA_df, L_df, elec_L_of_tufts = self.run(cell)
    nexus_seg_index_file_path = os.path.join(save_path, "nexus_seg_index.txt")
    with open(nexus_seg_index_file_path, "w") as nexus_seg_index_file:
        nexus_seg_index_file.write(f"Nexus Seg Index: {nexus_seg_index}")
    sa_df_to_save = pd.DataFrame(list(SA_df.items()), columns=['Model_Part', 'Surface_Area'])
    sa_df_to_save.to_csv(os.path.join(save_path, "SA.csv"), index=False)
    l_df_to_save = pd.DataFrame(list(L_df.items()), columns=['Model_Part', 'Length'])
    l_df_to_save.to_csv(os.path.join(save_path, "L.csv"), index=False)
    elec_L_of_tufts_file_path = os.path.join(save_path, "elec_L_of_tufts.txt")
    with open(elec_L_of_tufts_file_path, "w") as elec_L_of_tufts_file:
        elec_L_of_tufts_file.write(f"Tuft electrotonic lengths: {elec_L_of_tufts}")
    if expand_cable:
        MM.update_reduced_model_tuft_lengths(cell)
        nexus_seg_index, SA_df, L_df, elec_L_of_tufts = MM.run(cell)
        nexus_seg_index_file_path = os.path.join(save_path, "nexus_seg_index.txt")
        with open(nexus_seg_index_file_path, "w") as nexus_seg_index_file:
            nexus_seg_index_file.write(f"Nexus Seg Index: {nexus_seg_index}")
        sa_df_to_save = pd.DataFrame(list(SA_df.items()), columns=['Model_Part', 'Surface_Area'])
        sa_df_to_save.to_csv(os.path.join(save_path, "SA_after.csv"), index=False)
        l_df_to_save = pd.DataFrame(list(L_df.items()), columns=['Model_Part', 'Length'])
        l_df_to_save.to_csv(os.path.join(save_path, "L_after.csv"), index=False)
        elec_L_of_tufts_file_path = os.path.join(save_path, "elec_L_of_tufts_after.txt")
        with open(elec_L_of_tufts_file_path, "w") as elec_L_of_tufts_file:
            elec_L_of_tufts_file.write(f"Tuft electrotonic lengths: {elec_L_of_tufts}")
    
    
# try to organize into:
    
# --- computations
    
# --- set morphology based on parameters   

# --- record information from provided model