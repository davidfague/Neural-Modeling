import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from neuron_reduce.reducing_methods import (_get_subtree_biophysical_properties, find_space_const_in_cm)
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator
from Modules.cell_model import CellModel
from neuron import h,nrn

#def create_graph(seg_infos, seg_info_list):
#    G = nx.Graph()
#    for seg_info in seg_infos:
#        G.add_node(seg_info['seg_index_global'], attr_dict=seg_info)
#        for adj_segment in seg_info['adjacent_segments']:
#            adjacent_seg_info = find_segment_info(adj_segment, seg_info_list)
#            if adjacent_seg_info:
#                G.add_edge(seg_info['seg_index_global'], adjacent_seg_info['seg_index_global'])
#    return G
#
#def find_segment_info(segment, seg_info_list):
#    return next((info for info in seg_info_list if info['seg'] == segment), None)
#
#def get_elec_length(seg, frequency):
#    cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=seg['sec']), frequency)
#    cable_space_const_in_cm = find_space_const_in_cm(seg['seg_diam']/10000, rm, ra)
#    return seg['seg_L']/(cable_space_const_in_cm*10000)
#
#def get_euclidean_distance(seg1, seg2):
#    point1 = np.array([seg1['p0.5_x3d'], seg1['p0.5_y3d'], seg1['p0.5_z3d']])
#    point2 = np.array([seg2['p0.5_x3d'], seg2['p0.5_y3d'], seg2['p0.5_z3d']])
#    return np.linalg.norm(point1 - point2)
    
#
#def compute_electrotonic_distance(G, seg1, seg2, frequency):
#    # Calculate the initial distance (half of seg1's electrotonic length)
#    start_dist = get_elec_length(seg1, frequency) / 2
#    # Use NetworkX to get the shortest path between seg1 and seg2
#    path = nx.shortest_path(G, seg1['seg_index_global'], seg2['seg_index_global'])
#    # Accumulate the electrotonic lengths along the path
#    for seg_index in path[1:-1]:  # Exclude the start and end segments
#        seg = find_segment_info(seg_index)
#        start_dist += get_elec_length(seg, frequency)
#    # Add half of the electrotonic length of seg2 to the final distance
#    end_dist = get_elec_length(seg2, frequency) / 2
#    return start_dist + end_dist
#
#def calc_distance(G, seg1, seg2, frequency, use_euclidean):
#    # # consider removing check for faster computation
#    # if not nx.has_path(G, seg1['seg_index_global'], seg2['seg_index_global']): # check
#    #     raise ValueError(f"No path between seg1 (ID {seg1['seg_index_global']}) and seg2 (ID {seg2['seg_index_global']})")
#    return get_euclidean_distance(seg1, seg2) if use_euclidean else compute_electrotonic_distance(G, seg1, seg2, frequency)
#
#def cluster_synapses(synapses, n_clusters, seg_info_list, frequency=0, use_euclidean=False):
#    segment_infos = [find_segment_info(synapse.segment, seg_info_list) for synapse in synapses] # list for synapses
#    G = create_graph(segment_infos, seg_info_list)
#    distances = [[calc_distance(G, seg_info1, seg_info2, frequency, use_euclidean) for seg_info2 in segment_infos] for seg_info1 in segment_infos]
#    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distances)
#    segment_cluster_map = {seg_info['seg']: cluster for seg_info, cluster in zip(segment_infos, kmeans.labels_)}
#    synapse_cluster_map = {synapse: segment_cluster_map[synapse.segment] for synapse in synapses}
#    return synapse_cluster_map, segment_cluster_map
#TODO: relocate/remove above code

###############################classes
class FunctionalGroup:
  
	def __init__(self, target_segment_indices: list, target_segments_coordinates: np.ndarray, firing_rate_profile: np.ndarray = None, name: str = None, cluster_center:np.ndarray = None):
		'''
		Class representing a functional group of presynaptic cells that provide correlated inputs.
		Parameters:
		----------
		target_segment_indices: list
			List of all segments targeted by this FunctionalGroup according to the segment's index in CellModel.seg_info.
		firing_rate_profile: np.ndarray
			Firing rate profile used to correlate inputs from this FunctionalGroup.
		name: str = None
			Name of the functional group.
		'''
		self.target_segment_indices = target_segment_indices
		self.target_segments_coordinates = target_segments_coordinates
		self.firing_rate_profile = firing_rate_profile 
		self.name = name
		self.cluster_center = cluster_center
		self.presynaptic_cells = [] # list to store PresynapticCell objects within this FunctionalGroup

class PresynapticCell:
  
	def __init__(self, target_segment_indices: list = [], spike_train: np.ndarray = [], name: str = None, cluster_center: np.ndarray = None, cell_type: str = None):
		'''
		Class representing a functional group of presynaptic cells that provide correlated inputs.
		Parameters:
		----------
		target_segment_indices: list
			List of all segments targeted by this FunctionalGroup according to the segment's index in CellModel.seg_info.
		firing_rate_profile: np.ndarray
			Firing rate profile used to correlate inputs from this FunctionalGroup.
		name: str = None
			Name of the functional group.
		'''
		self.target_segment_indices = target_segment_indices
		self.synapses = []
		self.spike_train = spike_train
		self.name = name
		self.cluster_center=cluster_center
		self.mean_firing_rate = None
		self.cell_type = cell_type
############################################################
#functions
def create_functional_groups_of_presynaptic_cells(segments_coordinates: np.ndarray, 
                                                 n_functional_groups: int, 
                                                 n_presynaptic_cells_per_functional_group: int, 
                                                 name_prefix: str, cell: CellModel, synapses: list,
                                                 **kwargs):
                                                 
    # Cluster cell segments into functional groups
    seg_id_to_functional_group_index, functional_group_cluster_centers = cluster_segments(segments_coordinates=segments_coordinates, n_clusters=n_functional_groups)
    #print("Finish Clustering FuncGroups")
    # assemble functional groups and their presynaptic cells
    functional_groups = create_functional_groups(seg_id_to_functional_group_index=seg_id_to_functional_group_index, cell=cell, name_prefix=name_prefix, segments_coordinates=segments_coordinates, n_presynaptic_cells_per_functional_group=n_presynaptic_cells_per_functional_group, functional_group_cluster_centers=functional_group_cluster_centers)
    #print("Finish Creating FuncGroups and PreCells")
    # map synapses to PresynapticCells
    map_synapses_to_PresynapticCells(synapses=synapses, functional_groups=functional_groups, cell=cell)
    #print("Finish Mapping PreCells")
    # Assign syn_params to PresynapticCells
#    if kwards.get('syn_params', None) is not None: # not implemented
#      assign_syn_params_to_PresynapticCells(synapses=synapses, syn_params=kwargs.get('syn_params', None), syn_mod=kwargs.get('syn_mod', None), cell=kwargs.get('cell', None))
    # Calculate spike train for each cluster (implement this in SpikeGenerator)
    generate_spike_train_for_functional_groups(functional_groups=functional_groups, **kwargs)
    #print("Finish Generating Spike trains for FuncGroups and PreCells")
    return functional_groups

def get_euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def cluster_segments(segments_coordinates: np.ndarray, n_clusters: int):
  km = KMeans(n_clusters = n_clusters)
  seg_id_to_cluster_index = km.fit_predict(segments_coordinates)
  cluster_centers = km.cluster_centers_
  return seg_id_to_cluster_index, cluster_centers
  
def create_functional_groups(seg_id_to_functional_group_index: list, segments_coordinates: np.ndarray, n_presynaptic_cells_per_functional_group: int, cell: CellModel, name_prefix: str, functional_group_cluster_centers: np.ndarray):
  # create functional groups
  functional_groups = []
  for functional_group_index in np.unique(seg_id_to_functional_group_index):
    functional_group_cluster_center=functional_group_cluster_centers[functional_group_index]
    # gather functional group segments
    functional_group_target_segments_indices = []
    for seg_ind, seg in enumerate(cell.seg_info):
      if seg_id_to_functional_group_index[seg_ind] == functional_group_index: # segment is in functional group targets
        functional_group_target_segments_indices.append(seg_ind)
    # create FunctionalGroup object
    functional_group_name = name_prefix +str(functional_group_index)
    functional_group_target_segments_coordinates = segments_coordinates[functional_group_target_segments_indices]
    functional_group = FunctionalGroup(target_segment_indices = functional_group_target_segments_indices, name = functional_group_name, target_segments_coordinates = functional_group_target_segments_coordinates, cluster_center = functional_group_cluster_center)
    
    functional_groups.append(functional_group)
    create_presynaptic_cells(functional_group_target_segments_coordinates=functional_group_target_segments_coordinates, n_presynaptic_cells_per_functional_group=n_presynaptic_cells_per_functional_group, functional_group_name=functional_group_name, functional_group=functional_group)
    
  return functional_groups
  
def create_presynaptic_cells(functional_group_target_segments_coordinates: np.ndarray, n_presynaptic_cells_per_functional_group: int, functional_group_name: str, functional_group: FunctionalGroup):
  # cluster functional group segments into presynaptic cells
  seg_ind_to_presynaptic_cell_index, presynaptic_cluster_centers = cluster_segments(segments_coordinates = functional_group_target_segments_coordinates, n_clusters = n_presynaptic_cells_per_functional_group)
  #create presynaptic cells
  for presynaptic_cell_index in np.unique(seg_ind_to_presynaptic_cell_index):
    presynaptic_cell_cluster_center = presynaptic_cluster_centers[presynaptic_cell_index]
    # gather presynaptic cell segments
    presynaptic_cell_target_segments_indices = []
    for seg_ind, seg in enumerate(functional_group.target_segment_indices):
      if seg_ind_to_presynaptic_cell_index[seg_ind] == presynaptic_cell_index: # segment is in presynaptic cell targets
        presynaptic_cell_target_segments_indices.append(seg)
    # create PresynapticCell object
    presynaptic_cell = PresynapticCell(target_segment_indices = presynaptic_cell_target_segments_indices, name = functional_group_name + '_cell' + str(presynaptic_cell_index), cluster_center=presynaptic_cell_cluster_center)
    functional_group.presynaptic_cells.append(presynaptic_cell)
  return presynaptic_cell
  
def map_synapses_to_PresynapticCells(synapses: list, cell: CellModel, functional_groups: list):
  all_mapped_functional_group_seg_indices=[]
  all_mapped_presynaptic_cell_seg_indices=[]
  for functional_group in functional_groups:
    for seg_index in functional_group.target_segment_indices:
      all_mapped_functional_group_seg_indices.append(seg_index)
    for presynaptic_cell in functional_group.presynaptic_cells:
      for seg_index in presynaptic_cell.target_segment_indices:
        all_mapped_presynaptic_cell_seg_indices.append(seg_index)
        
  all_mapped_functional_group_seg_indices=np.sort(all_mapped_functional_group_seg_indices)
  all_mapped_presynaptic_cell_seg_indices=np.sort(all_mapped_presynaptic_cell_seg_indices)
  #print("all_mapped_functional_group_seg_indices:", len(all_mapped_functional_group_seg_indices), all_mapped_functional_group_seg_indices)
  #print("unique functional_group_seg_indices:", len(np.unique(all_mapped_functional_group_seg_indices)))
  #print("all_mapped_presynaptic_cell_seg_indices:", len(all_mapped_presynaptic_cell_seg_indices), all_mapped_presynaptic_cell_seg_indices)
  #print("unique all_mapped_presynaptic_cell_seg_indices:", len(np.unique(all_mapped_presynaptic_cell_seg_indices)))
  
  seg_index_to_synapses = []
  for synapse in synapses: # will need to separate synapses list because we do not want to give exc and inh synapses the same presynaptic cell
    synapse_seg_index = cell.segments.index(synapse.get_segment())
    # match the synapse's segment's index to a presynaptic cell
    synapse_already_mapped=False
    for functional_group in functional_groups:
      if synapse_seg_index in functional_group.target_segment_indices:
        for presynaptic_cell in functional_group.presynaptic_cells:
          if synapse_seg_index in presynaptic_cell.target_segment_indices:
            if not synapse_already_mapped:
              # add the synapse to the presynaptic cell
              presynaptic_cell.synapses.append(synapse)
              synapse_already_mapped=True
            else:
              raise(ValueError("Synapse being mapped to multiple presynaptic cells."))
    if not synapse_already_mapped:
      #print(synapse_seg_index,cell.seg_info[synapse_seg_index])
      raise(RuntimeError("Not able to map synapse to presynaptic cell"))
      
#def assign_syn_params_to_PresynapticCells(synapses: list, syn_params, syn_mod, cell): # not implemented
#    for synapse in synapses:
#        if isinstance(syn_params, list):
#          if 'AMPA' in syn_mod:
#            chosen_params = np.random.choice(syn_params, p=[0.9, 0.1]) # PC2PN and PN2PN
#          elif 'GABA' in syn_mod:
#            if h.distance(synapse.get_segment(), cell.soma[0](0.5)) > 100: # distal inh
#              chosen_params = syn_params[1]
#            elif h.distance(synapse.get_segment(), cell.soma[0](0.5)) < 100: # perisomatic inh
#              chosen_params = syn_params[0]
#        else:
#            chosen_params = syn_params
#        synapse.set_syn_params(chosen_params)
      

def generate_spike_train_for_functional_groups(functional_groups: list, 
                                               method: str, 
                                               random_state, 
                                               spike_generator: SpikeGenerator, 
                                               t: np.ndarray, 
                                               mean_firing_rate: object = None,
                                               proximal_fr_dist: object = None,
                                               distal_fr_dist: object = None,
                                               rhythmicity: bool = False, 
                                               rhythmic_mod: float = None, 
                                               rhythmic_f: float = None, 
                                               spike_trains_to_delay: list = None, 
                                               fr_time_shift: int = None, 
                                               spike_train_dt: float = 1e-3,
                                               soma_coordinates: np.ndarray = None,
                                               proximal_vs_distal_boundary: float = 100
                                               ):   
  for functional_group in functional_groups:
    functional_group.firing_rate_profile = spike_generator.get_firing_rate_profile(t = t, method = method, random_state = random_state, 
				      								  rhythmicity = rhythmicity,
				      								  rhythmic_mod = rhythmic_mod, rhythmic_f = rhythmic_f,
													  spike_trains_to_delay = spike_trains_to_delay,
													  fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
    for presynaptic_cell in functional_group.presynaptic_cells:
      if (proximal_fr_dist is not None) & (distal_fr_dist is not None):
        if get_euclidean_distance(soma_coordinates, presynaptic_cell.cluster_center) < proximal_vs_distal_boundary:
          mean_fr = spike_generator.get_mean_fr(proximal_fr_dist)
        else:
          mean_fr = spike_generator.get_mean_fr(distal_fr_dist)
      elif ((proximal_fr_dist is not None) & (distal_fr_dist is None)) or ((proximal_fr_dist is None) & (distal_fr_dist is not None)):
        raise(ValueError("Must specify either both or neither: proximal & distal firing rate distributions."))
      elif mean_firing_rate is not None:
        mean_fr = spike_generator.get_mean_fr(mean_firing_rate)
      else:
        raise(ValueError("Must specify either mean_firing_rate or both proximal_fr_dist and distal_fr_dist."))
      presynaptic_cell.mean_firing_rate = float(mean_fr)
      spikes = spike_generator.generate_spikes_from_profile(functional_group.firing_rate_profile, mean_fr, random_state)
      #print(mean_fr, functional_group.firing_rate_profile, spikes)
      #spike_trains.append(spikes)
      presynaptic_cell.spike_train.append(spikes)
      for synapse in presynaptic_cell.synapses:
        netcon = spike_generator.set_spike_train(synapse, spikes)
        #print(functional_group.name, presynaptic_cell.name, synapse.synapse_neuron_obj, netcon)
        
