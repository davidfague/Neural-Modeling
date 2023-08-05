import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from neuron_reduce.reducing_methods import (_get_subtree_biophysical_properties, find_space_const_in_cm)
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator
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
############################################################
#functions
def create_functional_presynaptic_cell_groups(segments_coordintes: ,n_functional_groups: int, n_presynaptic_cells_per_functional_group: int, name_prefix: str, ):
  # Cluster cell segments into functional groups
  seg_id_to_functional_group_index = cluster_segments(segments_coordinates = segments_coordinates, n_clusters = n_functional_groups)
  
  # assemble functional groups and their presynaptic cells
  functional_groups = create_functional_groups(seg_id_to_functional_group_index=seg_id_to_functional_group_index, cell=cell, name_prefix=name_prefix)
      
  # map synapses to PresynapticCells
  map_synapses_to_PresynapticCells(synapses_list=synapses_list, functional_groups=functional_groups)
      
  # Calculate spike train for each cluster (implement this in SpikeGenerator)
  generate_spike_train_for_functional_groups(functional_groups=functional_groups, mean_firing_rate=mean_firing_rate,method=method,random_state=random_state)

def cluster_segments(segments_coordinates: , n_clusters: int):
  km = KMeans(n_clusters = n_clusters)
  seg_id_to_cluster_index = km.fit_predict(segment_coordinates)
  return seg_id_to_cluster_index
  
def create_functional_groups(seg_id_to_functional_group_index: list, segment_coordinates: np.ndarray, cell: CellModel, name_prefix: str ):
  # create functional groups
  functional_groups = []
  for functional_group_index in np.unique(seg_id_to_functional_group_index):
    # gather functional group segments
    functional_group_target_segments = []
    for seg_ind, seg in enumerate(cell.seg_info):
      if seg_ind_to_functional_group_index[seg_ind] == functional_group_index: # segment is in functional group targets
        functional_group_target_segments_indices.append(seg_ind)
    # create FunctionalGroup object
    functional_group_name = name_prefix +str(functional_group_index)
    functional_group_target_segments_coordinates = segment_coordinates[functional_group_target_segments_indices]
    functional_group = FunctionalGroup(target_segment_indices = functional_group_target_segment_indices, name = functional_group_name, target_segments_coordinates = functional_group_target_segments_coordinates)
    
    functional_groups.append(functional_group)
    create_presynaptic_cells(functional_group_target_segments_coordinates=functional_group_target_segments_coordinates, n_presynaptic_cells_per_functional_group=n_presynaptic_cells_per_functional_group, functional_group_name=functional_group_name, functional_group=functional_group)
    
  return functional_groups
  
def create_presynaptic_cells(functional_group_target_segments_coordinates: np.ndarray, n_presynaptic_cells_per_functional_group: int, functional_group_name: str, functional_group: FunctionalGroup):
  # cluster functional group segments into presynaptic cells
  seg_ind_to_presynaptic_cell_index = cluster_segments(segments_coordinates = functional_group_target_segments_coordinates, n_clusters = n_presynaptic_cells_per_functional_group)
  #create presynaptic cells
  for presynaptic_cell_index in np.unique(seg_ind_to_presynaptic_cell_index):
    # gather presynaptic cell segments
    presynaptic_cell_target_segments_indices = []
    for seg_ind, seg in enumerate(functional_group.target_segment_indices):
      if seg_ind_to_presynaptic_cell_index[seg_ind] == presynaptic_cell_index: # segment is in presynaptic cell targets
        presynaptic_cell_target_segments_indices.append(seg_ind)
    # create PresynapticCell object
    presynaptic_cell = Cluster(target_segment_indices = presynaptic_cell_target_segments_indices, name = functional_group_name + '_cell' + presynaptic_cell_index)
    functional_group.presynaptic_cells.append(presynaptic_cell)
  return presynaptic_cell
  
def map_synapses_to_PresynapticCells(synapses_list: list, cell: CellModel, functional_groups: list):
  seg_index_to_synapses = []
  for synapse in synapses_list: # will need to separate synapses list because we do not want to give exc and inh synapses the same presynaptic cell
    synapse_seg_index = cell.segments.index(synapse.get_segment())
    # match the synapse's segment's index to a presynaptic cell
    for functional_group in functional_groups:
      if synapse_seg_index in functional_group.target_segment_indices:
      for presynaptic_cell in functional_group.presynaptic_cells:
        if synapse_seg_index in presynaptic_cell.target_segment_indices:
          # add the synapse to the presynaptic cell
          presynaptic_cell.synapses.append(synapse)

def generate_spike_train_for_functional_groups(functional_groups: list, mean_firing_rate: object, method: str, random_state, rhythmicity: bool = False, rhythmic_mod=None, rhythmic_f=None, spikes_trains_to_delay: list = None, fr_time_shift: int = None, spike_train_dt: float = 1e-3)):   
  for functional_group in functional_groups:
    functional_group.firing_rate_profile = spike_generator.get_firing_rate_profile(t = t, method = method, random_state = random_state, 
				      								  rhythmicity = rhythmicity,
				      								  rhythmic_mod = rhythmic_mod, rhythmic_f = rhythmic_f,
													  spike_trains_to_delay = spike_trains_to_delay,
													  fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
    for presynaptic_cell in functional_group.presynaptic_cells:
			mean_fr = spike_generator.get_mean_fr(mean_firing_rate)
			spikes = spike_generator.generate_spikes_from_profile(functional_group.firing_rate_profile, mean_fr, random_state)
			spike_trains.append(spikes)
      for synapse in presynaptic_cell.synapses:
			  netcon = spike_generator.set_spike_train(synapse, spikes)
###############################classes
class FunctionalGroup:
  
	def __init__(self, target_segment_indices: list, target_segments_coordinates: np.ndarray, firing_rate_profile: np.ndarray = None, name: str = None):
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
		self.presynaptic_cells = [] # list to store PresynapticCell objects within this FunctionalGroup

class PresynapticCell:
  
	def __init__(self, target_segment_indices: list = [], synapses: list = [], spike_train: np.ndarray = None, , FunctionalGroup: FunctionalGroup = None, name: str = None):
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
		self.synapses = synapses
		self.spike_train = spike_train
		self.FunctionalGroup = FunctionalGroup
		self.name = name