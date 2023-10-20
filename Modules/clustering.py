import numpy as np
from sklearn.cluster import KMeans
from Modules.spike_generator import SpikeGenerator
from Modules.cell_model import CellModel


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


def create_functional_groups_of_presynaptic_cells(segments_coordinates: np.ndarray, 
												 n_functional_groups: int, 
												 n_presynaptic_cells_per_functional_group: int, 
												 name_prefix: str, cell: CellModel, synapses: list,
												 **kwargs):
												 
	# Cluster cell segments into functional groups
	seg_id_to_functional_group_index, functional_group_cluster_centers = cluster_segments(segments_coordinates=segments_coordinates, n_clusters=n_functional_groups)

	# Assemble functional groups and their presynaptic cells
	functional_groups = create_functional_groups(seg_id_to_functional_group_index=seg_id_to_functional_group_index, cell=cell, name_prefix=name_prefix, segments_coordinates=segments_coordinates, n_presynaptic_cells_per_functional_group=n_presynaptic_cells_per_functional_group, functional_group_cluster_centers=functional_group_cluster_centers)

	# Map synapses to PresynapticCells
	map_synapses_to_PresynapticCells(synapses=synapses, functional_groups=functional_groups, cell=cell)

	# Calculate spike train for each cluster (implement this in SpikeGenerator)
	generate_spike_train_for_functional_groups(functional_groups = functional_groups, **kwargs)

	return functional_groups

def get_euclidean_distance(point1, point2):
	return np.linalg.norm(point1 - point2)

def cluster_segments(segments_coordinates: np.ndarray, n_clusters: int):
	km = KMeans(n_clusters = n_clusters, n_init = "auto")
	seg_id_to_cluster_index = km.fit_predict(segments_coordinates)
	cluster_centers = km.cluster_centers_
	return seg_id_to_cluster_index, cluster_centers
  
def create_functional_groups(seg_id_to_functional_group_index: list, segments_coordinates: np.ndarray, n_presynaptic_cells_per_functional_group: int, cell: CellModel, name_prefix: str, functional_group_cluster_centers: np.ndarray):
	# Create functional groups
	functional_groups = []
	for functional_group_index in np.unique(seg_id_to_functional_group_index):
		functional_group_cluster_center=functional_group_cluster_centers[functional_group_index]

		# Gather functional group segments
		functional_group_target_segments_indices = []
		for seg_ind, seg in enumerate(cell.seg_info):
			if seg_id_to_functional_group_index[seg_ind] == functional_group_index: # Segment is in functional group targets
				functional_group_target_segments_indices.append(seg_ind)

		# Create FunctionalGroup object
		functional_group_name = name_prefix +str(functional_group_index)
		functional_group_target_segments_coordinates = segments_coordinates[functional_group_target_segments_indices]
		functional_group = FunctionalGroup(target_segment_indices = functional_group_target_segments_indices, name = functional_group_name, target_segments_coordinates = functional_group_target_segments_coordinates, cluster_center = functional_group_cluster_center)

		functional_groups.append(functional_group)
		create_presynaptic_cells(functional_group_target_segments_coordinates=functional_group_target_segments_coordinates, n_presynaptic_cells_per_functional_group=n_presynaptic_cells_per_functional_group, functional_group_name=functional_group_name, functional_group=functional_group)

	return functional_groups
  
def create_presynaptic_cells(functional_group_target_segments_coordinates: np.ndarray, n_presynaptic_cells_per_functional_group: int, functional_group_name: str, functional_group: FunctionalGroup):
	# Cluster functional group segments into presynaptic cells
	seg_ind_to_presynaptic_cell_index, presynaptic_cluster_centers = cluster_segments(segments_coordinates = functional_group_target_segments_coordinates, n_clusters = n_presynaptic_cells_per_functional_group)
	
	# Create presynaptic cells
	for presynaptic_cell_index in np.unique(seg_ind_to_presynaptic_cell_index):
		presynaptic_cell_cluster_center = presynaptic_cluster_centers[presynaptic_cell_index]
	
		# Gather presynaptic cell segments
		presynaptic_cell_target_segments_indices = []
		for seg_ind, global_seg_index in enumerate(functional_group.target_segment_indices):
			if seg_ind_to_presynaptic_cell_index[seg_ind] == presynaptic_cell_index: # segment is in presynaptic cell targets
				presynaptic_cell_target_segments_indices.append(global_seg_index)
	
		# Create PresynapticCell object
		presynaptic_cell = PresynapticCell(target_segment_indices = presynaptic_cell_target_segments_indices, name = functional_group_name + '_cell' + str(presynaptic_cell_index), cluster_center=presynaptic_cell_cluster_center)
		functional_group.presynaptic_cells.append(presynaptic_cell)

	return presynaptic_cell
  
def map_synapses_to_PresynapticCells(synapses: list, cell: CellModel, functional_groups: list):
	all_mapped_functional_group_seg_indices = []
	all_mapped_presynaptic_cell_seg_indices = []

	for functional_group in functional_groups:
		for seg_index in functional_group.target_segment_indices:
			all_mapped_functional_group_seg_indices.append(seg_index)
	
		for presynaptic_cell in functional_group.presynaptic_cells:
			for seg_index in presynaptic_cell.target_segment_indices:
				all_mapped_presynaptic_cell_seg_indices.append(seg_index)

	all_mapped_functional_group_seg_indices = np.sort(all_mapped_functional_group_seg_indices)
	all_mapped_presynaptic_cell_seg_indices = np.sort(all_mapped_presynaptic_cell_seg_indices)

	# Will need to separate synapses list because we do not want to give exc and inh synapses the same presynaptic cell
	for synapse in synapses:
		synapse_seg_index = cell.segments.index(synapse.get_segment())

		# Match the synapse's segment's index to a presynaptic cell
		synapse_already_mapped = False
		for functional_group in functional_groups:
			if synapse_seg_index in functional_group.target_segment_indices:
				for presynaptic_cell in functional_group.presynaptic_cells:
					if synapse_seg_index in presynaptic_cell.target_segment_indices:
						if not synapse_already_mapped:
							# Add the synapse to the presynaptic cell
							presynaptic_cell.synapses.append(synapse)
							synapse_already_mapped=True
						else:
							raise(ValueError("Synapse being mapped to multiple presynaptic cells."))
								
		if not synapse_already_mapped:
			raise(RuntimeError("Not able to map synapse to presynaptic cell"))
	  

def generate_spike_train_for_functional_groups(
	functional_groups: list,
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
	proximal_vs_distal_boundary: float = 100):
	
	for functional_group in functional_groups:

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

			functional_group.firing_rate_profile = spike_generator.get_firing_rate_profile(
				t = t, 
				method = method, 
				random_state = random_state,
				mean_fr = mean_fr,
				rhythmicity = rhythmicity,
				rhythmic_mod = rhythmic_mod, 
				rhythmic_f = rhythmic_f,
				spike_trains_to_delay = spike_trains_to_delay,
				fr_time_shift = fr_time_shift, 
				spike_train_dt = spike_train_dt)
	
			spikes = spike_generator.generate_spikes_from_profile(functional_group.firing_rate_profile, random_state)

			presynaptic_cell.spike_train.append(spikes)
			for synapse in presynaptic_cell.synapses:
				_ = spike_generator.set_spike_train(synapse, spikes)