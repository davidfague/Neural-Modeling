from neuron import h, nrn
import numpy as np
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator

def calc_dist(p1: list, p2: list) -> float:
	'''
	Calculate euclidean distance between two 3D coordinates.

	Parameters:
	----------
	p1, p2 : list
		List of x, y, z 3D coordinates.

	Returns:
	----------
	dist: float
		Euclidean distance.
 	'''
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def make_seg_sphere(center: list, segments: list, segment_centers: list, radius: float = 50):
	'''
	Create a list of segments within spherical radius of the center.

	Parameters:
	----------
	center: list
		3D coordinates of the center of the sphere (usually 3D coordinates of a segment).

	segments: list
		List of all segments to consider.

	segment_centers: list
		List of 3D coordinates corresponding to segments list/

	radius: float = 50
		Radius of sphere for which to return possible_segs.

	Returns:
	----------
	possible_segs: list
		Segments within radius of the center segment.
	'''
	possible_segs = []
	for i, seg in enumerate(segments):
		dist = calc_dist(center, segment_centers[i])
		if dist <= radius: possible_segs.append(seg)
	return possible_segs

def generate_excitatory_functional_groups(all_segments: list, all_segments_centers: list, all_len_per_segment: list,
										  number_of_groups: int, cells_per_group: int, synapses_per_cluster: int,
										  functional_group_span: float, cluster_span: float, 
										  gmax_dist: object, mean_fr_dist: object, 
										  spike_generator: SpikeGenerator, synapse_generator: SynapseGenerator,
										  t: np.ndarray, random_state: np.random.RandomState, neuron_r: h.Random,
										  record: bool = False, syn_params: dict = None, syn_mod: str = 'GABA_AB',
										  vector_length: int = None) -> list:

	functional_groups = []

	center_segs = random_state.choice(all_segments, p = all_len_per_segment / sum(all_len_per_segment), replace = False, size = number_of_groups)
	for group_id in range(number_of_groups):
		# Create a functional group
		center_seg = center_segs[group_id]
		func_grp = FunctionalGroup(center_seg = center_seg, segments = all_segments, segment_centers = all_segments_centers, radius = functional_group_span, 
				 				   name = 'exc_' + str(group_id))
		functional_groups.append(func_grp)

		# Generate trace common to all cells within each functional group
		fr_profile = spike_generator.get_firing_rate_profile(t, method = '1f_noise', random_state = random_state)

		# Iterate through cells which each have a cluster of synapses
		for _ in range(cells_per_group):
			cluster_seg = random_state.choice(func_grp.segments, p=func_grp.len_per_segment / sum(func_grp.len_per_segment))

			# Generate a cluster
			cluster = Cluster(center_seg = cluster_seg, segments = all_segments, segment_centers = all_segments_centers, radius = cluster_span)

			# Add synapses to to the cluster
			cluster.synapses = synapse_generator.add_synapses(segments = cluster.segments, 
							 								  probs = cluster.len_per_segment / np.sum(cluster.len_per_segment),
							 								  gmax = gmax_dist, syn_mod = syn_mod,
															  number_of_synapses = synapses_per_cluster, record = record, 
															  syn_params = syn_params, random_state = random_state,
															  neuron_r = neuron_r,
															  vector_length = vector_length)

			# Generate spikes common to each synapse within synaptic cluster
			mean_fr = spike_generator.get_mean_fr(mean_fr_dist)
			spikes = spike_generator.generate_spikes_from_profile(fr_profile, mean_fr, random_state)

			for synapse in cluster.synapses:
				cluster.netcons_list.append(spike_generator.set_spike_train(synapse, spikes))
				cluster.spike_trains.append(spikes)

			func_grp.synapses.append(cluster.synapses)
			func_grp.clusters.append(cluster)

	return functional_groups

def generate_inhibitory_functional_groups(cell: object, all_segments: list, all_segments_centers: list, all_len_per_segment: list,
										  number_of_groups: int, cells_per_group: int, synapses_per_cluster: int,
										  functional_group_span: float, cluster_span: float, 
										  gmax_dist: object, proximal_inh_dist: object, distal_inh_dist: object,
										  spike_generator: SpikeGenerator, synapse_generator: SynapseGenerator,
										  t: np.ndarray, f_group_name_prefix: str, 
										  spike_trains_to_delay: list, fr_time_shift: int, 
										  random_state: np.random.RandomState, neuron_r: h.Random,
										  record: bool = False, syn_params: dict = None, syn_mod: str = 'GABA_AB',
										  vector_length: int = None, method: str = "delay", tiesinga_params: tuple = None) -> list:
	functional_groups = []

	for group_id in range(number_of_groups):
		# Create a functional group
		center_seg = None
		func_grp = FunctionalGroup(center_seg = center_seg, segments = all_segments, segment_centers = all_segments_centers, 
				 				   radius = functional_group_span, name = f_group_name_prefix + str(group_id))
		functional_groups.append(func_grp)

		# Generate trace common to all cells within each functional group
		fr_profile = spike_generator.get_firing_rate_profile(t, method = method, 
							   								 spike_trains_to_delay = spike_trains_to_delay, 
															 fr_time_shift = fr_time_shift,
															 random_state = random_state,
															 tiesinga_params = tiesinga_params)

		# Iterate through cells which each have a cluster of synapses
		for _ in range(cells_per_group):
			cluster_seg = random_state.choice(all_segments, p = all_len_per_segment / sum(all_len_per_segment))

			# Generate a cluster
			cluster = Cluster(center_seg = cluster_seg, segments = all_segments, segment_centers = all_segments_centers, radius = cluster_span)

			if h.distance(cluster.center_seg, cell.soma[0](0.5)) <= 100:
				mean_fr_dist = proximal_inh_dist
			else:
				mean_fr_dist = distal_inh_dist

			# Add synapses to to the cluster
			cluster.synapses = synapse_generator.add_synapses(segments = cluster.segments, probs = cluster.len_per_segment,
							 								  gmax = gmax_dist, syn_mod = syn_mod,
															  number_of_synapses = synapses_per_cluster, record = record,
															  syn_params = syn_params, random_state = random_state, 
															  neuron_r = neuron_r,
															  vector_length = vector_length)

			# Generate spikes common to each synapse within synaptic cluster
			mean_fr = spike_generator.get_mean_fr(mean_fr_dist)
			spikes = spike_generator.generate_spikes_from_profile(fr_profile, mean_fr, random_state)

			for synapse in cluster.synapses:
				cluster.netcons_list.append(spike_generator.set_spike_train(synapse, spikes))
				cluster.spike_trains.append(spikes)

			func_grp.synapses.append(cluster.synapses)
			func_grp.clusters.append(cluster)

	return functional_groups

class FunctionalGroup:
  
	def __init__(self, center_seg: nrn.Segment = None, segments: list = [], segment_centers: list = [], radius: float = 100, name: str = None):
		'''
		Parameters:
		----------
		center_seg: nrn.Segment
			Segment at the center of this FunctionalGroup.

		segments: list
			List of all segments to consider.

		segment_centers: list
			List of 3D coordinates according to segments list.

		radius: float = 100
			radius of sphere for which to return possible_segs

		name: str = None
			Name of the functional group.
		'''
		self.center_seg = center_seg
		self.segments = segments # list of segments within this FunctionalGroup
		self.len_per_segment = [] # list of lengths of segments within this FunctionalGroup
		self.synapses = [] # list of synapses in this cluster
		self.clusters = []
		self.spike_trains = []
		self.netcons_list = []
		self.name = name

		if center_seg is not None:
			# Get 3D coordinates of center_seg
			center = segment_centers[segments.index(center_seg)]
			# Get segments within this cluster
			self.segments = make_seg_sphere(center = center, segments = segments, segment_centers = segment_centers, radius = radius)
		if len(self.segments) != len(self.len_per_segment):
			# Get segment lengths
			self.len_per_segment = []
			for seg in self.segments:
				self.len_per_segment.append(seg.sec.L / seg.sec.nseg)
			self.len_per_segment = np.array(self.len_per_segment)

class Cluster:

	def __init__(self, center_seg: nrn.Segment = None, segments: list = [], segment_centers: list = [], radius: float = 10):
		'''
		Parameters:
		----------
		center_seg: nrn.Segment
			Segment at the center of this Cluster

		segments: list
			List of all segments to consider

		segment_centers: list
			List of 3D coordinates according to segments list.

		radius: float = 10
			Radius of the sphere for which to return possible_segs.
		'''
		self.center_seg = center_seg
		self.segments = [] # list of segments within this cluster
		self.len_per_segment = [] # list of lengths of segments within this cluster
		self.synapses = [] # list of synapses in this cluster
		self.clusters = []
		self.spike_trains = []
		self.netcons_list = []
	
		# Get 3D coordinates of center_seg
		center = segment_centers[segments.index(center_seg)]
		# Get segments within this cluster
		self.segments = make_seg_sphere(center = center, segments = segments, segment_centers = segment_centers, radius = radius)
		# Get segment lengths
		for seg in self.segments:
			self.len_per_segment.append(seg.sec.L / seg.sec.nseg)
		self.len_per_segment = np.array(self.len_per_segment)
