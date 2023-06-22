from neuron import h
import numpy as np
from Modules.cell_model import CellModel
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator

# TODO: update to use spherical radius around center_seg instead of path length.

def generate_functional_groups(cell: CellModel, all_segments: list, all_len_per_segment: list,
			       number_of_groups: int, cells_per_group: int, synapses_per_cluster: int,
			       functional_group_span: float, cluster_span: float, 
			       gmax_dist, mean_fr_dist, 
			       spike_generator: SpikeGenerator, synapse_generator: SynapseGenerator,
				t, record: bool = False):
	functional_groups = []

	rnd = np.random.RandomState(10)

	for group_id in range(number_of_groups):
		# Create a functional group
		center_seg = rnd.choice(all_segments, p = all_len_per_segment / sum(all_len_per_segment))
		func_grp = FunctionalGroup(cell = cell, center_seg = center_seg, span = functional_group_span, 
			     				   name = 'exc_' + str(group_id))
		functional_groups.append(func_grp)

		# Generate trace common to all cells within each functional group
		fr_profile = spike_generator.get_firing_rate_profile(t, method = '1f_noise')

		# Iterate through cells which each have a cluster of synapses
		for _ in range(cells_per_group):
			cluster_seg = rnd.choice(func_grp.segments, p=func_grp.len_per_segment / sum(func_grp.len_per_segment))

			# Generate a cluster
			cluster = Cluster(cell, cluster_seg, cluster_span)

			# Add synapses to to the cluster
			cluster.synapses = synapse_generator.add_synapses(segments = cluster.segments, probs = cluster.len_per_segment,
						     								  gmax = gmax_dist, syn_mod = 'AMPA_NMDA',
															  number_of_synapses = synapses_per_cluster, record = record)

			# Generate spikes common to each synapse within synaptic cluster
			mean_fr = spike_generator.get_mean_fr(mean_fr_dist)
			spikes = spike_generator.generate_spikes_from_profile(fr_profile,mean_fr)

			for synapse in cluster.synapses:
				cluster.netcons_list.append(spike_generator.set_spike_train(synapse, spikes))
				cluster.spike_trains.append(spikes)

			func_grp.synapses.append(cluster.synapses)
			func_grp.clusters.append(cluster)

	return functional_groups

class FunctionalGroup:
  
	def __init__(self, cell: object, center_seg: nrn.Segment, span: float, name: str):
		'''
		Parameters:
		----------
		cell: HocObject
			Cell to process.

		center_seg: nrn.Segment
			Segment at the center of the functional group

		span: float
			Length of the functional group.

		name: str
			Name of the functional group.
		'''
		self.name = name
		self.center_seg = center_seg
		self.segments = []
		self.len_per_segment = []
		self.synapses = []
		self.clusters = []

		if center_seg is not None:
			for sec in cell.all:
				for seg in sec:
					if h.distance(center_seg, seg) <= (span / 2):
						self.segments.append(seg)
						self.len_per_segment.append(seg.sec.L / seg.sec.nseg)
			
		self.len_per_segment = np.array(self.len_per_segment)

class Cluster:
  
	def __init__(self, cell, center_seg: nrn.Segment, span):
		'''
		Parameters:
		----------
		cell: HocObject
			Cell to process.

		center_seg: nrn.Segment
			Segment at the center of the Cluster.

		span: float
			Length of the functional group.

		functional_group: str
			Name of the functional group.
		'''
		self.center_seg = center_seg
		self.segments = []
		self.len_per_segment = []
		self.synapses = []
		self.clusters = []
		self.spike_trains = []
		self.netcons_list = []

		for sec in cell.all:
			for seg in sec:
				if h.distance(center_seg, seg) <= (span / 2):
					self.segments.append(seg)
					self.len_per_segment.append(seg.sec.L / seg.sec.nseg)

		self.len_per_segment = np.array(self.len_per_segment)
