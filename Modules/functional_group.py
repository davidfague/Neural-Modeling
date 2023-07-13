from typing import List
import numpy as np
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator
from neuron import nrn, h

def calc_dist(p1: List[float], p2: List[float]) -> float:
    '''Calculate euclidean distance between two 3D coordinates.'''
    return np.linalg.norm(np.subtract(p1, p2))

def get_segments_within_radius(center: List[float], segments: list, segment_centers: List[float], radius: float = 50):
    '''Return a list of segments within a specified radius of the center.'''
    return [seg for i, seg in enumerate(segments) if calc_dist(center, segment_centers[i]) <= radius]

def get_segment_lengths(segments: list):
    '''Return a list of segment lengths.'''
    return np.array([seg.sec.L / seg.sec.nseg for seg in segments])

class FunctionalGroup:
    '''A functional group of segments within a neuron model.'''
    def __init__(self, center_seg: nrn.Segment, segments: list, segment_centers: List[float], radius: float = 100, name: str = None):
        self.center_seg = center_seg
        self.name = name
        self.synapses = []
        self.clusters = []
        self.spike_trains = []
        self.netcons_list = []

        if center_seg is not None:
            center = segment_centers[segments.index(center_seg)]
            self.segments = get_segments_within_radius(center, segments, segment_centers, radius)
            self.len_per_segment = get_segment_lengths(self.segments)

class Cluster(FunctionalGroup):
    '''A cluster of segments within a functional group.'''
    def __init__(self, center_seg: nrn.Segment, segments: list, segment_centers: List[float], radius: float = 10):
        super().__init__(center_seg, segments, segment_centers, radius)
        self.synapses = []

def generate_excitatory_functional_groups(all_segments: list, all_segments_centers: list, all_len_per_segment: list,
                                          number_of_groups: int, cells_per_group: int, synapses_per_cluster: int,
                                          functional_group_span: float, cluster_span: float, 
                                          gmax_dist: object, mean_fr_dist: object, 
                                          spike_generator: SpikeGenerator, synapse_generator: SynapseGenerator,
                                          t: np.ndarray, random_state: np.random.RandomState, neuron_r: h.Random,
                                          record: bool = False, syn_params: dict = None, syn_mod: str = 'AMPA_NMDA') -> list:
    center_segs = random_state.choice(all_segments, p=all_len_per_segment / sum(all_len_per_segment), replace=False, size=number_of_groups)
    functional_groups = [FunctionalGroup(center_segs[i], all_segments, all_segments_centers, functional_group_span, 'exc_' + str(i)) for i in range(number_of_groups)]

    # Continue with the rest of the function body
    for func_grp in functional_groups:
        for _ in range(cells_per_group):
            cluster_seg = random_state.choice(func_grp.segments, p=func_grp.len_per_segment / sum(func_grp.len_per_segment))

            # Generate a cluster
            cluster = Cluster(center_seg=cluster_seg, segments=all_segments, segment_centers=all_segments_centers, radius=cluster_span)

            # Add synapses to the cluster
            cluster.synapses = synapse_generator.add_synapses(segments=cluster.segments, 
                                                              probs=cluster.len_per_segment / np.sum(cluster.len_per_segment),
                                                              gmax=gmax_dist, syn_mod=syn_mod,
                                                              number_of_synapses=synapses_per_cluster, record=record, 
                                                              syn_params=syn_params, random_state=random_state,
                                                              neuron_r=neuron_r)

            # Generate spikes common to each synapse within synaptic cluster
            mean_fr = spike_generator.get_mean_fr(mean_fr_dist)
            spikes = spike_generator.generate_spikes_from_profile(t, mean_fr, random_state)

            for synapse in cluster.synapses:
                cluster.netcons_list.append(spike_generator.set_spike_train(synapse, spikes))
                cluster.spike_trains.append(spikes)

            func_grp.synapses.extend(cluster.synapses)
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
                                          record: bool = False, syn_params: dict = None, syn_mod: str = 'GABA_AB') -> list:
    functional_groups = [FunctionalGroup(None, all_segments, all_segments_centers, functional_group_span, f_group_name_prefix + str(i)) for i in range(number_of_groups)]

    # Continue with the rest of the function body
    for func_grp in functional_groups:
        for _ in range(cells_per_group):
            cluster_seg = random_state.choice(all_segments, p=all_len_per_segment / sum(all_len_per_segment))

            # Generate a cluster
            cluster = Cluster(center_seg=cluster_seg, segments=all_segments, segment_centers=all_segments_centers, radius=cluster_span)

            if h.distance(cluster.center_seg, cell.soma[0](0.5)) <= 100:
                mean_fr_dist = proximal_inh_dist
            else:
                mean_fr_dist = distal_inh_dist

            # Add synapses to to the cluster
            cluster.synapses = synapse_generator.add_synapses(segments=cluster.segments, 
                                                              probs=cluster.len_per_segment / np.sum(cluster.len_per_segment),
                                                              gmax=gmax_dist, syn_mod=syn_mod,
                                                              number_of_synapses=synapses_per_cluster, record=record, 
                                                              syn_params=syn_params, random_state=random_state, 
                                                              neuron_r=neuron_r)

            # Generate spikes common to each synapse within synaptic cluster
            mean_fr = spike_generator.get_mean_fr(mean_fr_dist)
            spikes = spike_generator.generate_spikes_from_profile(t, mean_fr, random_state)

            for synapse in cluster.synapses:
                cluster.netcons_list.append(spike_generator.set_spike_train(synapse, spikes))
                cluster.spike_trains.append(spikes)

            func_grp.synapses.extend(cluster.synapses)
            func_grp.clusters.append(cluster)

    return functional_groups
