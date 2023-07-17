import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from neuron_reduce.reducing_methods import (_get_subtree_biophysical_properties, find_space_const_in_cm)
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator
from neuron import h,nrn

def create_graph(seg_infos, seg_info_list):
    G = nx.Graph()
    for seg_info in seg_infos:
        G.add_node(seg_info['seg_index_global'], attr_dict=seg_info)
        for adj_segment in seg_info['adjacent_segments']:
            adjacent_seg_info = find_segment_info(adj_segment, seg_info_list)
            if adjacent_seg_info:
                G.add_edge(seg_info['seg_index_global'], adjacent_seg_info['seg_index_global'])
    return G

def find_segment_info(segment, seg_info_list):
    return next((info for info in seg_info_list if info['seg'] == segment), None)

def get_elec_length(seg, frequency):
    cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=seg['sec']), frequency)
    cable_space_const_in_cm = find_space_const_in_cm(seg['seg_diam']/10000, rm, ra)
    return seg['seg_L']/(cable_space_const_in_cm*10000)

def get_euclidean_distance(seg1, seg2):
    point1 = np.array([seg1['p0.5_x3d'], seg1['p0.5_y3d'], seg1['p0.5_z3d']])
    point2 = np.array([seg2['p0.5_x3d'], seg2['p0.5_y3d'], seg2['p0.5_z3d']])
    return np.linalg.norm(point1 - point2)

def compute_electrotonic_distance(G, seg1, seg2, frequency):
    # Calculate the initial distance (half of seg1's electrotonic length)
    start_dist = get_elec_length(seg1, frequency) / 2
    # Use NetworkX to get the shortest path between seg1 and seg2
    path = nx.shortest_path(G, seg1['seg_index_global'], seg2['seg_index_global'])
    # Accumulate the electrotonic lengths along the path
    for seg_index in path[1:-1]:  # Exclude the start and end segments
        seg = find_segment_info(seg_index)
        start_dist += get_elec_length(seg, frequency)
    # Add half of the electrotonic length of seg2 to the final distance
    end_dist = get_elec_length(seg2, frequency) / 2
    return start_dist + end_dist

def calc_distance(G, seg1, seg2, frequency, use_euclidean):
    # # consider removing check for faster computation
    # if not nx.has_path(G, seg1['seg_index_global'], seg2['seg_index_global']): # check
    #     raise ValueError(f"No path between seg1 (ID {seg1['seg_index_global']}) and seg2 (ID {seg2['seg_index_global']})")
    return get_euclidean_distance(seg1, seg2) if use_euclidean else compute_electrotonic_distance(G, seg1, seg2, frequency)

def cluster_synapses(synapses, n_clusters, seg_info_list, frequency=0, use_euclidean=False):
    segment_infos = [find_segment_info(synapse.segment, seg_info_list) for synapse in synapses] # list for synapses
    G = create_graph(segment_infos, seg_info_list)
    distances = [[calc_distance(G, seg_info1, seg_info2, frequency, use_euclidean) for seg_info2 in segment_infos] for seg_info1 in segment_infos]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distances)
    segment_cluster_map = {seg_info['seg']: cluster for seg_info, cluster in zip(segment_infos, kmeans.labels_)}
    synapse_cluster_map = {synapse: segment_cluster_map[synapse.segment] for synapse in synapses}
    return synapse_cluster_map, segment_cluster_map
