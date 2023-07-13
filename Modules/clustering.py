import networkx as nx
from sklearn.cluster import KMeans
from neuron_reduce.reducing_methods import (_get_subtree_biophysical_properties, find_space_const_in_cm)
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator
from neuron import h,nrn

# Create a graph from the segments
def create_graph(segments):
    G = nx.Graph()
    for segment in segments:
        G.add_edge(*segment)
    return G

def calc_elec_distance(seg1, seg2, frequency):
    # Initialize a queue for BFS with the start segment
    # Add half of the electrotonic length of seg1 to the starting distance
    cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=seg1), frequency)
    cable_space_const_in_cm = find_space_const_in_cm(seg1.diam/10000, rm, ra)
    cable_elec_L = seg1.L/(cable_space_const_in_cm*10000)
    start_dist = cable_elec_L / 2

    queue = [(seg1, start_dist)]  
    visited = {seg1}

    while queue:
        current_seg, dist = queue.pop(0)
        if current_seg == seg2:
            # Subtract half of the electrotonic length of seg2 from the final distance
            cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=seg2), frequency)
            cable_space_const_in_cm = find_space_const_in_cm(seg2.diam/10000, rm, ra)
            cable_elec_L = seg2.L/(cable_space_const_in_cm*10000)
            end_dist = cable_elec_L / 2

            return dist - end_dist
        for adj_seg in current_seg.adj_segs:
            if adj_seg not in visited:
                # get biophysical properties
                cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=adj_seg), frequency)
                # calculate the space constant
                cable_space_const_in_cm = find_space_const_in_cm(adj_seg(0.5).diam/10000, rm, ra)
                # calculate the electrotonic length of the segment
                cable_elec_L = adj_seg.L/(cable_space_const_in_cm*10000)
                # add the segment to the queue with the updated distance
                queue.append((adj_seg, dist+cable_elec_L))
                visited.add(adj_seg)
    raise ValueError("No path between seg1 and seg2")

# Group synapses into clusters based on their segment's electrotonic distances
def cluster_synapses(synapses, n_clusters):
    segments = list(set([synapse.segment for synapse in synapses]))
    G = create_graph(segments)
    # Create a list of distances
    distances = [[calc_elec_distance(segment1, segment2, frequency) for segment2 in segments] for segment1 in segments]
    # Perform the clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distances)
    # Create a dictionary to map segments to clusters
    segment_cluster_map = {segment: cluster for segment, cluster in zip(segments, kmeans.labels_)}
    # Map each synapse to its segment's cluster
    synapse_cluster_map = {synapse: segment_cluster_map[synapse.segment] for synapse in synapses}
    return synapse_cluster_map

def add_synapses(cell):
    synapse_generator.add_synapses(segments=all_segments, 
                                                              probs=all_len_per_segment / np.sum(all_len_per_segment),
                                                              gmax=gmax_dist, syn_mod=syn_mod,
                                                              number_of_synapses=synapses_per_cluster, record=record, 
                                                              syn_params=syn_params, random_state=random_state,
                                                              neuron_r=neuron_r)
