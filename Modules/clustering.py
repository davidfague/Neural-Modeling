import networkx as nx
from sklearn.cluster import KMeans
from neuron_reduce.reducing_methods import (_get_subtree_biophysical_properties, find_space_const_in_cm)
from Modules.spike_generator import SpikeGenerator
from Modules.synapse_generator import SynapseGenerator
from neuron import h,nrn

# Create a graph from the segments
def create_graph(seg_infos):
    G = nx.Graph()
    for seg_info in seg_infos:
        G.add_node(seg_info['seg_index_global'], attr_dict=seg_info)
        for adj_segment in seg_info['adjacent_segments']:
            adjacent_seg_info = next((info for info in cell.seg_info if info['seg'] == adj_segment), None)
            if adjacent_seg_info:
                G.add_edge(seg_info['seg_index_global'], adjacent_seg_info['seg_index_global'])
            else:
              print(adjacent_seg_info)
    return G

def get_elec_length(seg, frequency):
    # compute the electrotonic length of a segment
    cm, rm, ra, e_pas, q = _get_subtree_biophysical_properties(h.SectionRef(sec=seg['sec']), frequency)
    cable_space_const_in_cm = find_space_const_in_cm(seg['seg_diam']/10000, rm, ra)
    return seg['seg_L']/(cable_space_const_in_cm*10000)

def calc_elec_distance(G, seg1, seg2, frequency):
    # Check if a path exists between seg1 and seg2
    if not nx.has_path(G, seg1['seg_index_global'], seg2['seg_index_global']):
        raise ValueError(f"No path between seg1 (ID {seg1['seg_index_global']}) and seg2 (ID {seg2['seg_index_global']})")

    # Calculate the initial distance (half of seg1's electrotonic length)
    start_dist = get_elec_length(seg1, frequency) / 2

    # Use NetworkX to get the shortest path between seg1 and seg2
    path = nx.shortest_path(G, seg1['seg_index_global'], seg2['seg_index_global'])

    # Accumulate the electrotonic lengths along the path
    for seg_index in path[1:-1]:  # Exclude the start and end segments
        seg = next((info for info in cell.seg_info if info['seg_index_global'] == seg_index), None)
        start_dist += get_elec_length(seg, frequency)

    # Add half of the electrotonic length of seg2 to the final distance
    end_dist = get_elec_length(seg2, frequency) / 2
    final_dist = start_dist + end_dist

    return final_dist

def cluster_synapses(synapses, n_clusters, frequency=0):
    # Use seg_info dictionaries instead of seg values
    segment_infos = list([seg_info for synapse in synapses for seg_info in cell.seg_info if seg_info['seg'] == synapse.segment])
    # print(segment_infos[0])

    # Adjust the rest of your function to work with seg_info dictionaries
    # G = create_graph([seg_info for seg_info in segment_infos])
    G = create_graph(segment_infos)
    print(G)
    print(nx.is_connected(G))

    # Create a list of distances
    distances = [[calc_elec_distance(G, seg_info1, seg_info2, frequency) for seg_info2 in segment_infos] for seg_info1 in segment_infos]
    # Perform the clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distances)
    # Create a dictionary to map segments to clusters
    segment_cluster_map = {seg_info['seg']: cluster for seg_info, cluster in zip(segment_infos, kmeans.labels_)}
    # Map each synapse to its segment's cluster
    synapse_cluster_map = {synapse: segment_cluster_map[synapse.segment] for synapse in synapses}
    # Create a map of segment to seg_info to be used later
    seg_info_map = {seg_info['seg']: seg_info for seg_info in segment_infos}

    # Return both the synapse_cluster_map and the seg_info_map
    return synapse_cluster_map, seg_info_map
