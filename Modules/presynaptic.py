import numpy as np
from sklearn.cluster import KMeans
from cell_model import CellModel
from neuron import h
import pandas as pd
import random

class FunctionalGroup:
    def __init__(self, seg_idxs: list, seg_coords: np.ndarray, name: str = None):
        self.seg_idxs = seg_idxs
        self.seg_coords = seg_coords
        self.name = name
        self.presynaptic_cells = []

class PresynapticCell:
    def __init__(self, seg_idxs: list, name: str = None, cluster_center: np.ndarray = None):
        self.seg_idxs = seg_idxs
        self.name = name
        self.cluster_center = cluster_center
        self.mean_fr = None
        self.spike_train = None
        self.vecstim = None
        self.spike_trains = []

    def set_spike_train(self, mean_fr, spike_train):
        self.mean_fr = mean_fr
        self.spike_train = spike_train
        self._set_vecstim(spike_train)

    def _set_vecstim(self, spike_train):
        vec = h.Vector(spike_train)
        stim = h.VecStim()
        stim.play(vec)
        self.vecstim = stim

class PCBuilder:
    @staticmethod
    def assign_presynaptic_cells(cell, n_func_gr, n_pc_per_fg, synapse_names, seg_names):
        segments, seg_data = cell.get_segments(seg_names)
        # Get segments based on synapse names
        seg_coords = [c.coords[["pc_0", "pc_1", "pc_2"]] for c in seg_data]
        seg_coords = pd.concat(seg_coords).to_numpy()

        # Ensure we do not request more functional groups than there are segments
        n_clusters = min(n_func_gr, len(seg_coords))
        labels, _ = PCBuilder._cluster_segments(seg_coords=seg_coords, n_clusters=n_clusters)

        functional_groups = []

        # Create functional groups based on clustering
        for fg_idx in np.unique(labels):
            fg_seg_idxs = [seg_ind for seg_ind in range(len(labels)) if labels[seg_ind] == fg_idx]
            functional_group = FunctionalGroup(
                seg_idxs=fg_seg_idxs,
                seg_coords=seg_coords[fg_seg_idxs],
                name="fg_" + str(fg_idx))
            functional_groups.append(functional_group)

        # Handle extra functional groups if needed
        if n_func_gr > len(functional_groups):
            seg_lengths = np.array([cell.get_segment_length(seg_idx, segments) for seg_idx in range(len(seg_coords))])
            probabilities = seg_lengths / seg_lengths.sum()

            extra_groups_needed = n_func_gr - len(functional_groups)
            for fg_idx in range(len(functional_groups), len(functional_groups) + extra_groups_needed):
                seg_idx = np.random.choice(range(len(seg_coords)), p=probabilities)
                functional_group = FunctionalGroup(
                    seg_idxs=[seg_idx],
                    seg_coords=seg_coords[seg_idx:seg_idx+1],
                    name="fg_" + str(fg_idx))
                functional_groups.append(functional_group)

        for functional_group in functional_groups:
            PCBuilder._build_presynaptic_cells_for_a_fg(cell, segments, functional_group, n_pc_per_fg)
            PCBuilder._map_synapses_to_pc(cell, segments, synapse_names, functional_group)

        return functional_groups

    @staticmethod
    def _cluster_segments(seg_coords: np.ndarray, n_clusters: int):
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        seg_id_to_cluster_index = km.fit_predict(seg_coords)
        cluster_centers = km.cluster_centers_
        return seg_id_to_cluster_index, cluster_centers
    
    @staticmethod
    def _build_presynaptic_cells_for_a_fg(cell, segments: list, fg: FunctionalGroup, n_pc_per_fg: int):
        n_segments = len(fg.seg_coords)
        n_clusters = min(n_pc_per_fg, n_segments)

        # Handle special case where there's only one segment
        if n_segments == 1:
            # Create the required number of presynaptic cells for the single segment
            for pc_idx in range(n_pc_per_fg):
                presynaptic_cell = PresynapticCell(
                    seg_idxs=fg.seg_idxs, 
                    name=fg.name + '_cell' + str(pc_idx), 
                    cluster_center=fg.seg_coords[0])
                fg.presynaptic_cells.append(presynaptic_cell)
            return

        # Cluster functional group segments into presynaptic cells
        labels, centers = PCBuilder._cluster_segments(
            seg_coords=fg.seg_coords, 
            n_clusters=n_clusters)

        # Create presynaptic cells for clustered segments
        for pc_idx in np.unique(labels):
            # Gather presynaptic cell segments
            pc_seg_idxs = [global_seg_index for seg_ind, global_seg_index in enumerate(fg.seg_idxs) if labels[seg_ind] == pc_idx]
            # Create PresynapticCell object
            presynaptic_cell = PresynapticCell(
                seg_idxs=pc_seg_idxs, 
                name=fg.name + '_cell' + str(pc_idx), 
                cluster_center=centers[pc_idx])
            fg.presynaptic_cells.append(presynaptic_cell)

        # Handle extra presynaptic cells
        if n_pc_per_fg > n_segments:
            seg_lengths = np.array([cell.get_segment_length(seg_idx, segments) for seg_idx in fg.seg_idxs])
            probabilities = seg_lengths / seg_lengths.sum()
            extra_cells_needed = n_pc_per_fg - n_segments

            for _ in range(extra_cells_needed):
                seg_idx = np.random.choice(fg.seg_idxs, p=probabilities)
                presynaptic_cell = PresynapticCell(
                    seg_idxs=[seg_idx], 
                    name=fg.name + '_cell' + str(len(fg.presynaptic_cells)), 
                    cluster_center=fg.seg_coords[fg.seg_idxs.index(seg_idx)])
                fg.presynaptic_cells.append(presynaptic_cell)

    @staticmethod
    def _map_synapses_to_pc(cell: CellModel, segments: list, synapse_names: list, functional_group: FunctionalGroup):
        for name in synapse_names:
            synapses = cell.get_synapses(synapse_names)
            for synapse in synapses:
                if synapse.pc is not None:
                    continue
                seg_index = segments.index(synapse.h_syn.get_segment())
                if seg_index not in functional_group.seg_idxs:
                    continue
                # Find the presynaptic cells that have this segment
                pcs_with_segment = [pc for pc in functional_group.presynaptic_cells if seg_index in pc.seg_idxs]
                if pcs_with_segment:
                    synapse.pc = random.choice(pcs_with_segment)
