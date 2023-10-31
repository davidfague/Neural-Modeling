import numpy as np
from neuron import h
from neuron_reduce import subtree_reductor

from Modules.cable_expander_func import cable_expander, get_syn_to_netcons
from Modules.synapse import Synapse
from Modules.cell_model import CellModel
from Modules.logger import Logger

import warnings


class Reductor():
    def __init__(self, logger: Logger):
        self.logger = logger

    def reduce_cell(self, complex_cell: object, reduce_cell: bool = False, optimize_nseg: bool = False, 
                    py_synapses_list: list = None, netcons_list: list = None, spike_trains: list = None, 
                    spike_threshold: int = 10, random_state: np.random.RandomState = None, 
                    var_names: list = None, reduction_frequency: float = 0, expand_cable: bool = False, 
                    choose_branches: list = None, seg_to_record: str = 'soma'):

        # Map Python Synapse objects to NEURON Synapse objects
        py_to_hoc_synapses = {syn: syn.synapse_hoc_obj for syn in py_synapses_list}

        # If no cell reduction is required
        if not reduce_cell:
            if optimize_nseg: 
                self.update_model_nseg_using_lambda(complex_cell)
            cell = self._create_cell_model(complex_cell, py_synapses_list, netcons_list, spike_trains, spike_threshold, 
                                           random_state, var_names, seg_to_record)
            return cell
        
        # Cell reduction
        reduced_cell, hoc_synapses_list, netcons_list, txt_nr = subtree_reductor(
            complex_cell, list(py_to_hoc_synapses.values()), netcons_list, reduction_frequency, return_seg_to_seg=True)
        
        # Clear old Synapse objects that didn't survive
        surviving_hoc_synapses = set(hoc_synapses_list)
        py_synapses_list[:] = [syn for syn, hoc_syn in py_to_hoc_synapses.items() if hoc_syn in surviving_hoc_synapses]
        
        # expand cable if requested
        if expand_cable:
            return self._handle_cable_expansion(reduced_cell, py_synapses_list, hoc_synapses_list, netcons_list, reduction_frequency, 
                                                random_state, spike_trains, spike_threshold, var_names, seg_to_record, choose_branches)
        
        # only for NR cell
        #Make sure section attributes are correct. (can update cell_model class to include this list formation)
        reduced_cell.all = []
        for model_part in ["soma", "apic", "dend", "axon"]:
            setattr(reduced_cell, model_part, CellModel.convert_section_list(reduced_cell, getattr(reduced_cell, model_part)))
        for sec in reduced_cell.soma + reduced_cell.apic + reduced_cell.dend + reduced_cell.axon:
            reduced_cell.all.append(sec)

        
        # Post-process cell
        return self._post_process_reduced_cell(reduced_cell, py_synapses_list, netcons_list, spike_trains, spike_threshold, random_state, var_names, seg_to_record)
  

    def _create_cell_model(self, hoc_model, py_synapses_list, netcons_list, spike_trains, spike_threshold, random_state, 
                           var_names, seg_to_record):
        """
        Helper method to create a CellModel instance.
        """
        cell = CellModel(
            hoc_model=hoc_model,
            synapses=py_synapses_list,
            netcons=netcons_list,
            spike_trains=spike_trains,
            spike_threshold=spike_threshold,
            random_state=random_state,
            var_names=var_names,
            seg_to_record=seg_to_record
        )
        self.logger.log(f"Reductor reported {len(cell.tufts)} terminal tuft branches in the model.")
        return cell
    
    def _handle_cable_expansion(self, reduced_cell, py_synapses_list, hoc_synapses_list, netcons_list, reduction_frequency, random_state, 
                                spike_trains, spike_threshold, var_names, seg_to_record, choose_branches):
        sections_to_expand = [reduced_cell.hoc_model.apic[0]]
        furcations_x = [0.289004]
        nbranches = [choose_branches]

        expanded_cell, hoc_synapses_list, netcons_list, _ = cable_expander(
            reduced_cell, 
            sections_to_expand, 
            furcations_x, 
            nbranches, 
            hoc_synapses_list, 
            netcons_list, 
            reduction_frequency, 
            return_seg_to_seg=True, 
            random_state=random_state
        )
        
        # Clear old Synapse objects that didn't survive the merging during expansion
        surviving_hoc_synapses = set(hoc_synapses_list)
        py_synapses_list[:] = [syn for syn, hoc_syn in py_to_hoc_synapses.items() if hoc_syn in surviving_hoc_synapses]
        
        # Create new python synapse for new hoc synapses that were created during expansion
        existing_hoc_synapses = set(hoc_synapse for py_syn, hoc_synapse in py_to_hoc_synapses.items()) # from overarching dictionary
        for hoc_syn in hoc_synapses_list: # from cable_axpander
            if hoc_syn not in existing_hoc_synapses:
                new_syn = Synapse(syn_obj=hoc_syn, record=True)  
                py_synapses_list.append(new_syn)
                py_to_hoc_synapses[new_syn] = hoc_syn
                

        return self._post_process_reduced_cell(expanded_cell, py_synapses_list, netcons_list, spike_trains, 
                                               spike_threshold, random_state, var_names, seg_to_record)

    def _post_process_reduced_cell(self, reduced_cell, py_synapses_list, netcons_list, spike_trains, spike_threshold, 
                                   random_state, var_names, seg_to_record):
        """
        Helper method to post-process a reduced cell.
        """
        # Map hoc synapses to netcons
        #hoc_syn_to_netcon = get_syn_to_netcons(netcons_list)

        #* OLD IMPLEMENTATION now python synapses are handled as hoc synapses are created or deleted during reduction
        ## Convert hoc synapse objects back to python Synapse class and append netcons
        #py_synapses_list = [Synapse(syn_obj=hoc_syn, record=True, ncs=hoc_syn_to_netcon[hoc_syn]) for hoc_syn in hoc_synapses_list]    
        
        # CURRENTLY FORGETTING ABOUT UPDATING Synapse.ncs        
        
        # * Should already be taken care of for reduced cells * # will need to pass optimize segments
        # Optimize segments if requested
        #if optimize_nseg: 
        #    self.update_model_nseg_using_lambda(reduced_cell)

        # Create a reduced cell model and return it.
        cell = CellModel(
            hoc_model=reduced_cell, 
            synapses=py_synapses_list, 
            netcons=netcons_list,
            spike_trains=spike_trains, 
            spike_threshold=spike_threshold, 
            random_state=random_state,
            var_names=var_names, 
            seg_to_record=seg_to_record
        )
        self.logger.log(f"Reductor reported {len(cell.tufts)} terminal tuft branches in the model.")
        return cell
        
    def find_space_const_in_cm(self, diameter: float, rm: float, ra: float) -> float:
        """Returns space constant (lambda) in cm, according to: space_const = sqrt(rm/(ri+r0))."""
        rm /= (np.pi * diameter)
        ri = 4 * ra / (np.pi * (diameter**2))
        return np.sqrt(rm / ri)
  
    def calculate_nseg_from_lambda(self, section: h.Section, segs_per_lambda: int) -> int:
        rm = 1.0 / section.g_pas  # (ohm * cm^2)
        ra = section.Ra  # (ohm * cm)
        diam_in_cm = section.L / 10000
        space_const_in_micron = 10000 * self.find_space_const_in_cm(diam_in_cm, rm, ra)
        return int((section.L / space_const_in_micron) * segs_per_lambda / 2) * 2 + 1
    
    def update_model_nseg_using_lambda(self, cell: object, segs_per_lambda: int = 10):
        """Optimizes number of segments using length constant."""
        initial_nseg, new_nseg = sum(sec.nseg for sec in cell.all), 0

        for sec in cell.all:
            sec.nseg = self.calculate_nseg_from_lambda(sec, segs_per_lambda)
            new_nseg += sec.nseg

        if initial_nseg != new_nseg:
            warnings.warn(f"Model nseg changed from {initial_nseg} to {new_nseg}.", RuntimeWarning)
  
    def merge_synapses(self, cell: object = None, py_synapses_list: list = None):
        if cell is None or py_synapses_list is None:
            return
    
        synapses_dict = {}
        seen_keys = set()
        
        # Build the unique set of synapses using a dictionary.
        # If duplicates are found, we move netcons to the first instance of the synapse.
        for this_synapse in py_synapses_list:
            synapse_key = (
                this_synapse.syn_type,
                this_synapse.gmax,
                this_synapse.synapse_hoc_obj.get_segment()
            )
            
            if synapse_key in seen_keys:
                # If this synapse is a duplicate, move its netcons to the previously-seen instance.
                other_synapse = synapses_dict[synapse_key]
                
                for netcon in this_synapse.ncs:
                    netcon.setpost(other_synapse.synapse_hoc_obj)
                    other_synapse.ncs.append(netcon)
                
                # Clear the netcon list of the duplicate synapse.
                this_synapse.ncs.clear()
            else:
                # If this is a new synapse, add it to the dictionary and the set.
                synapses_dict[synapse_key] = this_synapse
                seen_keys.add(synapse_key)
        
        # If a cell object is provided, update its synapses list to reflect the unique set of synapses.
        cell.synapses = list(synapses_dict.values())
