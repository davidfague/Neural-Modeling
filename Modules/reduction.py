import numpy as np
from neuron import h
from neuron_reduce import subtree_reductor

from Modules.cable_expander_func import cable_expander, get_hsyn_to_netcons
from Modules.synapse import Synapse
from Modules.cell_model import CellModel
from Modules.logger import Logger

import datetime

import warnings


class Reductor():
    def __init__(self, logger: Logger):
        self.logger = logger
        
    def log_with_timestamp(self, message):
        """Helper function for logging messages with a timestamp."""
        self.logger.log(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}")
    
    def reduce_cell(self, cell_model, reduction_frequency):
        original_py_to_hoc_syns = {syn: syn.h_syn for syn in cell_model.synapses}
        
        netcons_list = [netcon for synapse in cell_model.synapses for netcon in synapse.netcons]
        
        self.log_with_timestamp(f'using subtree reductor')
        reduced_skeleton_cell, hoc_synapses_list, netcons_list, txt_nr = subtree_reductor(
            cell_model.skeleton_cell, list(original_py_to_hoc_syns.values()), netcons_list, reduction_frequency, return_seg_to_seg=True)
            
        self.log_with_timestamp(f'printing topology after reduction')
        print(h.topology())
        
        self.log_with_timestamp(f'updating synapses after reduction')
        self.update_synapses_after_reduction(cell_model, netcons_list, original_py_to_hoc_syns)

        self.log_with_timestamp(f'updating cell_model sections after reduction')
        self.update_stored_sections_after_reduction(cell_model, reduced_skeleton_cell)
        
        #Debugging
        self.log_with_timestamp(f'reduced cell.all: {cell_model.all}')
        unique_netcons = set(netcon for syn in cell_model.synapses for netcon in syn.netcons) # debbuging
        if len(unique_netcons) != len(netcons_list):
          selg.log_with_timestep(f'len(unique netcons after expansion) != len(original netcons list) {len(unique_netcons)} {len(netcons_list)}')

        return cell_model
    
    def expand_cell(self, cell_model, choose_branches, reduction_frequency, random_state):
        # select expansion parameters
        sections_to_expand = [cell_model.skeleton_cell.apic[0]]
        furcations_x = [0.289004]
        nbranches = [choose_branches]
        
        original_py_to_hoc_syns = {syn: syn.h_syn for syn in cell_model.synapses}
        netcons_list = [netcon for synapse in cell_model.synapses for netcon in synapse.netcons]

        self.log_with_timestamp(f'using cable_expander')
        expanded_cell, hoc_synapses_list, netcons_list, _ = cable_expander(
            cell_model.skeleton_cell, 
            sections_to_expand, 
            furcations_x, 
            nbranches, 
            list(original_py_to_hoc_syns.values()), 
            netcons_list, 
            reduction_frequency, 
            return_seg_to_seg=True, 
            random_state=random_state
        )

        self.log_with_timestamp(f'printing topology after cable expander')
        print(h.topology())

        self.log_with_timestamp(f'updating synapses after cable expander')
        self.update_synapses_after_expansion(cell_model, netcons_list, original_py_to_hoc_syns)

        self.log_with_timestamp(f'updating stored sections after cable expander')
        self.update_stored_sections_after_expansion(cell_model, expanded_cell)
        
        #debugging
        self.log_with_timestamp(f'expanded cell.all: {cell_model.all}')
        unique_netcons = set(netcon for syn in cell_model.synapses for netcon in syn.netcons) # debbuging
        if len(unique_netcons) != len(netcons_list):
          selg.log_with_timestep(f'len(unique netcons after expansion) != len(original netcons list) {len(unique_netcons)} {len(netcons_list)}')

        return cell_model
    
    def update_synapses_after_expansion(self, cell_model, netcons_list, original_py_to_hoc_syns):
        '''The expansion may have made some original hoc synapses obsolete, and probably created new hoc synapses that will need a new python synapses.
        Netcons also probably moved around, and will need to be updated in the python Synapse.'''
        # map netcons to their h_syn
        hoc_syn_to_netcon = get_hsyn_to_netcons(netcons_list)
        # update list of all hoc synapses to only those that have a netcon
        new_hoc_synapses = set(hoc_syn_to_netcon.keys())
        # remove py synapses whose hoc synapse no longer have a netcon
        cell_model.synapses[:] = [py_syn for py_syn, hoc_syn in original_py_to_hoc_syns.items() if hoc_syn in new_hoc_synapses]

        # add py synapses for newly created hoc synapses
        hoc_synapses_before_expansion = set(hoc_synapse for py_syn, hoc_synapse in original_py_to_hoc_syns.items())
        for hoc_syn in new_hoc_synapses: # all hoc_synapses after cable_expander
            if hoc_syn not in hoc_synapses_before_expansion:
                new_syn = Synapse(segment=None, syn_mod=None, syn_params=None, gmax=None, neuron_r=None, name=None, hoc_syn=hoc_syn)
                # add map netcon to new syn
                new_syn.netcons = hoc_syn_to_netcon[hoc_syn]
                cell_model.synapses.append(new_syn)
                original_py_to_hoc_syns[new_syn] = hoc_syn
        
        if len(new_hoc_synapses) != len(cell_model.synapses):
            self.log_with_timestamp(f'WARNING: len(new_hoc_synapses) != len(cell_model.synapses).. {len(new_hoc_synapses)} != {len(cell_model.synapses)}')
        
        # update the netcons stored in py synapses
        self.update_py_syn_netcons(cell_model.synapses, hoc_syn_to_netcon)

    def update_synapses_after_reduction(self, cell_model, netcons_list, original_py_to_hoc_syns):
        # map netcons to their h_syn
        hoc_syn_to_netcon = get_hsyn_to_netcons(netcons_list)
        # update hoc synapses to those that have a netcon
        new_hoc_synapses = set(hoc_syn_to_netcon.keys())
        # remove the py synapses whose h_syn do not have a netcon
        cell_model.synapses[:] = [syn for syn, hoc_syn in original_py_to_hoc_syns.items() if hoc_syn in new_hoc_synapses]
        # update the netcons stored in py synapses
        self.update_py_syn_netcons(cell_model.synapses, hoc_syn_to_netcon)
        
        if len(new_hoc_synapses) != len(cell_model.synapses):
            self.log_with_timestamp(f'WARNING: len(new_hoc_synapses) != len(cell_model.synapses).. {len(new_hoc_synapses)} != {len(cell_model.synapses)}')

    def update_py_syn_netcons(self, py_synapses_list, hoc_syn_to_netcon):
        """Updates the netcons list of each Python Synapse object based on current hoc synapse to netcon mappings."""
        for py_syn in py_synapses_list:
            if py_syn.h_syn in hoc_syn_to_netcon:
                py_syn.netcons = hoc_syn_to_netcon[py_syn.h_syn]
            else:
                # This handles cases where a hoc synapse might no longer have netcons after reduction/expansion
                self.log_with_timestamp(f"{py_syn} has no netcons")
                py_syn.netcons = []
                
    def update_stored_sections_after_reduction(self, cell_model, reduced_skeleton_cell):
        # have to make this corrections to pass the new sections
        # need to pass skel.soma, skel.axon, AND skel.hoc_model.apic, skel.hoc_model.dend
        reduced_skeleton_cell.dend=reduced_skeleton_cell.hoc_model.dend
        reduced_skeleton_cell.apic=reduced_skeleton_cell.hoc_model.apic
        self.log_with_timestamp(f"reduced_skeleton_cell.hoc_model.all: {reduced_skeleton_cell.hoc_model.all}")
        reduced_skeleton_cell.all = reduced_skeleton_cell.hoc_model.all
        
        self.log_with_timestamp(f"updating cell_model.skeleton_cell")
        cell_model.skeleton_cell = reduced_skeleton_cell
        self.log_with_timestamp(f"updating section_lists")
        cell_model.update_section_lists()

    def update_stored_sections_after_expansion(self, cell_model, reduced_skeleton_cell):
        self.log_with_timestamp(f"updating cell_model.skeleton_cell")
        cell_model.skeleton_cell = reduced_skeleton_cell
        self.log_with_timestamp(f"updating section_lists")
        cell_model.update_section_lists()
        
###### reduction independent of neuron_reduce and cable_expander

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
    
    def merge_synapses(self, cell: object = None):#, py_synapses_list: list = None):
        synapses_dict = {}
        seen_keys = set()
        
        # Build the unique set of synapses using a dictionary.
        # If duplicates are found, we move netcons to the first instance of the synapse.
        for this_synapse in cell.synapses:
            synapse_key = (
                this_synapse.syn_mod,
                getattr(this_synapse.h_syn, this_synapse.gmax_var), # gmax
                this_synapse.h_syn.get_segment()
            )
            
            if synapse_key in seen_keys:
                # If this synapse is a duplicate, move its netcons to the previously-seen instance.
                other_synapse = synapses_dict[synapse_key]
                
                for netcon in this_synapse.netcons:
                    netcon.setpost(other_synapse.h_syn)
                    other_synapse.netcons.append(netcon)
                
                # Clear the netcon list of the duplicate synapse.
                this_synapse.netcons.clear()
            else:
                # If this is a new synapse, add it to the dictionary and the set.
                synapses_dict[synapse_key] = this_synapse
                seen_keys.add(synapse_key)
        
        # If a cell object is provided, update its synapses list to reflect the unique set of synapses.
        cell.synapses = list(synapses_dict.values())