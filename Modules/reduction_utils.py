import numpy as np
from neuron import h

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
            this_synapse.syn_type,
            this_synapse.gmax,
            this_synapse.h_syn.get_segment()
        )
        
        if synapse_key in seen_keys:
            # If this synapse is a duplicate, move its netcons to the previously-seen instance.
            other_synapse = synapses_dict[synapse_key]
            
            for netcon in this_synapse.ncs:
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