from Modules.synapse import Synapse
from Modules.cell_model import CellModel
from neuron import h
import numpy as np

class SynapseGenerator:

    def __init__(self):

        # List of lists of synapses that were generated using this class
        self.synapses = []

        # Random generators for release probability
        self.randomgenerators = []
        
    def choose_params(self, syn_mod, syn_params, segment_distance):
        '''
        chooses between the syn_params in constants.py
        '''
        if isinstance(syn_params, list):
            if 'AMPA' in syn_mod: # excitatory
                return np.random.choice(syn_params, p=[0.9, 0.1]) # 90% first option; 10% second option
            elif 'GABA' in syn_mod: # inhibitory
                return syn_params[1] if segment_distance > 100 else syn_params[0] # second option is for > 100 um from soma, else first option
        return syn_params
        
    def create_synapse(self, segment, gmax, syn_mod, record, syn_params, vector_length, neuron_r, cell):
        g = gmax(size=1) if callable(gmax) else gmax

        if cell and self.is_indexable(cell.soma):
            segment_distance = h.distance(segment, cell.soma[0](0.5))
        elif cell:
            segment_distance = h.distance(segment, cell.soma(0.5))
        else:
            segment_distance = None  # or some default value if necessary
        
        chosen_params = self.choose_params(syn_mod, syn_params, segment_distance)
        new_syn = Synapse(segment, syn_mod=syn_mod, gmax=g, record=record, syn_params=chosen_params, vector_length=vector_length)
        self.add_random_generator(syn_mod, new_syn.synapse_hoc_obj, neuron_r)
        return new_syn


    def add_synapses(self, segments, gmax, syn_mod, random_state, neuron_r, density=None, 
                     number_of_synapses=None, probs=None, record=False, 
                     syn_params=None, vector_length=None, P_dist=None, cell=None) -> list:
        '''
        Creates a list of synapses by specifying density or number of synapses.

        Parameters:
        ----------
        segments: list
            List of neuron segments to choose from.

        gmax: float or partial
            Maximum conductance. If partial, distribution to sample from.

        syn_mod:  str 
            Name of synapse modfile, ex. 'AMPA_NMDA', 'pyr2pyr', 'GABA_AB', 'int2pyr'.

        density: float
            Density of excitatory synapses in synapses / micron.

        number_of_synapses: int 
            Number of synapses to distribute to the cell.

        probs: list 
            List of probabilities of choosing each segment. 
            If not provided, the ratio (segment_length / total_segment_length) will be used.

        record: bool = False
            Whether or not to record synapse currents.

        syn_params: dict = None
            dictionary of key: synapse attributes, and values: attribute values

        P_dist:
        Distribution of release probability. Can be dictionary containing keys: soma for perisomatic, apic, and basal, and items callable distributions, or can be single callable distribution.
        Returns:
        ----------
        synapses: list
            Added synapses.
        '''

        if (density is not None) and (number_of_synapses is not None):
            raise ValueError('Cannot specify both density and number_of_synapses.')

        if probs is None:
            total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
            probs = [seg_length / total_length for seg_length in [seg.sec.L / seg.sec.nseg for seg in segments]]

        if density:
            total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
            number_of_synapses = int(total_length * density)
            
        synapses = []
        for _ in range(number_of_synapses):
            segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
        
        # Check if we should create a synapse based on the P_dist
            if P_dist:
                if isinstance(P_dist, dict):
                    sec_type = segment.sec.name().split('.')[1][:4]
                    P = P_dist[sec_type](size=1)
                else:
                    P = P_dist(size=1)
                P_compare = random_state.uniform(low=0.0, high=1.0, size=1)
                if P < P_compare:
                    continue
        
            new_syn = self.create_synapse(segment, gmax, syn_mod, record, syn_params, vector_length, neuron_r, cell)
            synapses.append(new_syn)

        self.synapses.append(synapses)
        return synapses


    def add_synapses_to_cell(self, cell_to_add_to: CellModel, **kwargs) -> None:
        '''
        Add synapses to cell after cell python object has already been initialized.

        Parameters:
        ----------
        cell: CellModel
            Cell to add to.

        segments: list
            List of neuron segments to choose from.

        gmax: float or partial
            Maximum conductance. If partial, distribution to sample from.

        syn_mod:  str 
            Name of synapse modfile, ex. 'AMPA_NMDA', 'pyr2pyr', 'GABA_AB', 'int2pyr'.

        density: float
            Density of excitatory synapses in synapses / micron.

        number_of_synapses: int 
            Number of synapses to distribute to the cell.

        probs: list 
            List of probabilities of choosing each segment. 
            If not provided, the ratio (segment_length / total_segment_length) will be used.

        record: bool = False
            Whether or not to record synapse currents.

        syn_params: dict = None
            dictionary of key: synapse attributes, and values: attribute values

        '''
        synapses = self.add_synapses(**kwargs)
        for syn in synapses:
            cell_to_add_to.synapses.append(syn)
            
            
    def add_random_generator(self, syn_mod, synapse, r: h.Random):				 
        if syn_mod in ['pyr2pyr', 'int2pyr']:
            r.uniform(0, 1)
            synapse.setRandObjRef(r)
        
        # A list of random generators is kept so that they are not automatically garbaged
        self.randomgenerators.append(r)
    
    def is_indexable(self, obj: object):
        """
        Check if the object is indexable.
        """
        try:
            _ = obj[0]
            return True
        except:
            return False