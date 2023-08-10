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

    def add_synapses(self, segments: list, gmax: object, syn_mod: str, 
                    random_state: np.random.RandomState, neuron_r: h.Random, density: float = None, 
                        number_of_synapses: int = None, probs: list = None, record: bool = False, 
                        syn_params: dict = None, vector_length: int = None, P_dist=None, cell:CellModel=None) -> list:
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
        synapses = []

        # Error checking
        if (density is not None) and (number_of_synapses is not None):
            raise ValueError('Cannot specify both density and number_of_synapses.')

        # Calculate probabilities if not given
        if probs is None:
            total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
            probs = [(seg.sec.L / seg.sec.nseg) / total_length for seg in segments]

        # Calculate number of synapses if given densities
        if density:
            total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
            number_of_synapses = int(total_length * density)

        # Add synapses
        if P_dist:
            if callable(P_dist): #P_dist is distribution
                if callable(gmax): # gmax is distribution
                    for i in range(number_of_synapses):
                        P = P_dist(size=1)
                        P_compare = random_state.uniform(low=0.0,high=1.0,size=1)
                        if P>=P_compare:
                            segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
                            new_syn = Synapse(segment, syn_mod = syn_mod, gmax = gmax(size = 1), record = record, syn_params = syn_params, vector_length = vector_length)
                            self.add_random_generator(syn_mod, new_syn.synapse_neuron_obj, neuron_r)
                            synapses.append(new_syn)
                else: # gmax is float
                    for i in range(number_of_synapses):
                        P = P_dist(size=1)
                        P_compare = random_state.uniform(low=0.0,high=1.0,size=1)
                        if P>=P_compare:
                            segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
                            new_syn = Synapse(segment, syn_mod = syn_mod, gmax = gmax, record = record, syn_params = syn_params, vector_length = vector_length)
                            self.add_random_generator(syn_mod, new_syn.synapse_neuron_obj, neuron_r)
                            synapses.append(new_syn)
                                        
            elif isinstance(P_dist, dict): # P_dist is dictionary
                if callable(gmax): # gmax is distribution
                    for i in range(number_of_synapses):
                        segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
                        if h.distance(segment,cell.soma[0](0.5))<100:
                            seg_type='soma'
                        else:
                            seg_type = segment.sec.name().split('.')[1][:4]
                        P = P_dist[seg_type](size=1)
                        P_compare = random_state.uniform(low=0.0,high=1.0,size=1)
                        if P>=P_compare:
                            new_syn = Synapse(segment, syn_mod = syn_mod, gmax = gmax(size = 1), record = record, syn_params = syn_params, vector_length = vector_length)
                            self.add_random_generator(syn_mod, new_syn.synapse_neuron_obj, neuron_r)
                            synapses.append(new_syn)
                else: # gmax is float
                    for i in range(number_of_synapses):
                        segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
                        if h.distance(segment,cell.soma[0](0.5))<100:
                            seg_type='soma'
                        else:
                            seg_type = segment.sec.name().split('.')[1][:4]
                        P = P_dist[seg_type](size=1)
                        P_compare = random_state.uniform(low=0.0,high=1.0,size=1)
                        if P>=P_compare:
                            new_syn = Synapse(segment, syn_mod = syn_mod, gmax = gmax, record = record, syn_params = syn_params, vector_length = vector_length)
                            self.add_random_generator(syn_mod, new_syn.synapse_neuron_obj, neuron_r)
                            synapses.append(new_syn)

        else:
            if callable(gmax): # gmax is distribution
                for _ in range(number_of_synapses):
                    segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
                    new_syn = Synapse(segment, syn_mod = syn_mod, gmax = gmax(size = 1), record = record, syn_params = syn_params, vector_length = vector_length)
                    self.add_random_generator(syn_mod, new_syn.synapse_neuron_obj, neuron_r)
                    synapses.append(new_syn)
            else: # gmax is float
                for _ in range(number_of_synapses):
                    segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
                    new_syn = Synapse(segment, syn_mod = syn_mod, gmax = gmax, record = record, syn_params = syn_params, vector_length = vector_length)
                    self.add_random_generator(syn_mod, new_syn.synapse_neuron_obj, neuron_r)
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