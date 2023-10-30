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
        
    def create_synapse(self, segment, gmax, syn_mod, record, syn_params, vector_length, neuron_r, cell):
      if callable(gmax):
          g = gmax(size=1)
      else:
          g = gmax
      if isinstance(syn_params, list):
          if 'AMPA' in syn_mod:
            chosen_params = np.random.choice(syn_params, p=[0.9, 0.1]) # PC2PN and PN2PN
          elif 'GABA' in syn_mod:
            if str(type(cell.soma)) != "<class 'nrn.Section'>":
              if h.distance(segment, cell.soma[0](0.5)) > 100: # distal inh
                chosen_params = syn_params[1]
              elif h.distance(segment, cell.soma[0](0.5)) < 100: # perisomatic inh
                chosen_params = syn_params[0]
            else:
              if h.distance(segment, cell.soma(0.5)) > 100: # distal inh
                chosen_params = syn_params[1]
              elif h.distance(segment, cell.soma(0.5)) < 100: # perisomatic inh
                chosen_params = syn_params[0]
      else:
          chosen_params = syn_params
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
        synapses = []

        # Error checking
        if (density is not None) and (number_of_synapses is not None):
            raise ValueError('Cannot specify both density and number_of_synapses.')
    
        # Calculate probabilities if not given
        if probs is None:
            total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
            probs = [(seg.sec.L / seg.sec.nseg) / total_length for seg in segments]
    
        if density:
            total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
            number_of_synapses = int(total_length * density)
    
        synapses = []
        for i in range(number_of_synapses):
            P = 1  # default to always creating a synapse
    
            segment = random_state.choice(segments, 1, True, probs / np.sum(probs))[0]
            if P_dist:
                if isinstance(P_dist, dict):
                    if str(type(cell.soma)) != "<class 'nrn.Section'>":
                      seg_type = 'soma' if h.distance(segment, cell.soma[0](0.5)) < 100 else segment.sec.name().split('.')[1][:4]
                    else:
                      seg_type = 'soma' if h.distance(segment, cell.soma(0.5)) < 100 else segment.sec.name().split('.')[1][:4]
                    P = P_dist[seg_type](size=1)
                else:
                    P = P_dist(size=1)
            
            P_compare = random_state.uniform(low=0.0, high=1.0, size=1)
            if P >= P_compare:
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