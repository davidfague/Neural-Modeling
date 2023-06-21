import random
from Modules.synapse import Synapse
from Modules.cell_model import CellModel

class SynapseGenerator:

	def __init__(self):

		# List of lists of synapses that were generated using this class
		self.synapses = []

	def add_synapses(self, segments: list, gmax: object, syn_mod: str, density: float = None, 
					 number_of_synapses: int = None, probs: list = None, record: bool = False, 
					 syn_params: dict = None) -> None:
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
		if callable(gmax): # gmax is distribution
			for _ in range(number_of_synapses):
				segment = random.choices(segments, probs)[0]
				synapses.append(Synapse(segment, syn_mod = syn_mod, gmax = gmax(size = 1), record = record, syn_params = syn_params))
		else: # gmax is float
			for _ in range(number_of_synapses):
				segment = random.choices(segments, probs)[0]
				synapses.append(Synapse(segment, syn_mod = syn_mod, gmax = gmax, record = record, syn_params = syn_params))
							 
		self.synapses.append(synapses)
		return synapses	
	
	def add_synapses_to_cell(self, cell: CellModel, segments: list, gmax: object, syn_mod: str, 
			  				 density: float = None, number_of_synapses: int = None, probs: list = None, 
							 record: bool = False, syn_params: dict = None) -> None:
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
		synapses = self.add_synapses(segments = segments, gmax = gmax, syn_mod = syn_mod, density = density, 
			       					 number_of_synapses = number_of_synapses, probs = probs, record = record,
									 syn_params = syn_params)
		for syn in synapses:
			cell.synapses.append(syn)
