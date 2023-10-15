from enum import Enum
from neuron import h
import pickle
import os
import numpy as np

from logger import Logger
from cell_utils import get_segments_and_len_per_segment
from synapse_generator import SynapseGenerator
from spike_generator import SpikeGenerator

from functools import partial
import scipy.stats as st
from constants import SimulationParameters

class CellType(Enum):
	Hay = {
		"biophys": "L5PCbiophys3ActiveBasal.hoc",
		"morph": "cell1.asc",
		"template": "L5PCtemplate.hoc",
		"pickle": None
		}
	HayNeymotin = {
		"biophys": "M1_soma_L5PC_dendrites.hoc",
		"morph": "cell1.asc",
		"template": "L5PCtemplate.hoc",
		"pickle": "../cells/pickled_parameters/neymotin_detailed/PT5B_full_cellParams.pkl"
	}
	NeymotinReduced = {
		"biophys": None,
		"morph": None,
		"template": "ziao_templates.hoc",
		"pickle": None
	}
	NeymotinDetailed = {
		"biophys": None,
		"morph": None,
		"template": "PTcell.hoc",
		"pickle": None
	}

class CellBuilder:

	templates_folder = "../cells/templates"

	def __init__(self, cell_type: CellType, parameters: SimulationParameters, logger: Logger) -> None:

		self.cell_type = cell_type
		self.parameters = parameters
		self.logger = logger

	def build_cell(self):

		random_state = np.random.RandomState(self.parameters.numpy_random_state)
		neuron_r = h.Random()
		neuron_r.MCellRan4(self.parameters.neuron_random_state)

		# Build complex cell
		self.logger.log(f"Building {self.cell_type}.")

		if self.cell_type == CellType.Hay:
			complex_cell = self.build_Hay_cell()

		elif self.cell_type == CellType.HayNeymotin:
			complex_cell = self.build_HayNeymotin_cell()

		elif self.cell_type == CellType.NeymotinDetailed:
			complex_cell = self.build_Neymotin_detailed_cell()

		# Increase nseg for complex cell for clustering of synapses by kmeans on segments
		for sec in complex_cell.all:
			sec.nseg = int(sec.L)+1

			all_segments, all_len_per_segment, all_SA_per_segment,\
			all_segments_center, soma_segments, soma_len_per_segment,\
			soma_SA_per_segment, soma_segments_center, no_soma_segments,\
			no_soma_len_per_segment, no_soma_SA_per_segment, no_soma_segments_center =\
			get_segments_and_len_per_segment(complex_cell)
		
		# Excitatory gmax distribution
		exc_gmax_mean_0 = self.parameters.exc_gmax_mean_0
		exc_gmax_std_0 = self.parameters.exc_gmax_std_0

		gmax_mean = np.log(exc_gmax_mean_0) - 0.5 * np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1)
		gmax_std = np.sqrt(np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1))

		# gmax distribution
		def log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size):
			val = np.random.lognormal(gmax_mean, gmax_std, size)
			s = gmax_scalar * float(np.clip(val, self.parameters.exc_gmax_clip[0], self.parameters.exc_gmax_clip[1]))
			return s

		gmax_exc_dist = partial(log_norm_dist, gmax_mean, gmax_std, self.parameters.exc_scalar, size = 1)

		# Excitatory firing rate distribution
		def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
			return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, 
											loc = loc, scale = scale, size = size)) + 1e-15
		
		spike_generator = SpikeGenerator()
		synapse_generator = SynapseGenerator()

		# Distribution of mean firing rates
		mean_fr_dist = partial(exp_levy_dist, alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1)
		
		# release probability distribution
		def P_release_dist(P_mean, P_std, size):
			val = np.random.normal(P_mean, P_std, size)
			s = float(np.clip(val, 0, 1))
			return s
		
		# exc release probability distribution everywhere
		exc_P_dist = partial(P_release_dist, P_mean = self.parameters.exc_P_release_mean, P_std = self.parameters.exc_P_release_std, size = 1)
		
		# New list to change probabilty of exc functional group nearing soma
		adjusted_no_soma_len_per_segment = []
		for i, seg in enumerate(no_soma_segments):
			if str(type(complex_cell.soma)) != "<class 'nrn.Section'>": # cell.soma is a list of sections
				if h.distance(seg, complex_cell.soma[0](0.5)) < 75:
					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
				elif seg in complex_cell.apic[0]: # trunk
					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 5)
				else:
					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])
			else: # cell.soma is a section
				if h.distance(seg, complex_cell.soma(0.5)) < 75:
					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
				elif seg in complex_cell.apic[0]: # trunk
					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 5)
				else:
					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])

		if self.parameters.CI_on:
			exc_synapses = []
		else:
			if self.parameters.use_SA_exc: # Use surface area instead of lengths for probabilities
				exc_synapses = synapse_generator.add_synapses(segments = all_segments, probs = all_SA_per_segment, 
												  density = self.parameters.exc_synaptic_density, record = True, 
												  vector_length = self.parameters.save_every_ms, gmax = gmax_exc_dist,
												  random_state = random_state, neuron_r = neuron_r,
												  syn_mod = self.parameters.exc_syn_mod,
												  P_dist = exc_P_dist, syn_params = self.parameters.exc_syn_params[0])
                                                    
			else: # Use lengths as probabilities
				exc_synapses = synapse_generator.add_synapses(segments = no_soma_segments, probs = no_soma_len_per_segment,
												  density = self.parameters.exc_synaptic_density, record = True,
												  vector_length = self.parameters.save_every_ms, gmax = gmax_exc_dist,
												  random_state = random_state, neuron_r = neuron_r,
												  syn_mod = self.parameters.exc_syn_mod,
												  P_dist = exc_P_dist, syn_params = self.parameters.exc_syn_params[0])


	def build_Hay_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, CellType.Hay.value["biophys"]))

		# Load morphology
		h.load_file("import3d.hoc")

		# Load template
		h.load_file(os.path.join(self.templates_folder, CellType.Hay.value["template"]))

		# Build complex_cell object
		complex_cell = h.L5PCtemplate(os.path.join(self.templates_folder, CellType.Hay.value["morph"]))

		return complex_cell

	def build_HayNeymotin_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, CellType.HayNeymotin.value["biophys"]))

		# Load morphology
		h.load_file("import3d.hoc")

		# Load template
		h.load_file(os.path.join(self.templates_folder, CellType.HayNeymotin.value["template"]))

		# Build complex_cell object
		complex_cell = h.L5PCtemplate(os.path.join(self.templates_folder, CellType.HayNeymotin.value["morph"]))

		# Swap soma and axon with the parameters from the pickle
		soma = complex_cell.soma[0] if self.is_indexable(complex_cell.soma) else complex_cell.soma
		axon = complex_cell.axon[0] if self.is_indexable(complex_cell.axon) else complex_cell.axon
		self.set_pickled_parameters_to_sections((soma, axon), CellType.HayNeymotin["pickle"])

		return complex_cell

	def build_Neymotin_detailed_cell(self) -> object:
		h.load_file(os.path.join(self.templates_folder, CellType.NeymotinDetailed.value["template"]))
		complex_cell = h.CP_Cell(3, 3, 3)

		return complex_cell

	def build_Neymotin_reduced_cell(self) -> object:
		h.load_file(os.path.join(self.templates_folder, CellType.NeymotinReduced.value["template"]))
		complex_cell = h.CP_Cell()

		return complex_cell

	def is_indexable(self, obj: object):
		"""
		Check if the object is indexable.
		"""
		try:
			_ = obj[0]
			return True
		except:
			return False
		
	def set_pickled_parameters_to_sections(self, sections: tuple, path: str):

		with open(path, 'rb') as file:
			params = pickle.load(file, encoding = 'latin1')

		for sec in sections:
			section_name = sec.name().split(".")[1]  # Remove Cell from name

			if "[" in section_name:
				section_type, section_type_index = section_name.split("[")
				section_type_index = section_type_index.strip("]")
				
				# Concatenate with "_"
				section_name_as_stored_in_pickle = f"{section_type}" #_{section_type_index}"
			else:
				# For sections like soma and axon
				section_name_as_stored_in_pickle = section_name  
		
			if section_name_as_stored_in_pickle in params['secs']:
				self.assign_parameters_to_section(sec, params['secs'][section_name_as_stored_in_pickle])
			else:
				raise ValueError(f"No parameters found for {section_name_as_stored_in_pickle}.")
					
	def assign_parameters_to_section(self, sec, section_data):

		# List of common state variables
		state_variables = []  # e.g. 'o_na', 'o_k', 'o_ca', 'm', 'h', 'n', 'i_na', ...
		
		# Initialize a dictionary for the section
		section_row = {'Section': sec.name()}
		
		# Set and record geometry parameters
		geom = section_data.get('geom', {})
		for param, value in geom.items():
			if str(param) not in ['pt3d']:
				setattr(sec, param, value)
				section_row[f"geom.{param}"] = value
		
		# Set and record ion parameters
		ions = section_data.get('ions', {})
		for ion, params in ions.items():
			for param, value in params.items():
				if param not in state_variables:
					main_attr_name = f"{ion}_ion"
					if param[-1] == 'o':
						sub_attr_name = f"{ion}{param}"
					else:
						sub_attr_name = f"{param}{ion}"
						for seg in sec:
							ion_obj = getattr(seg, main_attr_name)
							setattr(ion_obj, sub_attr_name, value)
					section_row[f"ions.{ion}.{param}"] = value
		
		# Set and record mechanism parameters
		mechs = section_data.get('mechs', {})
		for mech, params in mechs.items():
			if not hasattr(sec(0.5), mech):
				sec.insert(mech)
			for param, value in params.items():
				if param not in state_variables:
					for i, seg in enumerate(sec):
						if isinstance(value, list):
							try:
								setattr(seg, f"{param}_{mech}", value[i])
							except:
								print(f"Warning: Issue setting {mech} {param} in {seg} to {value[i]}. | value type: {type(value[i])} | nseg: {sec.nseg}; len(value): {len(value)}")
						else:
							try:
								setattr(seg, f"{param}_{mech}", value)
							except:
								print(f"Warning: Issue setting {mech} {param} in {sec.name()} to {value}. | value type {type(value)}")
		
					section_row[f"mechs.{mech}.{param}"] = value

		
