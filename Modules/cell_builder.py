from enum import Enum
import pickle
import os
import numpy as np

from functools import partial
import scipy.stats as st

from neuron import h

from Modules.logger import Logger
from Modules.cell_utils import get_segments_and_len_per_segment
from Modules.synapse_generator import SynapseGenerator
from Modules.spike_generator import SpikeGenerator
from Modules.constants import SimulationParameters
from Modules.cell_model import CellModel
from Modules.clustering import create_functional_groups_of_presynaptic_cells
from Modules.reduction import Reductor

class SkeletonCell(Enum):
	Hay = {
		"biophys": "L5PCbiophys3ActiveBasal.hoc",
		"morph": "cell1.asc",
		"template": "L5PCtemplate.hoc",
		"pickle": None,
		"modfiles": "../modfiles/hay"
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

def log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size, clip):
	val = np.random.lognormal(gmax_mean, gmax_std, size)
	s = gmax_scalar * float(np.clip(val, clip[0], clip[1]))
	return s

# Firing rate distribution
def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
	return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, loc = loc, scale = scale, size = size)) + 1e-15

def gamma_dist(mean, size = 1):
	shape = 5
	scale = mean / shape
	return np.random.gamma(shape, scale, size) + 1e-15

# Release probability distribution
def P_release_dist(P_mean, P_std, size):
	val = np.random.normal(P_mean, P_std, size)
	s = float(np.clip(val, 0, 1))
	return s

class CellBuilder:

	templates_folder = "../cells/templates"

	def __init__(self, cell_type: SkeletonCell, parameters: SimulationParameters, logger: Logger) -> None:

		self.cell_type = cell_type
		self.parameters = parameters
		self.logger = logger

	def build_cell(self):

		random_state = np.random.RandomState(self.parameters.numpy_random_state)
		np.random.seed(self.parameters.numpy_random_state)
		neuron_r = h.Random()
		neuron_r.MCellRan4(self.parameters.neuron_random_state)

		# Build complex cell
		self.logger.log(f"Building {self.cell_type}.")

		if self.cell_type == SkeletonCell.Hay:
			skeleton_cell = self.build_Hay_cell()

		elif self.cell_type == SkeletonCell.HayNeymotin:
			skeleton_cell = self.build_HayNeymotin_cell()

		elif self.cell_type == SkeletonCell.NeymotinDetailed:
			skeleton_cell = self.build_Neymotin_detailed_cell()

		# Increase nseg for complex cell for clustering of synapses by kmeans on segments
		for sec in skeleton_cell.all:
			sec.nseg = int(sec.L) + 1

		# Get segments to apply trains to
		(all_segments, all_len_per_segment, all_SA_per_segment, 
   		 all_segments_center, soma_segments, soma_len_per_segment, 
		 soma_SA_per_segment, soma_segments_center, no_soma_segments, 
		 no_soma_len_per_segment, no_soma_SA_per_segment, no_soma_segments_center) = get_segments_and_len_per_segment(skeleton_cell)
		
		synapse_generator = SynapseGenerator()
   
		# Update parameters from dictionary
		if self.parameters.use_param_update_dict:
			self.update_cell_parameters_from_dict(skeleton_cell, self.parameters.param_update_dict)

		# Build synapses
		self.logger.log("Building excitatory synapses.")
		exc_synapses = self.build_excitatory_synapses(
			skeleton_cell = skeleton_cell,
			synapse_generator = synapse_generator,
			no_soma_segments = no_soma_segments,
			no_soma_len_per_segment = no_soma_len_per_segment,
			all_segments = all_segments,
			all_SA_per_segment = all_SA_per_segment,
			random_state = random_state,
			neuron_r = neuron_r
		)

		self.logger.log("Building inhibitory synapses.")
		inh_synapses = self.build_inhibitory_synapses(
			skeleton_cell = skeleton_cell,
			synapse_generator = synapse_generator,
			all_segments = all_segments,
			all_SA_per_segment = all_SA_per_segment,
			random_state = random_state,
			neuron_r = neuron_r
		)

		self.logger.log("Building soma synapses.")
		soma_inh_synapses = self.build_soma_synapses(
			synapse_generator = synapse_generator,
			soma_segments = soma_segments,
			soma_SA_per_segment = soma_SA_per_segment,
			random_state = random_state,
			neuron_r = neuron_r
		)

		# Get all synapses
		all_syns_before_reduction = [synapse for synapses_list in synapse_generator.synapses for synapse in synapses_list]
		self.logger.log(f"Number of Synapses Before Reduction: {len(all_syns_before_reduction)}")

		
		# Initialize the dummy cell model used for calculating coordinates and 
		# generating functional groups
		dummy_cell = CellModel(hoc_model = skeleton_cell, random_state = random_state)

		# Create functional groups
		spike_generator = SpikeGenerator()

		self.logger.log("Building excitatory functional groups.")
		_ = self.build_excitatory_functional_groups(
			cell = dummy_cell,
			exc_synapses = exc_synapses,
			spike_generator = spike_generator,
			random_state = random_state
		)
		exc_spikes = spike_generator.spike_trains.copy()

		self.logger.log("Building inhibitory functional groups.")
		_ = self.build_inhibitory_functional_groups(
			cell = dummy_cell,
			inh_synapses = inh_synapses,
			spike_generator = spike_generator,
			random_state = random_state,
			exc_spikes = exc_spikes
		)

		self.logger.log("Building soma functional groups.")
		_ = self.build_soma_functional_groups(
			cell = dummy_cell,
			soma_inh_synapses = soma_inh_synapses,
			spike_generator = spike_generator,
			random_state = random_state,
			exc_spikes = exc_spikes
		)

		self.detailed_seg_info = dummy_cell.seg_info.copy()

		# Build the final cell
		self.logger.log("Creating a CellModel object.")

		reductor = Reductor(logger = self.logger)
		cell = reductor.reduce_cell(
			complex_cell = skeleton_cell, 
			reduce_cell = self.parameters.reduce_cell, 
			optimize_nseg = self.parameters.optimize_nseg_by_lambda, 
			py_synapses_list = all_syns_before_reduction,
			netcons_list = spike_generator.netcons, 
			spike_trains = spike_generator.spike_trains,
			spike_threshold = self.parameters.spike_threshold, 
			random_state = random_state,
			var_names = self.parameters.channel_names, 
			reduction_frequency = self.parameters.reduction_frequency, 
			expand_cable = self.parameters.expand_cable, 
			choose_branches = self.parameters.choose_branches,
      vector_length = self.parameters.vector_length)
      
		self.logger.log("Finish creating a CellModel object.")
   
		if (not self.parameters.CI_on) and (not self.parameters.trunk_exc_synapses):
			# Turn off certain presynaptic neurons to simulate in vivo
			for synapse in cell.synapses:
				if (synapse.get_segment().sec in cell.apic) and (synapse.syn_type in self.parameters.exc_syn_mod) and (synapse.get_segment().sec not in cell.obliques) and (synapse.get_segment().sec.y3d(0) < 600):
					for netcon in synapse.ncs: netcon.active(False)
		
		# Turn off perisomatic exc neurons
		if not self.parameters.perisomatic_exc_synapses:
			for synapse in cell.synapses:
				if (h.distance(synapse.get_segment(), cell.soma[0](0.5)) < 75) and (synapse.syn_type in self.parameters.exc_syn_mod):
					for netcon in synapse.ncs: netcon.active(False)
		
		# Merge synapses
		if self.parameters.merge_synapses:
			reductor.merge_synapses(cell)

		# Set recorders
		cell.setup_recorders(vector_length = self.parameters.vector_length)

		# Add current injection
		if self.parameters.CI_on:
			cell.add_injection(
				sec_index = cell.all.index(cell.soma[0]), 
				record = True, 
				delay = self.parameters.h_i_delay, 
				dur = self.parameters.h_i_duration, 
				amp = self.parameters.h_i_amplitude)
			
		self.logger.log(f"There were {len(cell.errors_in_setting_params)} errors when trying to insert unused channels.")
		self.logger.log(f"The Sections in cell.all before returning from cell_builder.build_cell(): {cell.all}")

		return cell, dummy_cell, cell.synapses.copy()

	def build_soma_functional_groups(self, cell, soma_inh_synapses, spike_generator, random_state, exc_spikes):

		soma_coordinates = np.zeros(3)
		segment_coordinates = np.zeros((len(cell.seg_info), 3))

		for ind, seg in enumerate(cell.seg_info):
			segment_coordinates[ind, 0] = seg['p0.5_x3d']
			segment_coordinates[ind, 1] = seg['p0.5_y3d']
			segment_coordinates[ind, 2] = seg['p0.5_z3d']
	
			if seg['seg'] == cell.soma[0](0.5):
				soma_coordinates[0] = seg['p0.5_x3d']
				soma_coordinates[1] = seg['p0.5_y3d']
				soma_coordinates[2] = seg['p0.5_z3d']

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		# Distal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_distal_mean_fr, self.parameters.inh_distal_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		t = np.arange(0, self.parameters.h_tstop, 1)

		inh_soma_functional_groups = create_functional_groups_of_presynaptic_cells(
			segments_coordinates = segment_coordinates,
			n_functional_groups = 1,
			n_presynaptic_cells_per_functional_group = 1,
			name_prefix = 'soma_inh',
			cell = cell, 
			synapses = soma_inh_synapses, 
			proximal_fr_dist = proximal_inh_dist, 
			distal_fr_dist=distal_inh_dist, 
			spike_generator=spike_generator, 
			t = t, 
			random_state = random_state, 
			spike_trains_to_delay = exc_spikes, 
			fr_time_shift = self.parameters.inh_firing_rate_time_shift, 
			soma_coordinates = soma_coordinates, 
			method = 'delay')
		
		return inh_soma_functional_groups


	def build_inhibitory_functional_groups(self, cell, inh_synapses, spike_generator, random_state, exc_spikes):

		soma_coordinates = np.zeros(3)
		segment_coordinates = np.zeros((len(cell.seg_info), 3))

		for ind, seg in enumerate(cell.seg_info):
			segment_coordinates[ind, 0] = seg['p0.5_x3d']
			segment_coordinates[ind, 1] = seg['p0.5_y3d']
			segment_coordinates[ind, 2] = seg['p0.5_z3d']
	
			if seg['seg'] == cell.soma[0](0.5):
				soma_coordinates[0] = seg['p0.5_x3d']
				soma_coordinates[1] = seg['p0.5_y3d']
				soma_coordinates[2] = seg['p0.5_z3d']

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		# Distal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_distal_mean_fr, self.parameters.inh_distal_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		t = np.arange(0, self.parameters.h_tstop, 1)

		inh_distributed_functional_groups = create_functional_groups_of_presynaptic_cells(
			segments_coordinates = segment_coordinates,
			n_functional_groups = self.parameters.inh_distributed_n_FuncGroups,
			n_presynaptic_cells_per_functional_group = self.parameters.inh_distributed_n_PreCells_per_FuncGroup,
			name_prefix = 'inh',
			cell = cell, 
			synapses = inh_synapses,
			proximal_fr_dist = proximal_inh_dist, 
			distal_fr_dist = distal_inh_dist, 
			spike_generator = spike_generator, 
			t = t, 
			random_state = random_state, 
			spike_trains_to_delay = exc_spikes, 
			fr_time_shift = self.parameters.inh_firing_rate_time_shift, 
			soma_coordinates = soma_coordinates, 
			method = 'delay')
		
		return inh_distributed_functional_groups


	def build_excitatory_functional_groups(self, cell, exc_synapses, spike_generator, random_state):
		
		soma_coordinates = np.zeros(3)
		segment_coordinates = np.zeros((len(cell.seg_info), 3))

		for ind, seg in enumerate(cell.seg_info):
			segment_coordinates[ind, 0] = seg['p0.5_x3d']
			segment_coordinates[ind, 1] = seg['p0.5_y3d']
			segment_coordinates[ind, 2] = seg['p0.5_z3d']
	
			if seg['seg'] == cell.soma[0](0.5):
				soma_coordinates[0] = seg['p0.5_x3d']
				soma_coordinates[1] = seg['p0.5_y3d']
				soma_coordinates[2] = seg['p0.5_z3d']

		# Distribution of mean firing rates
		#mean_fr_dist = partial(exp_levy_dist, alpha = 1.37, beta = -1.00, loc = 5.3, scale = 0.44, size = 1)
		mean_fr_dist = partial(gamma_dist, mean = 5.3, size = 1)

		t = np.arange(0, self.parameters.h_tstop, 1)

		exc_functional_groups = create_functional_groups_of_presynaptic_cells(
			segments_coordinates = segment_coordinates,
			n_functional_groups = self.parameters.exc_n_FuncGroups,
			n_presynaptic_cells_per_functional_group = self.parameters.exc_n_PreCells_per_FuncGroup,
			name_prefix = 'exc',
			synapses = exc_synapses, 
			cell = cell, 
			mean_firing_rate = mean_fr_dist, 
			spike_generator = spike_generator, 
			t = t, 
			random_state = random_state, 
			method = '1f_noise')
		
		return exc_functional_groups

				
	def build_soma_synapses(
			self, 
			synapse_generator, 
			soma_segments, 
			soma_SA_per_segment, 
			random_state, 
			neuron_r) -> list:
		
		if (self.parameters.CI_on) or (not self.parameters.add_soma_inh_synapses):
			return []
		
		inh_soma_P_dist = partial(P_release_dist, P_mean = self.parameters.inh_soma_P_release_mean, P_std = self.parameters.inh_soma_P_release_std, size = 1)
		
		soma_inh_synapses = synapse_generator.add_synapses(
			segments = soma_segments,
			probs = soma_SA_per_segment,
			number_of_synapses = self.parameters.num_soma_inh_syns,
			record = True,
			vector_length = self.parameters.vector_length,
			gmax = self.parameters.soma_gmax_dist,
			random_state=random_state,
			neuron_r = neuron_r,
			syn_mod = self.parameters.inh_syn_mod,
			P_dist = inh_soma_P_dist)
	
		return soma_inh_synapses
			
	def build_inhibitory_synapses(
			self, 
			skeleton_cell, 
			synapse_generator, 
			all_segments, 
			all_SA_per_segment, 
			random_state, 
			neuron_r) -> list:
		
		# Inhibitory release probability distributions
		inh_soma_P_dist = partial(P_release_dist, P_mean = self.parameters.inh_soma_P_release_mean, P_std = self.parameters.inh_soma_P_release_std, size = 1)
		inh_apic_P_dist = partial(P_release_dist, P_mean = self.parameters.inh_apic_P_release_mean, P_std = self.parameters.inh_apic_P_release_std, size = 1)
		inh_basal_P_dist = partial(P_release_dist, P_mean = self.parameters.inh_basal_P_release_mean, P_std = self.parameters.inh_basal_P_release_std, size = 1)
		
		inh_P_dist = {}
		inh_P_dist["soma"] = inh_soma_P_dist
		inh_P_dist["apic"] = inh_apic_P_dist
		inh_P_dist["dend"] = inh_basal_P_dist

		if self.parameters.CI_on:
			return []
		
		inh_synapses = synapse_generator.add_synapses(
			segments = all_segments, 
			probs = all_SA_per_segment, 
			density = self.parameters.inh_synaptic_density,
			record = True,
			vector_length = self.parameters.vector_length,
			gmax = self.parameters.inh_gmax_dist,
			random_state = random_state,
			neuron_r = neuron_r,
			syn_mod = self.parameters.inh_syn_mod,
			P_dist = inh_P_dist,
			cell = skeleton_cell, # Redundant? # no. weight changes be distance from soma.
			syn_params = self.parameters.inh_syn_params)
		
		return inh_synapses
			
	def build_excitatory_synapses(
			self, 
			skeleton_cell,
			synapse_generator,
			no_soma_segments, 
			no_soma_len_per_segment,
			all_segments,
			all_SA_per_segment,
			random_state,
			neuron_r) -> list:

		# Excitatory gmax distribution
		exc_gmax_mean_0 = self.parameters.exc_gmax_mean_0
		exc_gmax_std_0 = self.parameters.exc_gmax_std_0

		gmax_mean = np.log(exc_gmax_mean_0) - 0.5 * np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1)
		gmax_std = np.sqrt(np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1))

		# gmax distribution
		gmax_exc_dist = partial(
			log_norm_dist, 
			gmax_mean, 
			gmax_std, 
			self.parameters.exc_scalar, 
			size = 1, 
			clip = self.parameters.exc_gmax_clip)
		
		# exc release probability distribution everywhere
		exc_P_dist = partial(
			P_release_dist, 
			P_mean = self.parameters.exc_P_release_mean, 
			P_std = self.parameters.exc_P_release_std, 
			size = 1)
		
#		# New list to change probabilty of exc functional group nearing soma
#		adjusted_no_soma_len_per_segment = []
#		for i, seg in enumerate(no_soma_segments):
#			if str(type(skeleton_cell.soma)) != "<class 'nrn.Section'>": # cell.soma is a list of sections
#				if h.distance(seg, skeleton_cell.soma[0](0.5)) < 75:
#					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
#				elif seg in skeleton_cell.apic[0]: # trunk
#					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 5)
#				else:
#					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])
#			else: # cell.soma is a section
#				if h.distance(seg, skeleton_cell.soma(0.5)) < 75:
#					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
#				elif seg in skeleton_cell.apic[0]: # trunk
#					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 5)
#				else:
#					adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])

		if self.parameters.CI_on:
			return []
		
		if self.parameters.use_SA_exc: # Use surface area instead of lengths for probabilities
			segments = all_segments
			probs = all_SA_per_segment
		else:
			segments = no_soma_segments
			probs = no_soma_len_per_segment

		exc_synapses = synapse_generator.add_synapses(
			segments = segments, 
			probs = probs, 
			density = self.parameters.exc_synaptic_density, 
			record = True, 
			vector_length = self.parameters.vector_length, 
			gmax = gmax_exc_dist,
			random_state = random_state, 
			neuron_r = neuron_r,
			syn_mod = self.parameters.exc_syn_mod,
			P_dist = exc_P_dist, 
			syn_params = self.parameters.exc_syn_params[0])
		
		return exc_synapses

	def build_Hay_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.Hay.value["biophys"]))

		# Load morphology
		h.load_file("import3d.hoc")

		# Load template
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.Hay.value["template"]))

		# Build skeleton_cell object
		skeleton_cell = h.L5PCtemplate(os.path.join(self.templates_folder, SkeletonCell.Hay.value["morph"]))

		return skeleton_cell

	def build_HayNeymotin_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.HayNeymotin.value["biophys"]))

		# Load morphology
		h.load_file("import3d.hoc")

		# Load template
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.HayNeymotin.value["template"]))

		# Build skeleton_cell object
		skeleton_cell = h.L5PCtemplate(os.path.join(self.templates_folder, SkeletonCell.HayNeymotin.value["morph"]))

		# Swap soma and axon with the parameters from the pickle
		soma = skeleton_cell.soma[0] if self.is_indexable(skeleton_cell.soma) else skeleton_cell.soma
		axon = skeleton_cell.axon[0] if self.is_indexable(skeleton_cell.axon) else skeleton_cell.axon
		self.set_pickled_parameters_to_sections((soma, axon), SkeletonCell.HayNeymotin["pickle"])

		return skeleton_cell

	def build_Neymotin_detailed_cell(self) -> object:
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.NeymotinDetailed.value["template"]))
		skeleton_cell = h.CP_Cell(3, 3, 3)

		return skeleton_cell

	def build_Neymotin_reduced_cell(self) -> object:
		h.load_file(os.path.join(self.templates_folder, SkeletonCell.NeymotinReduced.value["template"]))
		skeleton_cell = h.CP_Cell()

		return skeleton_cell

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

# TODO    ( commented because inconsistent use of tabs and spaces)                                            
#    def update_cell_parameters_from_dict(self, cell, update_dict):
#        # Setting reference for distance
#        if self.is_indexable(cell.soma):
#            h.distance(sec=cell.soma[0])
#        else:
#            h.distance(sec=cell.soma)
#                    
#        for sec_type in update_dict.keys():
#            sections_to_update = getattr(cell, sec_type)
#            
#            if not self.is_indexable(sections_to_update):
#                sections_to_update = [sections_to_update]
#
#            for attribute_to_update, values in update_dict[sec_type].items():
#                att_and_sub_atts = attribute_to_update.split('.')
#                initial_att = att_and_sub_atts[0]
#                sec_or_seg_att = None
#                    
#                if hasattr(sections_to_update[0], initial_att):
#                    sec_or_seg_att = 'sec'
#                elif hasattr(sections_to_update[0](0.5), initial_att):
#                    sec_or_seg_att = 'seg'
#                else:
#                    raise AttributeError(f"{initial_att} of {att_and_sub_atts} is not found in either sec or seg of {sections_to_update[0]}. May need to insert mechanism")
#
#                # Check if values contain a dict for distance-based assignment
#                if isinstance(values, dict):
#                    for distance_condition, assignment_value in values.items():
#                        dist_limit = int(distance_condition[1:])  # Extract the numeric value
#                        
#                        if sec_or_seg_att == 'seg':
#                            for section in sections_to_update:
#                                for seg in section:
#                                    seg_distance = h.distance(seg.x, sec=section)
#                                    if (distance_condition.startswith("<") and seg_distance < dist_limit) or \
#                                    (distance_condition.startswith(">=") and seg_distance >= dist_limit):
#                                        obj = seg
#                                        for att in att_and_sub_atts[:-1]:
#                                            obj = getattr(obj, att)
#                                        setattr(obj, att_and_sub_atts[-1], assignment_value)
#                        
#                        elif sec_or_seg_att == 'sec':
#                            for section in sections_to_update:
#                                sec_distance = h.distance(0.5, sec=section)
#                                if (distance_condition.startswith("<") and sec_distance < dist_limit) or \
#                                (distance_condition.startswith(">=") and sec_distance >= dist_limit):
#                                    obj = section
#                                    for att in att_and_sub_atts[:-1]:
#                                        obj = getattr(obj, att)
#                                    setattr(obj, att_and_sub_atts[-1], assignment_value)
#                
#                else:  # Handle the case where the value is not a dict
#                    assignment_value = values
#
#                    if sec_or_seg_att == 'seg':
#                        for section in sections_to_update:
#                            for seg in section:
#                                obj = seg
#                                for att in att_and_sub_atts[:-1]:
#                                    obj = getattr(obj, att)
#                                setattr(obj, att_and_sub_atts[-1], assignment_value)
#
#                    elif sec_or_seg_att == 'sec':
#                        for section in sections_to_update:
#                            obj = section
#                            for att in att_and_sub_atts[:-1]:
#                                obj = getattr(obj, att)
#                            setattr(obj, att_and_sub_atts[-1], assignment_value)

# Below may have correct spacing ( commented because inconsistent use of tabs and spaces)
#	#TODO                                           
#	def update_cell_parameters_from_dict(self, cell, update_dict): # update_dict should come from parameters.py
#		if self.is_indexable(cell.soma):
#			h.distance(sec=cell.soma[0])
#		else:
#			h.distance(sec=cell.soma)
#							
#		for sec_type in update_dict.keys():
#			sections_to_update = getattr(cell, sec_type)
#
#			if not self.is_indexable(sections_to_update):
#				sections_to_update = [sections_to_update]
#
#			for attribute_to_update, values in update_dict[sec_type].items():
#				att_and_sub_atts = attribute_to_update.split('.')
#				initial_att = att_and_sub_atts[0]
#				sec_or_seg_att = None  # Identify whether this attribute will be in segments or sections
#				
#				if hasattr(sections_to_update[0], initial_att):
#					sec_or_seg_att = 'sec'
#				elif hasattr(sections_to_update[0](0.5), initial_att):
#					sec_or_seg_att = 'seg'
#				else:
#					raise AttributeError(f"{initial_att} of {att_and_sub_atts} is not found in either sec or seg of {sections_to_update[0]}. May need to insert mechanism")
#
#				# Check if values contain a dict for distance-based assignment
#				if isinstance(values, dict):
#					for distance_condition, assignment_value in values.items():
#						dist_limit = int(distance_condition[1:])  # Extract the numeric value
#
#						if sec_or_seg_att == 'seg':
#							for section in sections_to_update:
#								for seg in section:
#									seg_distance = h.distance(seg.x, sec=section)
#									if (distance_condition.startswith("<") and seg_distance < dist_limit) or \
#										(distance_condition.startswith(">=") and seg_distance >= dist_limit):
#										obj = seg
#										for att in att_and_sub_atts:
#											if hasattr(obj, att):
#												obj = getattr(obj, att)
#											else:
#												raise AttributeError(f"Failed to access {att} in segment")
#										obj = assignment_value
#						elif sec_or_seg_att == 'sec':  # Assuming you want similar logic for sections
#							for section in sections_to_update:
#								sec_distance = h.distance(0.5, sec=section)
#								if (distance_condition.startswith("<") and sec_distance < dist_limit) or \
#									(distance_condition.startswith(">=") and sec_distance >= dist_limit):
#									obj = section
#									for att in att_and_sub_atts:
#										if hasattr(obj, att):
#											obj = getattr(obj, att)
#										else:
#											raise AttributeError(f"Failed to access {att} in section")
#									obj = assignment_value
#				else:  # Handle the case where the value is not a dict
#					assignment_value = values
#
#					if sec_or_seg_att == 'seg':
#						for section in sections_to_update:
#							for seg in section:
#								obj = seg
#								for att in att_and_sub_atts:
#									if hasattr(obj, att):
#										obj = getattr(obj, att)
#									else:
#										raise AttributeError(f"Failed to access {att} in segment")
#								obj = assignment_value
#					elif sec_or_seg_att == 'sec':
#						for section in sections_to_update:
#							obj = section
#							for att in att_and_sub_atts:
#								if hasattr(obj, att):
#									obj = getattr(obj, att)
#								else:
#									raise AttributeError(f"Failed to access {att} in section")
#							obj = assignment_value
