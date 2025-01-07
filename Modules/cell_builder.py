from enum import Enum
import pickle
import os
import numpy as np
from functools import partial
from typing import List, Union
import scipy.stats as st

from neuron import h

from logger import Logger
from spike_generator import PoissonTrainGenerator
from constants import SimulationParameters
from cell_model import CellModel
from presynaptic import PCBuilder
from reduction import Reductor
# from morphology_manipulator import MorphologyManipulator
import pandas as pd
import time

from electrotonic_distance import *
from surface_area import *

from Modules.morph_reduction_utils import get_reduced_cell, replace_dend_with_CI

from stylized_module import Builder

import h5py

#from reduction_utils import update_model_nseg_using_lambda, merge_synapses

class SkeletonCell(Enum):

	def __eq__(self, other):
		if type(self).__qualname__ != type(other).__qualname__: 
			return NotImplemented
		return self.name == other.name and self.value == other.value
	
	Hay = {
		"biophys": "L5PCbiophys3.hoc",#"L5PCbiophys3ActiveBasal.hoc",
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

def precompute_bin_means(gmax_mean, gmax_std, gmax_scalar, clip, large_sample_size=10000):
    # Generate a large number of log-normal distributed values
    val = np.random.lognormal(gmax_mean, gmax_std, large_sample_size)
    s = gmax_scalar * np.clip(val, clip[0], clip[1])

    # Determine bins and compute the mean for each bin
    num_bins = 10
    bin_edges = np.percentile(s, np.linspace(0, 100, num_bins + 1))
    bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]

    return bin_means

def binned_log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size, clip, bin_means):
    # Generate log-normal distributed values
    val = np.random.lognormal(gmax_mean, gmax_std, size)
    # Clip the values
    s = gmax_scalar * np.clip(val, clip[0], clip[1])
    # Assign each value to the nearest bin mean
    binned_values = np.zeros_like(s)
    for i in range(size):
        # Find the bin the value belongs to
        bin_index = np.digitize(s[i], bin_means) - 1
        # Assign the value to the bin mean
        binned_values[i] = bin_means[bin_index]
    return binned_values

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
	stylized_templates_folder = "../cells/stylized_morphologies"

	def __init__(self, cell_type: SkeletonCell, parameters: SimulationParameters, logger: Logger) -> None:

		self.cell_type = cell_type
		self.parameters = parameters
		self.logger = logger

	def build_cell(self):
		start_time = time.time()
		random_state = np.random.RandomState(self.parameters.numpy_random_state)
		np.random.seed(self.parameters.numpy_random_state)
		neuron_r = h.Random()
		neuron_r.MCellRan4(self.parameters.neuron_random_state)

		# Build skeleton cell
		self.logger.log(f"Building {self.cell_type}.")
   
		if self.parameters.build_stylized:
			skeleton_cell = self.build_stylized_cell()

		elif self.cell_type == SkeletonCell.Hay:
			skeleton_cell = self.build_Hay_cell()

		elif self.cell_type == SkeletonCell.HayNeymotin:
			skeleton_cell = self.build_HayNeymotin_cell()

		elif self.cell_type == SkeletonCell.NeymotinDetailed:
			skeleton_cell = self.build_Neymotin_detailed_cell()

		cell = CellModel(skeleton_cell, random_state, neuron_r, self.logger)

		# @DEPRACATING debugging
		# for model_part in ['all','soma','dend','apic','axon']:
		# 		print(f"{model_part}: {getattr(cell, model_part)}")
   
        
    # ----
    	# Build synapses @deprecating, neuron_reduce/cable_expander reduction
		# if not self.parameters.all_synapses_off:
		# 		self.build_synapses(cell, random_state)
		# if self.parameters.reduce_cell_NRCE: # @deprecating, neuron_reduce/cable_expander reduction
		# 		reductor = Reductor(logger = self.logger)
		# 		cell = self.perform_reduction(reductor = reductor, cell = cell, random_state = random_state)
  
	# Build synapses & reduce cell
		if self.parameters.synapse_mapping:
			self.build_synapses(cell, random_state)
			if self.parameters.reduce_apic or self.parameters.reduce_basals or self.parameters.reduce_obliques:
				cell, original_seg_data, all_deleted_seg_indices = get_reduced_cell(self, reduce_tufts = self.parameters.reduce_tufts, 
							reduce_basals = self.parameters.reduce_basals,
							reduce_obliques = self.parameters.reduce_obliques, 
							reduce_apic=self.parameters.reduce_apic,
							cell = cell)
		else:
			if self.parameters.reduce_apic or self.parameters.reduce_basals or self.parameters.reduce_obliques:
					cell, original_seg_data, all_deleted_seg_indices = get_reduced_cell(self, reduce_tufts = self.parameters.reduce_tufts, 
							reduce_basals = self.parameters.reduce_basals,
							reduce_obliques = self.parameters.reduce_obliques,
							reduce_apic=self.parameters.reduce_apic,
							cell = cell)
					self.build_synapses(cell, random_state)
			else:
				self.build_synapses(cell, random_state)
    
		# replace with current injection
		replace_start_time = time.time()
		if (self.parameters.num_basal_to_replace_with_CI + self.parameters.num_tuft_to_replace_with_CI) > 0:
			cell = replace_dend_with_CI(cell, self.parameters)
		replace_end_time = time.time()
		total_replace_time = replace_end_time - replace_start_time
		replace_file_path = os.path.join(self.parameters.path, "replace_runtime.txt")
		with open(replace_file_path, "w") as replace_file:
			replace_file.write(f"{total_replace_time:.3f} seconds")
		
		# merge synapses/optimize nseg by lambda
		reductor = Reductor(logger = self.logger)
		if self.parameters.optimize_nseg_by_lambda:
				self.logger.log("Updating nseg using lambda.")
				reductor.update_model_nseg_using_lambda(cell, segs_per_lambda=self.parameters.segs_per_lambda)
		if self.parameters.merge_synapses:
				self.logger.log("Merging synapses.")
				reductor.merge_synapses(cell)

		# set v_init for all compartments
		h.v_init = self.parameters.h_v_init
		h.finitialize(h.v_init)

    #---
      
		self.logger.log("Finished creating a CellModel object.")

		# @CHECK ---- @MARK remove @KEEP as reference for controlling regions of active synapses?
		# Turn off certain presynaptic neurons to simulate in vivo
#		if (self.parameters.CI_on == False) and (self.parameters.trunk_exc_synapses == False):
#			for synapse in cell.synapses:
#				if (
#					(synapse.h_syn.get_segment().sec in cell.apic) and 
#					(synapse.syn_mod in self.parameters.exc_syn_mod) and 
#					(synapse.h_syng.get_segment().sec in cell.get_tufts_obliques()[1] == False) and 
#					(synapse.h_syn.get_segment().sec.y3d(0) < 600)):
#					for netcon in synapse.netcons: netcon.active(False)
#		
#		# Turn off perisomatic exc neurons
#		if (self.parameters.perisomatic_exc_synapses == False):
#			for synapse in cell.synapses:
#				if (
#					(h.distance(synapse.h_syn.get_segment(), cell.soma[0](0.5)) < 75) and 
#					(synapse.syn_mod in self.parameters.exc_syn_mod)):
#					for netcon in synapse.netcons: netcon.active(False)
   
		# ----

		# Add current 
		if self.parameters.CI_on:
			cell.set_injection(
				amp = self.parameters.h_i_amplitude,
				dur = self.parameters.h_i_duration, 
				delay = self.parameters.h_i_delay,
        target = self.parameters.CI_target)
        
 		# ----

    # report runtime
		end_time = time.time()
		run_time = end_time - start_time
		self.logger.log(f"Finish building in {run_time}")
    # Record the  runtime to a file
		runtime_file_path = os.path.join(self.parameters.path, "builder_runtime.txt")
		with open(runtime_file_path, "w") as runtime_file:
				runtime_file.write(f"{run_time} seconds")
        
		return cell, skeleton_cell

	def build_synapses(self, cell, random_state):
		if (self.parameters.all_synapses_off):
			self.logger.log("Not building synapses.")
			return None

		# # increase nseg for clustering segments for clustering synapses
		# all_nseg = []
		# for sec in cell.all:
		# 	nseg = sec.nseg
		# 	all_nseg.append(nseg)
		# 	sec.nseg = 1+2*int(sec.L/10)
  
		print(f"soma segments:{cell.get_segments_without_data(['soma'])}")
		# craete synapse objects
		self.logger.log("Building excitatory synapses.")
		self.build_exc_synapses_by_type(cell = cell)

		self.logger.log("Building inhibitory synapses.")
		self.build_inh_synapses(cell = cell)

		self.logger.log("Building soma synapses.")
		self.build_soma_synapses(cell = cell)

		# Assign spike trains
		self.logger.log("Assigning excitatory spike trains.")
		self.assign_excitatory_spike_trains(cell = cell, random_state = random_state)
  
		# calc exc for delayed inhibition
		exc_spike_trains = [syn.pc.spike_train for syn in cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"])]
     
		exc_mean_frs = [syn.pc.mean_fr for syn in cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"])]
		print(f"exc_mean_frs: {exc_mean_frs}")

		self.logger.log("Assigning inhibitory spike trains.")
		self.assign_inhibitory_spike_trains(cell = cell, random_state = random_state, exc_spike_trains=exc_spike_trains)

		self.logger.log("Assigning soma spike trains.")
		self.assign_soma_spike_trains(cell = cell, random_state = random_state, exc_spike_trains=exc_spike_trains)
  
		# record spike trains
		if self.parameters.record_spike_trains:
			spike_train_data = {
				'exc_spike_trains': exc_spike_trains,
				'soma_spike_trains': [syn.pc.spike_train for syn in cell.get_synapses(['soma_inh'])],
				'inh_spike_trains': [syn.pc.spike_train for syn in cell.get_synapses(['inh', 'inh_dend','inh_apic'])]
			}
			for dataset_name, data in spike_train_data.items():
				file_path = os.path.join(self.parameters.path, f'{dataset_name}.h5')
				with h5py.File(file_path, 'w') as h5f:
					for i, sequence in enumerate(data):
						h5f.create_dataset(f'spike_train_{i}', data=sequence)

		# Record synapse distributions
		if self.parameters.record_synapse_distributions:
			all_segments = cell.get_segments_without_data(['all'])
			soma_synapses = cell.get_synapses(['soma_inh'])
			inh_synapses = cell.get_synapses(['inh', 'inh_dend', 'inh_apic'])
			exc_synapses = cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"], all_with_prefix=True)
			synapse_data = {
				'synapse_type': (
					['soma_inh'] * len(soma_synapses) +
					['inh'] * len(inh_synapses) +
					['exc'] * len(exc_synapses)
				),
				'mean_firing_rate': (
					[syn.pc.mean_fr for syn in soma_synapses] +
					[syn.pc.mean_fr for syn in inh_synapses] +
					[syn.pc.mean_fr for syn in exc_synapses]
				),
				'weight': (
					[syn.gmax_val for syn in soma_synapses] +
					[syn.gmax_val for syn in inh_synapses] +
					[syn.gmax_val for syn in exc_synapses]
				),
				'seg_id': (
					[all_segments.index(syn.h_syn.get_segment()) for syn in soma_synapses] +
					[all_segments.index(syn.h_syn.get_segment()) for syn in inh_synapses] +
					[all_segments.index(syn.h_syn.get_segment()) for syn in exc_synapses]
				),
				'pc_name': (
					[syn.pc.name for syn in soma_synapses] +
					[syn.pc.name for syn in inh_synapses] +
					[syn.pc.name for syn in exc_synapses]
				)
			}
			# Save synapse data to file
			synapse_file_path = os.path.join(self.parameters.path, 'synapse_data.h5')
			with h5py.File(synapse_file_path, 'w') as h5f:
				for key, values in synapse_data.items():
					h5f.create_dataset(key, data=values)


		#@CHECKING resulting mean firing rate distribution
		# print(f"exc_mean_frs result distribution {np.mean(exc_mean_frs), np.std(exc_mean_frs)}")

		#@CHECKING PCs
		# Extract synaptic cells
		# exc_pcs = [syn.pc for syn in cell.get_synapses('exc')]
		# inh_pcs = [syn.pc for syn in cell.get_synapses('inh') if syn.h_syn.get_segment() in cell.get_segments_without_data(['dend', 'apic'])]
		# soma_pcs = [syn.pc for syn in cell.get_synapses('soma') if syn.h_syn.get_segment() in cell.get_segments_without_data(['soma'])]

		# # Extract unique pcs based on names
		# exc_pcs_dict = {pc.name: pc for pc in exc_pcs}
		# inh_pcs_dict = {pc.name: pc for pc in inh_pcs}
		# soma_pcs_dict = {pc.name: pc for pc in soma_pcs}

		# exc_pcs_uni = list(exc_pcs_dict.values())
		# inh_pcs_uni = list(inh_pcs_dict.values())
		# soma_pcs_uni = list(soma_pcs_dict.values())

		# # Get counts
		# exc_pc_count = len(exc_pcs_uni)
		# inh_pc_count = len(inh_pcs_uni)
		# soma_pc_count = len(soma_pcs_uni)

		# # Calculate synapses per unique pc
		# exc_synapses_per_pc = [exc_pcs.count(pc) for pc in exc_pcs_uni]
		# inh_synapses_per_pc = [inh_pcs.count(pc) for pc in inh_pcs_uni]
		# soma_synapses_per_pc = [soma_pcs.count(pc) for pc in soma_pcs_uni]

		# # Print results
		# print(f"number of EXC pcs: {exc_pc_count} mean/std number of synapses per pc: {np.mean(exc_synapses_per_pc)}, {np.std(exc_synapses_per_pc)}")
		# print(f"number of INH pcs: {inh_pc_count} mean/std number of synapses per pc: {np.mean(inh_synapses_per_pc)}, {np.std(inh_synapses_per_pc)}")
		# print(f"number of SOMA pcs: {soma_pc_count} mean/std number of synapses per pc: {np.mean(soma_synapses_per_pc)}, {np.std(soma_synapses_per_pc)}")

		# # change nseg back
		# for i, sec in enumerate(cell.all):
		# 	sec.nseg = all_nseg[i]
  
	def assign_soma_spike_trains(self, cell, random_state, exc_spike_trains) -> None:

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		soma_fgs = PCBuilder.assign_presynaptic_cells(
		cell = cell,
		n_func_gr = self.parameters.soma_n_fun_gr,
		n_pc_per_fg = self.parameters.soma_n_pc_per_fg,
		synapse_names = ["soma_inh"],
		seg_names = ["soma"]
		) #5,20
		firing_rates = PoissonTrainGenerator.generate_lambdas_by_delaying(self.parameters.h_tstop, exc_spike_trains)
		for fg in soma_fgs: # one fr profile per fg
			# In this case the firing rate profile is the average exc spike train delayed. All functional groups would have the same, unless we subset by nearby exc spike train only
			for pc in fg.presynaptic_cells: # one spike train per pc
				mean_fr = proximal_inh_dist(size = 1)
				pc_firing_rates = PoissonTrainGenerator.shift_mean_of_lambdas(firing_rates, desired_mean=mean_fr)#, divide_1000=True)
				spike_train = PoissonTrainGenerator.generate_spike_train(
				lambdas = pc_firing_rates, 
				random_state = random_state)
				pc.set_spike_train(spike_train.mean_fr, spike_train.spike_times)
		for syn in cell.get_synapses(["soma_inh"]):
			if syn.h_syn.get_segment() in cell.get_segments_without_data(["soma_inh"]):
				syn.set_spike_train_from_pc()


	def assign_inhibitory_spike_trains(self, cell, random_state, exc_spike_trains) -> None:

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		# Distal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_distal_mean_fr, self.parameters.inh_distal_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		soma_coords = cell.get_segments(["soma"])[1][0].coords[["pc_0", "pc_1", "pc_2"]].to_numpy()
  
		inh_fgs = PCBuilder.assign_presynaptic_cells(
			cell = cell,
			n_func_gr = self.parameters.inh_n_FuncGroups,
			n_pc_per_fg = self.parameters.inh_n_PreCells_per_FuncGroup,
			synapse_names = ["inh", "inh_dend", "inh_apic"],
			seg_names = ["dend", "apic"]
		)
		for fg in inh_fgs: # one fr profile per fg
			firing_rates = PoissonTrainGenerator.generate_lambdas_by_delaying(self.parameters.h_tstop, exc_spike_trains)
			for pc in fg.presynaptic_cells: # one spike train per pc
				if np.linalg.norm(soma_coords - pc.cluster_center) < 100:
					mean_fr = proximal_inh_dist(size = 1)
				else:
					mean_fr = distal_inh_dist(size = 1)
				firing_rates = PoissonTrainGenerator.shift_mean_of_lambdas(firing_rates, desired_mean=mean_fr)#, divide_1000=True)
				print(f"firing_rates: {firing_rates}")
				spike_train = PoissonTrainGenerator.generate_spike_train(
				lambdas = firing_rates, 
				random_state = random_state)
				pc.set_spike_train(spike_train.mean_fr, spike_train.spike_times)

		for syn in cell.get_synapses(["inh", "inh_dend", "inh_apic"]):
				if syn.h_syn.get_segment() in cell.get_segments_without_data(["dend", "apic"]):
					syn.set_spike_train_from_pc()

	def assign_excitatory_spike_trains(self, cell, random_state) -> None:

		exc_spike_trains = []
		exc_mean_frs = []

		# Distribution of mean firing rates
		# mean_fr_dist = partial(gamma_dist, mean = self.parameters.exc_mean_fr, size = 1)
		mean_fr, std_fr = self.parameters.exc_mean_fr, self.parameters.exc_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		if self.parameters.use_levy_dist_for_exc:
			mean_fr_dist = partial(st.levy_stable.rvs, alpha=1.37, beta=-1.00, loc=0.92, scale=0.44, size=1)
		else:
			mean_fr_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		# if self.parameters.clustering: # note one segment belongs to one precell
		exc_fgs = PCBuilder.assign_presynaptic_cells(
			cell = cell,
			n_func_gr = self.parameters.exc_n_FuncGroups,
			n_pc_per_fg = self.parameters.exc_n_PreCells_per_FuncGroup,
			synapse_names = ["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"],
			seg_names = ["all"]
		)
		for fg in exc_fgs: # one fr profile per fg
			firing_rates = PoissonTrainGenerator.generate_lambdas_from_pink_noise(
					num = self.parameters.h_tstop,
					random_state = random_state)
			for pc in fg.presynaptic_cells: # one spike train per pc
				if self.parameters.exc_constant_fr:
					lambda_mean_fr = 0 + self.parameters.excFR_increase
				else:
					lambda_mean_fr = (mean_fr_dist(size = 1) + self.parameters.excFR_increase)
				firing_rates = PoissonTrainGenerator.shift_mean_of_lambdas(lambdas=firing_rates, desired_mean=lambda_mean_fr)
				spike_train = PoissonTrainGenerator.generate_spike_train(
				lambdas = firing_rates, 
				random_state = random_state)
				# print(spike_train.spike_times)
				pc.set_spike_train(spike_train.mean_fr, spike_train.spike_times)

		for syn in cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"]):
				exc_spike_trains.append(spike_train.spike_times)
				exc_mean_frs.append(spike_train.mean_fr)
				syn.set_spike_train_from_pc()
    
		return exc_spike_trains, exc_mean_frs

	def build_synapses_with_specs(self, 
					cell, 
					sec_type_to_get: Union[str, List[str]], 
					synapse_type: str, 
					use_density: bool,
					synaptic_density: float = None,
					syn_number: int = None,
					gmax_dist_params: dict = None,
					P_release_params: dict = None,
					name: str = None,
					exclude_within: float = None) -> None:
		"""
		Builds synapses of a given type on specified segments.

		Parameters:
			cell: The cell object on which synapses will be built.
			sec_type_to_get: The segment type(s) to retrieve ('dend', 'apic', etc.). Can be a string or list of strings.
			synapse_type: Type of synapse ('inh', 'exc', etc.).
			use_density: Whether to distribute synapses based on density or a fixed count.
			synaptic_density: Density of synapses per unit length (required if use_density is True).
			syn_number: Total number of synapses to distribute (required if use_density is False).
			gmax_dist_params: Parameters for gmax distribution (mean, std, etc.).
			P_release_params: Parameters for release probability distribution (mean, std, etc.).
			name: Name for the synapse type (e.g., 'soma', 'inh', 'exc').
			exclude_within: Distance from soma to exclude segments (optional).
		"""
		if gmax_dist_params is None or P_release_params is None:
			raise ValueError("Both gmax_dist_params and P_release_params must be provided.")

		# Create gmax and P_release distributions
		if gmax_dist_params['dist_func'] is not None:
			gmax_dist = partial(
				gmax_dist_params['dist_func'],
				**gmax_dist_params['params'],
				size=1
			)
		else:
			gmax_dist = gmax_dist_params['params']['gmax_mean']

		P_dist = partial(
			P_release_params['dist_func'],
			**P_release_params['params'],
			size=1
		)

		# Get segments of the specified type(s)
		if isinstance(sec_type_to_get, str):
			sec_type_to_get = [sec_type_to_get]

		segments = []
		segment_probs = []

		for sec_type in sec_type_to_get:
			segs = cell.get_segments_of_type(sec_type)
			segments.extend(segs)
			if self.parameters.use_SA_probs:
				segment_probs.extend([np.pi * seg.diam * (seg.sec.L / seg.sec.nseg) for seg in segs])
			else:
				segment_probs.extend([seg.sec.L / seg.sec.nseg for seg in segs])

		# Optionally exclude segments close to the soma
		if exclude_within is not None:
			to_remove = [
				i for i, seg in enumerate(segments)
				if self.h.distance(seg, cell.soma[0](0.5)) < exclude_within
			]
			segments = [seg for i, seg in enumerate(segments) if i not in to_remove]
			segment_probs = [prob for i, prob in enumerate(segment_probs) if i not in to_remove]

		# Calculate synapse count or density
		if use_density:
			nsyn = synaptic_density
		else: #@MARK in this case we should iterate through the segments of types sec_type_to_get if it is a list and calculate the proportional length/area for each type.
			NotImplementedError(f"When specifying total inh synapses we have to calc the number needed for apic, dend. This functionality was inadvertently removed when merging functions. See commit history.")
			total_length = sum(segment_probs)
			# print(f"syn_number: {syn_number} \n segment_probs: {segment_probs} \n total_length: {total_length}")
			nsyn = int(syn_number * sum(segment_probs) / total_length) if total_length > 0 else 0 
			# @ MARK originaly this was two iterations through apic,dend segment_probs to calculate the number for each

		# Add synapses to the cell
		cell.add_synapses_over_segments(
			segments=segments,
			nsyn=nsyn,
			syn_mod=self.parameters.inh_syn_mod if 'inh' in name else self.parameters.exc_syn_mod if 'exc' in name else NotImplementedError(f"'inh' or 'exc' should be in 'name'"),
			syn_params=self.parameters.inh_syn_params if 'inh' in name else self.parameters.exc_syn_params if 'exc' in name else NotImplementedError(f"'inh' or 'exc' should be in 'name'"),
			gmax=gmax_dist,
			name=name or synapse_type,
			density=use_density,
			seg_probs=segment_probs,
			release_p=P_dist
		)

	def build_soma_synapses(self, cell):
		self.build_synapses_with_specs(
		cell=cell,
		sec_type_to_get='soma',
		synapse_type='inh',
		use_density=False,
		syn_number=self.parameters.num_soma_inh_syns,
		gmax_dist_params={
			'dist_func': None,
			'params': {
				'gmax_mean': self.parameters.soma_gmax_dist,
				# 'gmax_std': self.parameters.soma_gmax_std,
				# 'clip': self.parameters.soma_gmax_clip
			}
		},
		P_release_params={
			'dist_func': P_release_dist,
			'params': {
				'P_mean': self.parameters.inh_soma_P_release_mean,
				'P_std': self.parameters.inh_soma_P_release_std
			}
		},
		name='soma_inh'
		)
		
	def build_inh_synapses(self, cell):
		# Calculate total length or surface area
		if self.parameters.use_SA_probs:
			total_length = sum(segment.membrane_surface_area for segment in cell.get_segments(['apic', 'dend'])[1])
		else:
			total_length = sum(segment.L for segment in cell.get_segments(['apic', 'dend'])[1])

		# Prepare lists for syn_numbers, gmax_means, and sec_types
		syn_numbers = [
			self.parameters.inh_syn_number * (
				sum(segment.membrane_surface_area for segment in cell.get_segments(['apic'])[1])
				if self.parameters.use_SA_probs else
				sum(segment.L for segment in cell.get_segments(['apic'])[1])
			) / total_length,
			self.parameters.inh_syn_number * (
				sum(segment.membrane_surface_area for segment in cell.get_segments(['dend'])[1])
				if self.parameters.use_SA_probs else
				sum(segment.L for segment in cell.get_segments(['dend'])[1])
			) / total_length
		]
		gmax_means = [self.parameters.apic_inh_gmax_dist, self.parameters.basal_inh_gmax_dist]
		sec_types = ['apic', 'dend']
		P_release_means = [self.parameters.inh_apic_P_release_mean, self.parameters.inh_basal_P_release_mean]
		P_release_stds = [self.parameters.inh_apic_P_release_std, self.parameters.inh_basal_P_release_std]

		# Iterate using zip
		for syn_number, gmax_mean, sec_type_to_get, P_release_mean, P_release_std in zip(syn_numbers, gmax_means, sec_types, P_release_means, P_release_stds):
			self.build_synapses_with_specs(
			cell=cell,
			sec_type_to_get=sec_type_to_get,
			synapse_type='inh',
			use_density=self.parameters.inh_use_density,
			synaptic_density=self.parameters.inh_synaptic_density,
			syn_number=syn_number,
			gmax_dist_params={
				'dist_func': None,
				'params': {
					'gmax_mean': gmax_mean, #@MARK what about parameters.inh_gmax_mean
					# 'gmax_std': self.parameters.inh_gmax_std,
					# 'clip': self.parameters.inh_gmax_clip
				}
			},
			P_release_params={
				'dist_func': P_release_dist,
				'params': {
					'P_mean': P_release_mean,
					'P_std': P_release_std
				}
			},
			name='inh'
			)

	def build_exc_synapses(self, cell, sec_type_to_get: Union[str, List[str]], gmax_dist_params: dict, exclude_within: float = None):
		"""
		Builds excitatory synapses for specified segment types.

		Parameters:
			cell: The cell object on which synapses will be built.
			sec_type_to_get: The segment type(s) to retrieve (e.g., 'apic', 'dend'). Can be a string or list of strings.
			gmax_dist_params: Parameters for gmax distribution (mean, std, etc.).
			exclude_within: Distance from soma to exclude segments (optional).
		"""
		self.build_synapses_with_specs(
			cell=cell,
			sec_type_to_get=sec_type_to_get,
			synapse_type='exc',
			use_density=self.parameters.exc_use_density,
			synaptic_density=self.parameters.exc_synaptic_density if self.parameters.exc_use_density else None,
			syn_number=self.parameters.exc_syn_number if not self.parameters.exc_use_density else None,
			gmax_dist_params=gmax_dist_params,
			P_release_params={
				'dist_func': P_release_dist,
				'params': {
					'P_mean': self.parameters.exc_P_release_mean,
					'P_std': self.parameters.exc_P_release_std
				}
			},
			name=f"exc_{sec_type_to_get}",
			exclude_within=exclude_within
		)

	def build_exc_synapses_by_type(self, cell):
		self.build_exc_synapses(cell, sec_type_to_get = "trunk", gmax_dist_params = {
        "dist_func": binned_log_norm_dist if self.parameters.bin_exc_gmax else log_norm_dist,
        "params": {
            "gmax_mean": self.parameters.trunk_exc_gmax_mean,
            "gmax_std": self.parameters.trunk_exc_gmax_std,
			'gmax_scalar': self.parameters.exc_scalar,
            "clip": self.parameters.exc_gmax_clip
        	}
		}
		)
		self.build_exc_synapses(cell, sec_type_to_get = "oblique", gmax_dist_params = {
        "dist_func": binned_log_norm_dist if self.parameters.bin_exc_gmax else log_norm_dist,
        "params": {
            "gmax_mean": self.parameters.oblique_exc_gmax_mean,
            "gmax_std": self.parameters.oblique_exc_gmax_std,
			'gmax_scalar': self.parameters.exc_scalar,
            "clip": self.parameters.exc_gmax_clip
        	}
		}
		)
		self.build_exc_synapses(cell, sec_type_to_get = "tuft", gmax_dist_params = {
        "dist_func": binned_log_norm_dist if self.parameters.bin_exc_gmax else log_norm_dist,
        "params": {
            "gmax_mean": self.parameters.tuft_exc_gmax_mean,
            "gmax_std": self.parameters.tuft_exc_gmax_std,
			'gmax_scalar': self.parameters.exc_scalar,
            "clip": self.parameters.exc_gmax_clip
        	}
		}
		)

	def build_stylized_cell(self) -> object:
		geometry_path = os.path.join(self.stylized_templates_folder, self.parameters.geometry_file)
		geo_standard = pd.read_csv(geometry_path,index_col='id')         
		builder = Builder(geo_standard)
		cell = builder.cells[0]
		return cell

	def build_Hay_cell(self) -> object:
		# Load biophysics
		h.load_file(os.path.join(self.templates_folder, self.parameters.Hay_biophys))#SkeletonCell.Hay.value["biophys"]))

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
  
	# @DEPRACATING neuron_reduce/cable_expander
	# def perform_reduction(self, reductor, cell, random_state):
	# 	if self.parameters.reduce_cell:
	# 			cell, nr_seg_to_seg = reductor.reduce_cell(
	# 					cell_model = cell,  
	# 					#random_state = random_state,
	# 					reduction_frequency = self.parameters.reduction_frequency)
	# 			if self.parameters.record_seg_to_seg and not self.parameters.expand_cable:
	# 							nr_seg_to_seg_df = pd.DataFrame(list(nr_seg_to_seg.items()), columns=['detailed', 'neuron_reduce'])
	# 							nr_seg_to_seg_df.to_csv(os.path.join(self.parameters.path, "nr_seg_to_seg.csv"))
	# 			if self.parameters.expand_cable:
	# 					cell, ce_seg_to_seg = reductor.expand_cell(
	# 							cell_model = cell, 
    #         		choose_branches = self.parameters.choose_branches, 
    #         		reduction_frequency = self.parameters.reduction_frequency, 
    #         		random_state = random_state)
	# 					if self.parameters.record_seg_to_seg:
	# 							ce_seg_to_seg_df = pd.DataFrame(list(ce_seg_to_seg.items()), columns=['neuron_reduce', 'cable_expander'])
	# 							ce_seg_to_seg_df.to_csv(os.path.join(self.parameters.path, "ce_seg_to_seg.csv"))
	# 			cell._assign_sec_coords(random_state)
            
	# 	elif self.parameters.expand_cable:
	# 			raise(ValueError("expand_cable cannot be True without reduce_cell being True"))
      
	# 	else: # call standalone reduction methods without NR or CE
	# 			if self.parameters.optimize_nseg_by_lambda:
	# 					self.logger.log("Updating nseg using lambda.")
	# 					reductor.update_model_nseg_using_lambda(cell)
	# 			if self.parameters.merge_synapses:
	# 					self.logger.log("Merging synapses.")
	# 					reductor.merge_synapses(cell)
	# 	return cell  
  
    # @DEPCRATING Useful for calculating surface area, length constants... 
    # def perform_MM(self, cell, MM): # need to separate recording nexus_seg_index from this and create constants to control
	# 	nexus_seg_index, SA_df, L_df, elec_L_of_tufts = MM.run(cell)
	# 	nexus_seg_index_file_path = os.path.join(self.parameters.path, "nexus_seg_index.txt")
	# 	with open(nexus_seg_index_file_path, "w") as nexus_seg_index_file:
	# 			nexus_seg_index_file.write(f"Nexus Seg Index: {nexus_seg_index}")
	# 	sa_df_to_save = pd.DataFrame(list(SA_df.items()), columns=['Model_Part', 'Surface_Area'])
	# 	sa_df_to_save.to_csv(os.path.join(self.parameters.path, "SA.csv"), index=False)
	# 	l_df_to_save = pd.DataFrame(list(L_df.items()), columns=['Model_Part', 'Length'])
	# 	l_df_to_save.to_csv(os.path.join(self.parameters.path, "L.csv"), index=False)
	# 	elec_L_of_tufts_file_path = os.path.join(self.parameters.path, "elec_L_of_tufts.txt")
	# 	with open(elec_L_of_tufts_file_path, "w") as elec_L_of_tufts_file:
	# 		elec_L_of_tufts_file.write(f"Tuft electrotonic lengths: {elec_L_of_tufts}")
	# 	if self.parameters.expand_cable:
	# 		MM.update_reduced_model_tuft_lengths(cell)
	# 		nexus_seg_index, SA_df, L_df, elec_L_of_tufts = MM.run(cell)
	# 		nexus_seg_index_file_path = os.path.join(self.parameters.path, "nexus_seg_index.txt")
	# 		with open(nexus_seg_index_file_path, "w") as nexus_seg_index_file:
	# 			nexus_seg_index_file.write(f"Nexus Seg Index: {nexus_seg_index}")
	# 		sa_df_to_save = pd.DataFrame(list(SA_df.items()), columns=['Model_Part', 'Surface_Area'])
	# 		sa_df_to_save.to_csv(os.path.join(self.parameters.path, "SA_after.csv"), index=False)
	# 		l_df_to_save = pd.DataFrame(list(L_df.items()), columns=['Model_Part', 'Length'])
	# 		l_df_to_save.to_csv(os.path.join(self.parameters.path, "L_after.csv"), index=False)
	# 		elec_L_of_tufts_file_path = os.path.join(self.parameters.path, "elec_L_of_tufts_after.txt")
	# 		with open(elec_L_of_tufts_file_path, "w") as elec_L_of_tufts_file:
	# 			elec_L_of_tufts_file.write(f"Tuft electrotonic lengths: {elec_L_of_tufts}")