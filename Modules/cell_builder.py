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

def norm_dist(gmax_mean, gmax_std, size, clip): # inh
  val = np.random.normal(gmax_mean, gmax_std, size)
  s = np.clip(val, clip[0], clip[1])
  return s


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
    
		# replace dendrite with current injection
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
      
		self.logger.log("Finished creating a CellModel object.")

		# Add current 
		if self.parameters.CI_on:
			self.logger.log("Adding current injection.")
			cell.set_injection(
				amp = self.parameters.h_i_amplitude,
				dur = self.parameters.h_i_duration, 
				delay = self.parameters.h_i_delay,
        target = self.parameters.CI_target)
        

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

		# # increase nseg for clustering segments for clustering synapses Not needed with template that gives very high segment resolution.
		# # increase segment resolution makes more even cluster bounds
		# all_nseg = []
		# for sec in cell.all:
		# 	nseg = sec.nseg
		# 	all_nseg.append(nseg)
		# 	sec.nseg = 1+2*int(sec.L/10)
  
		# print(f"soma segments:{cell.get_segments_without_data(['soma'])}")
		# craete synapse objects
		self.logger.log("Building excitatory synapses.")
		self.build_exc_synapses(cell = cell)

		self.logger.log("Building inhibitory synapses.")
		self.build_inh_synapses(cell = cell)

		# self.logger.log("Building soma synapses.") # merged into build_inh_synapses
		# self.build_soma_synapses(cell = cell)

		# Assign spike trains
		self.logger.log("Assigning excitatory spike trains.")
		self.assign_excitatory_spike_trains(cell = cell, random_state = random_state)
  
		# calc exc for delayed inhibition
		exc_spike_trains = [syn.pc.spike_train for syn in cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"])]
		self.logger.log(f"{len(exc_spike_trains)} exc spikes trains")

		exc_mean_frs = [syn.pc.mean_fr for syn in cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"])]
		# print(f"exc_mean_frs: {exc_mean_frs}")

		self.logger.log("Assigning soma spike trains.")
		self.assign_soma_spike_trains(cell = cell, random_state = random_state, exc_spike_trains=exc_spike_trains)

		self.logger.log("Assigning inhibitory spike trains.")
		self.assign_inhibitory_spike_trains(cell = cell, random_state = random_state, exc_spike_trains=exc_spike_trains)

		self.logger.log(f"Total number of synapses: {len(cell.synapses)}")

		# synapses_without_spike_train = [syn for syn in cell.synapses if len(syn.netcons) == 0]
		# if len(synapses_without_spike_train) > 0:
		# 	raise ValueError(f"spike trains not assigned to synapses: {np.unique([syn.name for syn in synapses_without_spike_train])}")
		# else:
		# 	self.logger.log("All spike trains assigned.")

		self.logger.log(f"Built synapse types: {np.unique([syn.name for syn in cell.synapses])}")

		# Check for synapses missing netcons
		names_no_spike_train = [syn.name for syn in cell.synapses if len(syn.netcons) == 0]
		if names_no_spike_train:
			self.logger.log(
				f"Spike trains not assigned to synapses: "
				f"{ {name: names_no_spike_train.count(name) for name in set(names_no_spike_train)} }\n"
				f"Unique names: {np.unique(names_no_spike_train)}"
			)
		else:
			self.logger.log("All spike trains assigned.")

		# Check for synapses missing a presynaptic cell
		names_no_presyn = [syn.name for syn in cell.synapses if syn.pc is None]
		if names_no_presyn:
			self.logger.log(
				f"Synapses without presynaptic cell found: "
				f"{ {name: names_no_presyn.count(name) for name in set(names_no_presyn)} }\n"
				f"Unique names: {np.unique(names_no_presyn)}"
			)
		else:
			self.logger.log("All synapses have a presynaptic cell.")

		if names_no_spike_train:
			segments_no_spike_train = [syn.h_syn.get_segment() for syn in cell.synapses if len(syn.netcons) == 0]
			self.logger.log(
				f"Segments of synapses without spike trains: "
				f"{ {name: segments_no_spike_train.count(name) for name in set(segments_no_spike_train)} }\n"
				f"Unique names: {np.unique(segments_no_spike_train)}"
			)

  
		# record spike trains
		if self.parameters.record_spike_trains:
			spike_train_data = {
				'exc_spike_trains': exc_spike_trains,
				'soma_spike_trains': [syn.pc.spike_train for syn in cell.get_synapses(['inh_perisomatic'])],
				'inh_spike_trains': [syn.pc.spike_train for syn in cell.get_synapses(['inh_distal_basal','inh_distal_apic'])]
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
			inh_synapses = cell.get_synapses(['inh', 'inh_distal_basal', 'inh_distal_apic'])
			# exc_synapses = cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"], all_with_prefix=True)
			exc_synapses = cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"])
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
		self.logger.log(f"exc_mean_frs result distribution {np.mean(exc_mean_frs):.2f}, {np.std(exc_mean_frs):.2f}")

		#@CHECKING PCs
		# Extract synaptic cells
		# exc_pcs = [syn.pc for syn in cell.get_synapses(['exc_distal_basal', 'exc_oblique', 'exc_trunk', 'exc_tuft'])]
		# inh_pcs = [syn.pc for syn in cell.get_synapses(['inh_distal_basal', 'inh_distal_apic']) if syn.h_syn.get_segment() in cell.get_segments_without_data(['dend', 'apic'])]
		# soma_pcs = [syn.pc for syn in cell.get_synapses(['inh_perisomatic']) if syn.h_syn.get_segment() in cell.get_segments_of_type('perisomatic')]
		exc_pcs = [syn.pc for syn in cell.get_synapses([f"exc_{sec_type}" for sec_type in self.parameters.exc_syn_properties.keys()])]
		inh_pcs = [syn.pc for syn in cell.get_synapses([f"inh_{sec_type}" for sec_type in self.parameters.inh_syn_properties.keys() if sec_type != 'perisomatic']) if (syn.h_syn.get_segment() in cell.get_segments_without_data(['dend', 'apic']))]
		soma_pcs = [syn.pc for syn in cell.get_synapses(['inh_perisomatic']) if syn.h_syn.get_segment() in cell.get_segments_of_type('perisomatic')]

		# Extract unique pcs based on names
		exc_pcs_dict = {pc.name: pc for pc in exc_pcs}
		inh_pcs_dict = {pc.name: pc for pc in inh_pcs}
		soma_pcs_dict = {pc.name: pc for pc in soma_pcs}

		exc_pcs_uni = list(exc_pcs_dict.values())
		inh_pcs_uni = list(inh_pcs_dict.values())
		soma_pcs_uni = list(soma_pcs_dict.values())

		# Get counts
		exc_pc_count = len(exc_pcs_uni)
		inh_pc_count = len(inh_pcs_uni)
		soma_pc_count = len(soma_pcs_uni)

		# Calculate synapses per unique pc
		exc_synapses_per_pc = [exc_pcs.count(pc) for pc in exc_pcs_uni]
		inh_synapses_per_pc = [inh_pcs.count(pc) for pc in inh_pcs_uni]
		soma_synapses_per_pc = [soma_pcs.count(pc) for pc in soma_pcs_uni]

		# Print results
		self.logger.log(f"number of EXC pcs: {exc_pc_count} mean/std number of synapses per pc: {np.mean(exc_synapses_per_pc):.2f}, {np.std(exc_synapses_per_pc):.2f}")
		self.logger.log(f"number of INH pcs: {inh_pc_count} mean/std number of synapses per pc: {np.mean(inh_synapses_per_pc):.2f}, {np.std(inh_synapses_per_pc):.2f}")
		self.logger.log(f"number of SOMA pcs: {soma_pc_count} mean/std number of synapses per pc: {np.mean(soma_synapses_per_pc):.2f}, {np.std(soma_synapses_per_pc):.2f}")

		# calculate the mean fr distribution
		exc_mean_frs = [pc.mean_fr for pc in exc_pcs_uni]
		inh_mean_frs = [pc.mean_fr for pc in inh_pcs_uni]
		soma_mean_frs = [pc.mean_fr for pc in soma_pcs_uni]

		# Print results
		self.logger.log(f"EXC mean fr distribution: {np.mean(exc_mean_frs):.2f}, {np.std(exc_mean_frs):.2f}")
		self.logger.log(f"INH mean fr distribution: {np.mean(inh_mean_frs):.2f}, {np.std(inh_mean_frs):.2f}")
		self.logger.log(f"SOMA mean fr distribution: {np.mean(soma_mean_frs):.2f}, {np.std(soma_mean_frs):.2f}")

		# # change nseg back
		# for i, sec in enumerate(cell.all):
		# 	sec.nseg = all_nseg[i]
  
	def assign_soma_spike_trains(self, cell, random_state, exc_spike_trains) -> None: #@MARK CHECK: merging with assign_inhibitory_spike_trains

		# Proximal inh mean_fr distribution
		mean_fr, std_fr = self.parameters.inh_prox_mean_fr, self.parameters.inh_prox_std_fr
		a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
		proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

		soma_fgs = PCBuilder.assign_presynaptic_cells(
		cell = cell,
		n_func_gr = self.parameters.soma_n_fun_gr,
		n_pc_per_fg = self.parameters.soma_n_pc_per_fg,
		synapse_names = ["inh_perisomatic"],
		seg_names = ["soma"]
		) #5,20
		firing_rates = PoissonTrainGenerator.generate_lambdas_by_delaying(self.parameters.h_tstop, exc_spike_trains)
		for fg in soma_fgs: # one fr profile per fg
			# In this case the firing rate profile is the average exc spike train delayed. All functional groups would have the same, unless we subset by nearby exc spike train only
			for pc in fg.presynaptic_cells: # one spike train per pc
				mean_fr = proximal_inh_dist(size = 1)
				pc_firing_rates = PoissonTrainGenerator.shift_mean_of_lambdas(firing_rates, desired_mean=mean_fr, logger=self.logger)#, divide_1000=True)
				pc_firing_rates = PoissonTrainGenerator.rhythmic_modulation(pc_firing_rates, self.parameters.rhyth_frequency_inh_perisomatic, self.parameters.rhyth_depth_inh_perisomatic, self.parameters.h_dt)
				spike_train = PoissonTrainGenerator.generate_spike_train(
				lambdas = pc_firing_rates, 
				random_state = random_state)
				pc.set_spike_train(spike_train.mean_fr, spike_train.spike_times)
		for syn in cell.get_synapses(["inh_perisomatic"]):
			if syn.h_syn.get_segment() in cell.get_segments_without_data(["soma"]):
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
			synapse_names = [f"inh_{sec_type}" for sec_type in self.parameters.inh_syn_properties.keys()],#["inh_perisomatic", "inh_distal_basal", "inh_distal_apic"],
			seg_names = ["dend", "apic"]
		)
		for fg in inh_fgs: # one fr profile per fg
			firing_rates = PoissonTrainGenerator.generate_lambdas_by_delaying(self.parameters.h_tstop, exc_spike_trains)
			for pc in fg.presynaptic_cells: # one spike train per pc
				if np.linalg.norm(soma_coords - pc.cluster_center) < 100:
					mean_fr = proximal_inh_dist(size = 1)
					rhyth_mod_depth_to_use = self.parameters.rhyth_depth_inh_perisomatic
					rhyth_mod_freq_to_use = self.parameters.rhyth_frequency_inh_perisomatic
				else:
					mean_fr = distal_inh_dist(size = 1)
					rhyth_mod_depth_to_use = self.parameters.rhyth_depth_inh_distal
					rhyth_mod_freq_to_use = self.parameters.rhyth_frequency_inh_distal
				firing_rates = PoissonTrainGenerator.shift_mean_of_lambdas(firing_rates, desired_mean=mean_fr, logger=self.logger)#, divide_1000=True)
				firing_rates = PoissonTrainGenerator.rhythmic_modulation(firing_rates, rhyth_mod_freq_to_use, rhyth_mod_depth_to_use, self.parameters.h_dt)
				# print(f"firing_rates: {firing_rates}")
				spike_train = PoissonTrainGenerator.generate_spike_train(
				lambdas = firing_rates, 
				random_state = random_state)
				pc.set_spike_train(spike_train.mean_fr, spike_train.spike_times)

		for syn in cell.get_synapses([f"inh_{sec_type}" for sec_type in self.parameters.inh_syn_properties.keys()]):
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
			synapse_names = [f'exc_{sec_type}' for sec_type in self.parameters.exc_syn_properties.keys()],#["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique", "exc_distal_basal", "exc_distal_apic"], # probably only need last 2. can check build_exc_synapses.
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
				firing_rates = PoissonTrainGenerator.shift_mean_of_lambdas(lambdas=firing_rates, desired_mean=lambda_mean_fr, logger=self.logger)
				spike_train = PoissonTrainGenerator.generate_spike_train(
				lambdas = firing_rates, 
				random_state = random_state)
				# print(spike_train.spike_times)
				pc.set_spike_train(spike_train.mean_fr, spike_train.spike_times)

		for syn in cell.get_synapses([f'exc_{sec_type}' for sec_type in self.parameters.exc_syn_properties.keys()]):#["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique", "exc_distal_basal", "exc_distal_apic"]):
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

		for sec_type in sec_type_to_get: # gather the segments we want to distribute synapses over
			segs = cell.get_segments_of_type(sec_type)
			segments.extend(segs)
			if self.parameters.use_SA_probs:
				segment_probs.extend([np.pi * seg.diam * (seg.sec.L / seg.sec.nseg) for seg in segs])
			else:
				segment_probs.extend([seg.sec.L / seg.sec.nseg for seg in segs])

		# optionally exclude segments close to the soma (for exc) (get_segments_of_type(sec_type) is probably already doing this. Would need to check.)
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
		else:
			# calculate the PROPORTIONAL number of synapses to use if a whole-cell number is provided
			# gather the total length or surface area of the segments in case we want to distribute synapses proportionally
			if self.use_SA_probs: # probably better to calculate this only once
				total_length = sum([seg.membrane_surface_area for seg in cell.get_segments(['all'])])
			else:
				total_length = sum([seg.sec.L for seg in cell.get_segments(['all'])])
			nsyn = int(syn_number * sum(segment_probs) / total_length) if total_length > 0 else ValueError("Total length is zero.")
			self.logger.log(f"total synapses for {name}: {nsyn}")
			# print(f"syn_number: {syn_number} \n segment_probs: {segment_probs} \n total_length: {total_length}")

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
		
	def build_inh_synapses(self, cell):
		for sec_type in self.parameters.inh_syn_properties.keys():
			syn_props = self.parameters.inh_syn_properties[sec_type]
			self.build_synapses_with_specs(
				cell=cell,
				sec_type_to_get=sec_type,
				synapse_type='inh',
				use_density=self.parameters.inh_use_density,
				synaptic_density=syn_props['syn_density'] if self.parameters.inh_use_density else None,
				syn_number=syn_props['syn_number'] if not self.parameters.inh_use_density else None,
				gmax_dist_params={
					'dist_func': norm_dist,
					'params': {
						'gmax_mean': syn_props['gmax_params']['mean'],
						'gmax_std': syn_props['gmax_params']['std'],
						'clip': (0, 10*syn_props['gmax_params']['mean']) # clip between 0 and 10 times the mean
					}
				},
				P_release_params={
					'dist_func': P_release_dist,
					'params': {
						'P_mean': syn_props['P_release_params']['mean'],
						'P_std': syn_props['P_release_params']['std']
					}
				},
				name=f"inh_{sec_type}"
			)

	def build_exc_synapses(self, cell, exclude_within: float = None):#, sec_type_to_get: Union[str, List[str]], gmax_dist_params: dict, exclude_within: float = None):
		"""
		Builds excitatory synapses for specified segment types.

		Parameters:
			cell: The cell object on which synapses will be built.
			sec_type_to_get: The segment type(s) to retrieve (e.g., 'apic', 'dend'). Can be a string or list of strings.
			gmax_dist_params: Parameters for gmax distribution (mean, std, etc.).
			exclude_within: Distance from soma to exclude segments (optional).
		"""
		for sec_type in self.parameters.exc_syn_properties.keys():
			syn_props = self.parameters.exc_syn_properties[sec_type]
			self.build_synapses_with_specs(
				cell=cell,
				sec_type_to_get=sec_type,
				synapse_type='exc',
				use_density=self.parameters.exc_use_density,
				synaptic_density=syn_props['syn_density'] if self.parameters.exc_use_density else None,
				syn_number=syn_props['syn_number'] if not self.parameters.exc_use_density else None,
				gmax_dist_params={
					'dist_func': binned_log_norm_dist if self.parameters.bin_exc_gmax else log_norm_dist,
					'params': {
						'gmax_mean': syn_props['gmax_params']['mean'],
						'gmax_std': syn_props['gmax_params']['std'],
						'gmax_scalar': syn_props['gmax_params']['scalar'],
						'clip': syn_props['gmax_params']['clip']
					}
				},
				P_release_params={
					'dist_func': P_release_dist,
					'params': {
						'P_mean': self.parameters.exc_P_release_mean,#syn_props['P_release_params']['mean'],
						'P_std': self.parameters.exc_P_release_std#syn_props['P_release_params']['std']
					}
				},
				name=f"exc_{sec_type}",
				exclude_within=exclude_within
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