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

from Modules.allen_interfacing import load_skeleton_cell_from_allen

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
	Allen = {
		"biophys": None,
		"morph": None,
		"template": None,
		"pickle": None,
		"directory": "../Allen/Cell_477127614",
		"modfiles": "../Allen/Cell_477127614/modfiles"
	}

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

		elif self.cell_type == SkeletonCell.Allen:
			skeleton_cell = self.build_Allen_cell()

		cell = CellModel(skeleton_cell, random_state, neuron_r, self.logger)    

		if self.parameters.reduce_apic or self.parameters.reduce_basals or self.parameters.reduce_obliques:
			cell, original_seg_data, all_deleted_seg_indices = get_reduced_cell(self, reduce_tufts = self.parameters.reduce_tufts, 
						reduce_basals = self.parameters.reduce_basals,
						reduce_obliques = self.parameters.reduce_obliques, 
						reduce_apic = self.parameters.reduce_apic,
						cell = cell)

    
		# replace dendrite with current injection
		replace_start_time = time.time()
		if (self.parameters.num_basal_to_replace_with_CI + self.parameters.num_tuft_to_replace_with_CI) > 0:
			cell = replace_dend_with_CI(cell, self.parameters)
		replace_end_time = time.time()
		total_replace_time = replace_end_time - replace_start_time
		self.logger.log_runtime("cell_builder", "replace_dend_with_CI", total_replace_time)
		
		# merge synapses/optimize nseg by lambda
		reductor = Reductor(logger = self.logger)
		if self.parameters.optimize_nseg_by_lambda and self.parameters.set_nseg_by_length:
			raise ValueError("Cannot set nseg by length and optimize nseg by lambda at the same time. Please choose one of these options in the parameters.")
		elif self.parameters.set_nseg_by_length:
			self.logger.log("Setting nseg by length.")
			for sec in cell.all:
				if sec not in cell.soma:
					sec.nseg = max(1, int(math.ceil(sec.L / self.parameters.microns_per_segment)))
		elif self.parameters.optimize_nseg_by_lambda:
				self.logger.log("Updating nseg using lambda.")
				reductor.update_model_nseg_using_lambda(cell, segs_per_lambda=self.parameters.segs_per_lambda)
		self.logger.log(f"Total number of segments: {sum([sec.nseg for sec in cell.all])}")

		if self.parameters.merge_synapses:
				self.logger.log("Merging synapses.")
				reductor.merge_synapses(cell)
		self.logger.log(f"Total number of synapses after merging: {len(cell.get_synapses(['all']))}")

		# self.check_and_save_generated_synapses(cell, exc_spike_trains, exc_mean_frs) # will be using synapses.csv

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
		self.logger.log_runtime("cell_builder", "build_cell", run_time)
        
		return cell, skeleton_cell

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
	
	def build_Allen_cell(self) -> object:
		skeleton_cell = load_skeleton_cell_from_allen(SkeletonCell.Allen.value["directory"])
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
  
	def check_and_save_generated_synapses(self, cell, exc_spike_trains, exc_mean_frs) -> None:
		if self.parameters.all_synapses_off:
			self.logger.log("Not checking synapses.")
			return None
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


		# record spike trains #@MARK need to update this to gather the names of the synapses for each spike train.
		if self.parameters.record_spike_trains:
			spike_train_data = {
				'exc_spike_trains': exc_spike_trains,
				'soma_spike_trains': [syn.pc.spike_train for syn in cell.get_synapses(['inh_perisomatic'])],
				'inh_spike_trains': [syn.pc.spike_train for syn in cell.get_synapses([self.parameters.inh_syn_properties.keys()])]
			}
			for dataset_name, data in spike_train_data.items():
				file_path = os.path.join(self.parameters.path, f'{dataset_name}.h5')
				with h5py.File(file_path, 'w') as h5f:
					for i, sequence in enumerate(data):
						h5f.create_dataset(f'spike_train_{i}', data=sequence)

		# Record synapse distributions
		if self.parameters.record_synapse_distributions:
			all_segments = cell.get_segments_without_data(['all'])
			# print(f"length of all segments in builder when saving synapses: {len(all_segments)}")
			# soma_synapses = cell.get_synapses(['soma_inh'])
			# if len(soma_synapses) == 0:
			# 	print("No soma synapses found. Feel free to delete.")
			# inh_synapses = cell.get_synapses(['inh', 'inh_distal_basal', 'inh_distal_apic'])
			inh_synapses = cell.get_synapses([f'inh_{sec_type}' for sec_type in self.parameters.inh_syn_properties.keys()])
			# exc_synapses = cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"], all_with_prefix=True)
			# exc_synapses = cell.get_synapses(["exc", "exc_apic", "exc_tuft","exc_basal","exc_dend","exc_trunk","exc_oblique"])
			exc_synapses = cell.get_synapses([f"exc_{sec_type}" for sec_type in self.parameters.exc_syn_properties.keys()])
			synapse_data = {
				'synapse_type': (
					# [syn.name for syn in soma_synapses] +
					[syn.name for syn in inh_synapses] +
					[syn.name for syn in exc_synapses]
					# ['soma_inh'] * len(soma_synapses) +
					# ['inh'] * len(inh_synapses) +
					# ['exc'] * len(exc_synapses)
				),
				'mean_firing_rate': (
					# [syn.pc.mean_fr for syn in soma_synapses] +
					[syn.pc.mean_fr for syn in inh_synapses] +
					[syn.pc.mean_fr for syn in exc_synapses]
				),
				'weight': (
					# [syn.gmax_val for syn in soma_synapses] +
					[syn.gmax_val for syn in inh_synapses] +
					[syn.gmax_val for syn in exc_synapses]
				),
				'seg_id': (
					# [all_segments.index(syn.h_syn.get_segment()) for syn in soma_synapses] +
					[all_segments.index(syn.h_syn.get_segment()) for syn in inh_synapses] +
					[all_segments.index(syn.h_syn.get_segment()) for syn in exc_synapses]
				),
				'pc_name': (
					# [syn.pc.name for syn in soma_synapses] +
					[syn.pc.name for syn in inh_synapses] +
					[syn.pc.name for syn in exc_synapses]
				)
			}

			# # Check which elements are of object dtype since that is giving error.
			# for key, values in synapse_data.items():
			# 	values_array = np.array(values)
			# 	if values_array.dtype == np.object:
			# 		print(f"Key '{key}' has object dtype: {values_array.dtype}")
			# 		print(f"Unique values: {np.unique(values_array)}")
			# 		for idx,value in enumerate(values_array):
			# 			synapse_type = synapse_data['synapse_type'][idx]
			# 			print(f"{synapse_type} Value type: {type(value)}, value: {value}")
			# 	else:
			# 		print(f"Key '{key}' has dtype: {values_array.dtype}")

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