import numpy as np
from neuron import h
from neuron_reduce import subtree_reductor

from Modules.cable_expander_func import cable_expander, get_syn_to_netcons
from Modules.synapse import Synapse
from Modules.cell_model import CellModel
from Modules.logger import Logger

import warnings

class Reductor():

	def __init__(self, logger: Logger):
		self.logger = logger

	def reduce_cell(
		self, 
		complex_cell: object,
		reduce_cell: bool = False, 
		optimize_nseg: bool = False, 
		synapses_list: list = None, 
		netcons_list: list = None, 
		spike_trains: list = None, 
		spike_threshold: int = 10, 
		random_state: np.random.RandomState = None, 
		var_names: list = None, 
		reduction_frequency: float = 0, 
		expand_cable: bool = False, 
		choose_branches: list = None, 
		seg_to_record: str = 'soma'):
		
		# Convert Synapse objects to nrn.Synapse objects and keep a dictionary to reverse the process
		synapse_to_nrn = {syn: syn.synapse_neuron_obj for syn in synapses_list}
		nrn_synapses_list = list(synapse_to_nrn.values())

		# No reduction is needed
		if reduce_cell == False:
			if optimize_nseg: self.update_model_nseg_using_lambda(complex_cell)
			cell = CellModel(
				hoc_model = complex_cell, 
				synapses = synapses_list, 
				netcons = netcons_list,
				spike_trains = spike_trains, 
				spike_threshold = spike_threshold, 
				random_state = random_state,
				var_names = var_names, 
				seg_to_record = seg_to_record)
			
			self.logger.log(f"Reductor reported {len(cell.tufts)} terminal tuft branches in complex_cell.")
			return cell
		
		# Else -- reduce cell
		(
			self.reduced_cell, 
			nrn_synapses_list, 
			netcons_list, 
			txt_nr
		) = subtree_reductor(
			complex_cell, 
			nrn_synapses_list, 
			netcons_list, 
			reduction_frequency, 
			return_seg_to_seg = True)
		
		# Delete the old Synapse objects
		for syn in synapses_list: del syn

		# Expand cable if needed
		if expand_cable:
			sections_to_expand = [self.reduced_cell.hoc_model.apic[0]]
			furcations_x = [0.289004]
			nbranches = [choose_branches]
			(
				self.reduced_dendritic_cell, 
				nrn_synapses_list, 
				netcons_list, 
				txt_ce
			) = cable_expander(
				self.reduced_cell, 
				sections_to_expand, 
				furcations_x, 
				nbranches, 
				nrn_synapses_list, 
				netcons_list, 
				reduction_frequency, 
				return_seg_to_seg = True, 
				random_state = random_state)
			
			# Remove basal dend 3D coordinates because the point in the wrong direction for some reason
			for sec in self.reduced_dendritic_cell.dend: sec.pt3dclear()
			# Remove axon 3D coordinates because the point in the wrong direction for some reason
			for sec in self.reduced_dendritic_cell.axon: sec.pt3dclear()
			# Get the mapping of nrn.Synapse to NetCon
			syn_to_netcon = get_syn_to_netcons(netcons_list)

			# Convert nrn.Synapse objects back to Synapse class and append netcons
			synapses_list = []
			synapses_without_netcons = []

			for nrn_syn in nrn_synapses_list:
				if nrn_syn in syn_to_netcon.keys():
					syn = Synapse(syn_obj = nrn_syn)
					syn.ncs = syn_to_netcon[nrn_syn]
					synapses_list.append(syn)
				else: # Synapse did not receive netcons during cable_expander.redistribute_netcons
					nrn_syn.loc(-1)  # Disconnect the synapse in NEURON
					synapses_without_netcons.append(nrn_syn)

			self.logger.log(f'Reductor reported {len(synapses_without_netcons)} unused synapses after expansion.')
			self.logger.log(f'Reductor reported {len(synapses_list)} synapses are being used.')

	
			if optimize_nseg: self.update_model_nseg_using_lambda(self.reduced_dendritic_cell)

			cell = CellModel(
				hoc_model = self.reduced_dendritic_cell, 
				synapses = synapses_list, 
				netcons = netcons_list, 
				spike_trains = spike_trains, 
				spike_threshold = spike_threshold, 
				random_state = random_state,
				var_names = var_names, 
				seg_to_record = seg_to_record)
			
			self.logger.log(f"Reductor: {len(cell.tufts)} terminal tuft branches in reduced_dendritic_cell.")

			return cell

		# Else -- return reduced cell
		self.reduced_cell.all = []
		for model_part in ["soma", "apic", "dend", "axon"]:
			setattr(self.reduced_cell, model_part, CellModel.convert_section_list(self.reduced_cell, getattr(self.reduced_cell, model_part)))
		for sec in self.reduced_cell.soma + self.reduced_cell.apic + self.reduced_cell.dend + self.reduced_cell.axon:
			self.reduced_cell.all.append(sec)

		# Get the mapping of nrn.Synapse to NetCon
		syn_to_netcon = get_syn_to_netcons(netcons_list)
		
		# Convert nrn.Synapse objects back to Synapse class and append netcons
		synapses_list = []
		for nrn_syn in nrn_synapses_list:
			syn = Synapse(syn_obj=nrn_syn)
			syn.ncs = syn_to_netcon[nrn_syn]
			synapses_list.append(syn)
		
		if optimize_nseg: self.update_model_nseg_using_lambda(self.reduced_cell)
		cell = CellModel(
			hoc_model = self.reduced_cell, 
			synapses = synapses_list, 
			netcons = netcons_list, 
			spike_trains = spike_trains, 
			spike_threshold = spike_threshold, 
			random_state = random_state,
			var_names = var_names, 
			seg_to_record = seg_to_record)
		self.logger.log(f"Reductor reported {len(cell.tufts)} terminal tuft branches in NR reduced_cell.")

		return cell

	def find_space_const_in_cm(self, diameter: float, rm: float, ra: float) -> float:
		'''
		Returns space constant (lambda) in cm, according to: space_const = sqrt(rm/(ri+r0)) 
		'''
		# rm = Rm/(PI * diam), diam is in cm and Rm is in ohm * cm^2
		rm = float(rm) / (np.pi * diameter)
		# ri = 4*Ra/ (PI * diam^2), diam is in cm and Ra is in ohm * cm
		ri = float(4 * ra) / (np.pi * (diameter**2))
		space_const = np.sqrt(rm / ri)  # r0 is negligible

		return space_const

	def calculate_nseg_from_lambda(self, section: h.Section, segs_per_lambda: int) -> int:
		rm = 1.0 / section.g_pas  # (ohm * cm^2)
		ra = section.Ra  # (ohm * cm)
		diam_in_cm = section.L / 10000
		space_const_in_cm = self.find_space_const_in_cm(diam_in_cm, rm, ra)
		space_const_in_micron = 10000 * space_const_in_cm
		nseg = int((float(section.L) / space_const_in_micron) * segs_per_lambda / 2) * 2 + 1
		return nseg
  
	def update_model_nseg_using_lambda(self, cell: object, segs_per_lambda: int = 10):
		'''
		Optimizes number of segments using length constant.
		'''
		initial_nseg, new_nseg = 0, 0

		for sec in cell.all:
			initial_nseg += sec.nseg
			sec.nseg = self.calculate_nseg_from_lambda(sec, segs_per_lambda)
			new_nseg += sec.nseg

		if initial_nseg != new_nseg:
			warnings.warn(f"Model nseg changed from {initial_nseg} to {new_nseg}.", RuntimeWarning)

	def merge_synapses(self, cell: object = None, synapses_list: list = None):

		if cell is not None: synapses_list = cell.synapses
		
		# Dictionary to store unique synapses
		synapses_dict = {}
	
		for this_synapse in synapses_list:
			synapse_key = (this_synapse.syn_mod, this_synapse.gmax, this_synapse.synapse_neuron_obj.get_segment())
			
			# If this synapse is already present in synapses_dict, merge with existing synapse
			if synapse_key in synapses_dict:

				other_synapse = synapses_dict[synapse_key]
				
				# Move netcons to other_synapse
				for netcon in this_synapse.ncs:
					netcon.setpost(other_synapse.synapse_neuron_obj)
					other_synapse.ncs.append(netcon)
				del this_synapse
			else:
				# Otherwise, add to synapses_dict
				synapses_dict[synapse_key] = this_synapse
	
		# Reassign cell's synapses to the list of unique synapses
		if cell is not None: cell.synapses = list(synapses_dict.values())
