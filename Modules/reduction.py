import numpy as np
from neuron import h
from neuron_reduce import subtree_reductor

from Modules.cable_expander_func import cable_expander, get_syn_to_netcons
from Modules.synapse import Synapse
from Modules.cell_model import CellModel

import warnings

class Reductor():

	def reduce_cell(self, complex_cell: object, reduce_cell: bool = False, optimize_nseg: bool = False, synapses_list: list = None, netcons_list: list = None, 
		 			spike_trains: list = None, spike_threshold: int = 10, random_state: np.random.RandomState = None, 
					var_names: list = None, reduction_frequency: float = 0, expand_cable: bool = False, choose_branches: list = None):
		
		# Convert Synapse objects to nrn.Synapse objects and keep a dictionary to reverse the process
		synapse_to_nrn = {syn: syn.synapse_neuron_obj for syn in synapses_list}
		nrn_synapses_list = list(synapse_to_nrn.values())
		
		if reduce_cell:
			self.reduced_cell, nrn_synapses_list, netcons_list, txt_nr = subtree_reductor(complex_cell, nrn_synapses_list, netcons_list, reduction_frequency, return_seg_to_seg = True)
			
			# Delete the old Synapse objects
			for syn in synapses_list: del syn

			if expand_cable:
				sections_to_expand = [self.reduced_cell.hoc_model.apic[0]]
				furcations_x = [0.289004]
				nbranches = [choose_branches]
				self.reduced_dendritic_cell, nrn_synapses_list, netcons_list, txt_ce = cable_expander(self.reduced_cell, sections_to_expand, 
											  														  furcations_x, nbranches, nrn_synapses_list, 
																									  netcons_list, reduction_frequency, 
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
				synapses_without_netcons=[]
				for nrn_syn in nrn_synapses_list:
					if nrn_syn in syn_to_netcon.keys():
						syn = Synapse(syn_obj = nrn_syn)
						syn.ncs = syn_to_netcon[nrn_syn]
						synapses_list.append(syn)
					else: # Synapse did not receive netcons during cable_expander.redistribute_netcons
						synapses_without_netcons.append(nrn_syn)
				print(f'Reductor: {len(synapses_without_netcons)} unused synapses after expansion')
        
				if optimize_nseg:
					self.update_model_nseg_using_lambda(self.reduced_dendritic_cell)

				cell = CellModel(hoc_model = self.reduced_dendritic_cell, synapses = synapses_list, netcons = netcons_list, 
								 spike_trains = spike_trains, spike_threshold = spike_threshold, random_state = random_state,
								 var_names = var_names)
				print(f"Reductor: {len(cell.tufts)} terminal tuft branches in reduced_dendritic_cell")

			else:
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
				if optimize_nseg:
					self.update_model_nseg_using_lambda(self.reduced_cell)
				cell = CellModel(hoc_model = self.reduced_cell, synapses = synapses_list, netcons = netcons_list, 
								  spike_trains = spike_trains, spike_threshold = spike_threshold, random_state = random_state,
								  var_names = var_names)
				print(f"Reductor: {len(cell.tufts)} terminal tuft branches in NR reduced_cell")
		else: # No reduction
			if optimize_nseg:
				reductor.update_model_nseg_using_lambda(self.complex_cell)
			cell = CellModel(hoc_model = complex_cell, synapses = synapses_list, netcons = netcons_list,
		    				 spike_trains = spike_trains, spike_threshold = spike_threshold, random_state = random_state,
							 var_names = var_names)
			print(f"Reductor: {len(cell.tufts)} terminal tuft branches in complex_cell")

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

	def merge_synapses(cell: object = None, synapses_list: list = None):

		if cell is not None: synapses_list = cell.synapses
		
		# Dictionary to store unique synapses
		synapses_dict = {}
	
		for this_synapse in synapses_list:
			synapse_key = (this_synapse.syn_mod, this_synapse.gmax, this_synapse.neuron_synapse_obj.get_segment())
			
			# If this synapse is already present in synapses_dict, merge with existing synapse
			if synapse_key in synapses_dict:

				other_synapse = synapses_dict[synapse_key]
				
				# Move netcons to other_synapse
				for netcon in this_synapse.ncs:
					netcon.set_post(other_synapse.neuron_synapse_obj)
					other_synapse.ncs.append(netcon)
				del this_synapse
			else:
				# Otherwise, add to synapses_dict
				synapses_dict[synapse_key] = this_synapse
	
		# Reassign cell's synapses to the list of unique synapses
		if cell is not None: cell.synapses = list(synapses_dict.values())


	# @DEPRECATED
	# ----------------------------------------

	# def __init__(self, cell: object = None, method = None, synapses_list = None, netcons_list = None, reduction_frequency = 0, 
	# 	  		 sections_to_expand = None, furcations_x = None, nbranches = None, segs_per_lambda: int = 10, 
	# 			 return_seg_to_seg: bool = False):
		
	# 	self.original_cell = cell

		# if self.original_cell is not None:
		# 	if method == 'expand cable':
		# 		self.check_sanity_in_expand_cable(sections_to_expand, furcations_x, nbranches)
		# 		self.reduced_cell, self.synapses_list, self.netcons_list, self.txt = cable_expander(cell, sections_to_expand, furcations_x, 
		# 										nbranches, synapses_list, netcons_list, 
		# 										reduction_frequency=reduction_frequency, 
		# 										return_seg_to_seg = True)
		# 	elif method == 'neuron_reduce':
		# 			self.reduced_cell, self.synapses_list, self.netcons_list, self.txt = subtree_reductor(cell, synapses_list, netcons_list, 
		# 											  reduction_frequency = reduction_frequency, 
		# 											  return_seg_to_seg = True)
		# 	elif method == 'lambda':
		# 			self.update_model_nseg_using_lambda(cell, segs_per_lambda)
		# 	elif method is not None:
		# 		raise NotImplementedError
					
					
	# def type_of_point_process(self,PP):
	#     s = PP.hname()
	#     ix = PP.hname().find("[")
	#     return s[:ix]
	
	# def add_PP_properties_to_dict(self, PP, PP_params_dict):
	#     """
	#     add the properties of a point process to PP_params_dict.
	#     The only properties added to the dictionary are those worth comparing
	#     attributes not worth comparing are not synapse properties or do not differ in value.
	#     """
	#     skipped_params = {
	#         "Section", "allsec", "baseattr", "cas", "g", "get_loc", "has_loc", "hname",
	#         'hocobjptr', "i", "loc", "next", "ref", "same", "setpointer", "state",
	#         "get_segment", "DA1", "eta", "omega", "DA2", "NEn", "NE2", "GAP1", "unirand", "randGen", "sfunc", "erand",
	#         "randObjPtr", "A_AMPA", "A_NMDA", "B_AMPA", "B_NMDA", "D1", "D2", "F", "P", "W_nmda", "facfactor", "g_AMPA", "g_NMDA", "iampa", "inmda", "on_ampa", "on_nmda", "random",  "thr_rp","AlphaTmax_gaba", "Beta_gaba", "Cainf", "Cdur_gaba", "Erev_gaba", "ICag", "Icatotal", "P0g", "W", "capoolcon", "destid", "fCag", "fmax", "fmin", "g_gaba", "gbar_gaba", "igaba", "limitW", "maxChange", "neuroM", "normW", "on_gaba", "pooldiam", "postgid", "pregid", "r_gaba", "r_nmda", "scaleW", "srcid", "tauCa", "type", "z",
	#         "d1", "gbar_ampa", "gbar_nmda","tau_d_AMPA","tau_d_NMDA","tau_r_AMPA","tau_r_NMDA","Erev_ampa","Erev_nmda", "lambda1", "lambda2", "threshold1", "threshold2",
	#     }
	
	#     syn_params_list = {
	#         "tau_r_AMPA", "tau_r_NMDA", "Use", "Dep", "Fac", "e", "u0", "initW", "taun1", "taun2", "gNMDAmax", "enmda", "taua1", "taua2", "gAMPAmax", "eampa", "AlphaTmax_ampa", "Beta_ampa", "Cdur_ampa", "AlphaTmax_nmda", "Beta_nmda", "Cdur_nmda", "initW_random", "Wmax", "Wmin", "tauD1", "tauD2", "f", "tauF", "P_0", "d2",
	#     }
	
	#     PP_params = [param for param in dir(PP) if not (param.startswith("__") or callable(getattr(PP, param)))]
	
	#     PP_params = list(filter(lambda x: x not in skipped_params, PP_params))
	
	#     syn_params = list(filter(lambda x: x in syn_params_list, PP_params))
	
	
	#     PP_params_dict[type_of_point_process(PP)] = syn_params
	
	# def synapse_properties_match(self, synapse, PP, PP_params_dict):
	#     if PP.hname()[:PP.hname().rindex('[')] != synapse.hname()[:synapse.hname().rindex('[')]:
	#         return False
	#     for param in PP_params_dict[type_of_point_process(PP)]:
	#         if(param not in ['rng'] and  # https://github.com/neuronsimulator/nrn/issues/136
	#            str(type(getattr(PP, param))) != "<type 'hoc.HocObject'>" and  # ignore hoc objects
	#            getattr(PP, param) != getattr(synapse, param)):
	#             return False
	#     return True
	
	# # from decimal import Decimal
	
	# def merge_cell_synapses(self, cell):
	#   '''WORK IN PROGRESS trouble is iterating through synapses while comparing with other synapses'''
	#   self.PP_params_dict={}
	#   # go over all point processes in this segment and see whether one
	#   # of them has the same proporties of this synapse
	#   # If there's such a synapse link the original NetCon with this point processes
	#   # If not, move the synapse to this segment.
	#   for syn_index,synapse in enumerate(cell.synapses):
	#       for PP in synapse.get_segment().point_processes():
	#           if self.type_of_point_process(PP) not in self.PP_params_dict:
	#               self.add_PP_properties_to_dict(PP, self.PP_params_dict)
	
	#           if self.synapse_properties_match(synapse, PP, self.PP_params_dict):
	#               self.netcons[syn_index].setpost(PP)
	#               break
	#       else:  # If for finish the loop -> first appearance of this synapse
	#           x=Decimal(str(x)) # patch error for passing float to synapse.loc
	#           #print("x:",x,"type:",type(x),"|section_for_synapse:",section_for_synapse,"type:",type(section_for_synapse),"|synapse:",synapse,"type:",type(synapse))
	#           synapse.loc(x, sec=section_for_synapse)
	#           new_synapses_list.append(synapse)
