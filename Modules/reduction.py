from neuron_reduce import subtree_reductor
from Modules.cable_expander_func import cable_expander
import numpy as np

class Reductor():

	def __init__(self, cell = None, method = None, synapses_list = None, 
			   netcons_list = None, reduction_frequency = 0, sections_to_expand = None, 
			   furcations_x = None, nbranches = None, segs_per_lambda: int = 10, return_seg_to_seg: bool = False) -> None:
		'''
		Paramters:
		----------
		cell: hoc model cell object (TO DO: providing python cell_model object instead)
		method: str for method to use ex. 'expand cable', 'neuron_reduce'
		synapses_list: list of synapse objects
		netcons_list: list of netcon objects
		reduction_frequency: frequency used in calculated transfer impedance
		return_seg_to_seg: bool for returning a dictionary mapping original segments to reduced.
		'''
		self.original_cell = cell
	
		if self.original_cell is not None:
			if method == 'expand cable':
				self.check_sanity_in_expand_cable(sections_to_expand, furcations_x, nbranches)
				self.reduced_cell, self.synapses_list, self.netcons_list, self.txt = cable_expander(cell, sections_to_expand, furcations_x, 
												nbranches, synapses_list, netcons_list, 
												reduction_frequency=reduction_frequency, 
												return_seg_to_seg = True)
			elif method == 'neuron_reduce':
					self.reduced_cell, self.synapses_list, self.netcons_list, self.txt = subtree_reductor(cell, synapses_list, netcons_list, 
													  reduction_frequency = reduction_frequency, 
													  return_seg_to_seg = True)
			elif method == 'lambda':
					self.update_model_nseg_using_lambda(cell, segs_per_lambda)
			else:
				raise NotImplementedError
	
	
	def check_sanity_in_expand_cable(self, sections_to_expand = None, furcations_x = None, nbranches = None):
		if not sections_to_expand:
			raise ValueError('Must specify sections_to_expand for cable_expander().')
		if not furcations_x:
			raise ValueError('Must specify furcations_x list for cable_expander().')
		if not nbranches:
			raise ValueError('Must specify nbranches list for cable_expander().')

	def get_other_seg_from_seg_to_seg(self, segments, seg: str, seg_to_seg: dict):
		'''
		WORK IN PROGRESS
		segments: list of segments to search through
		seg: str or nrn.segment for which you wish to return the mapped segments
		works with find_seg_to_seg, get_str_from_dict, and get_seg_from_str to return segment objects from seg_to_seg
		'''
		seg_mapping = self.find_seg_to_seg(seg, seg_to_seg)
		# seg_to_find=not_original_seg
		seg_to_find_str = self.get_str_from_dict(seg_mapping, seg_to_find_str)
		seg = self.get_seg_from_str(segments, seg_to_find_str)
		return seg
	
	def find_seg_to_seg(self, seg: str, seg_to_seg: dict):
		'''
		TO DO:
		Finds and returns a str of the given segment's mapping where the segment could be either a key (original model seg) or item (reduced model seg)
		in the case where the new seg is an item from a cable_expand model, the dictionary will have {original seg: reduced seg, reduced seg, ... } if seg is expanded section
		can add a bool for if you specify reduced seg in this case to only return {original seg: desired reduced seg}
		return mapping
		'''
		pass

	#TODO: potentially a @staticmethod or an out-of-class function
	def get_str_from_dict(self):
		'''
		separates a desired string from a dictionary (to use in combination
		'''
		pass
  
	def get_seg_from_str(self, segments, string: str):
		'''
		searches list of segments for segment corresponding to str
		returns segment
		'''
		pass

	def find_space_const_in_cm(self, diameter, rm, ra):
		''' returns space constant (lambda) in cm, according to: space_const = sqrt(rm/(ri+r0)) '''
		# rm = Rm/(PI * diam), diam is in cm and Rm is in ohm * cm^2
		rm = float(rm) / (np.pi * diameter)
		# ri = 4*Ra/ (PI * diam^2), diam is in cm and Ra is in ohm * cm
		ri = float(4 * ra) / (np.pi * (diameter**2))
		space_const = np.sqrt(rm / ri)  # r0 is negligible
		return space_const

	def calculate_nseg_from_lambda(self, section, segs_per_lambda):
		rm = 1.0 / section.g_pas  # in ohm * cm^2
		ra = section.Ra  # in ohm * cm
		diam_in_cm = section.L / 10000
		space_const_in_cm = self.find_space_const_in_cm(diam_in_cm, rm, ra)
		space_const_in_micron = 10000 * space_const_in_cm
		nseg = int((float(section.L) / space_const_in_micron) * segs_per_lambda / 2) * 2 + 1
		return nseg
  
	def update_model_nseg_using_lambda(self, cell, segs_per_lambda: int = 10):
		'''
		Optimizes number of segments using length constant
		'''
		initial_nseg, new_nseg = 0, 0

		for sec in cell.all:
			initial_nseg += sec.nseg
			sec.nseg = self.calculate_nseg_from_lambda(sec, segs_per_lambda)
			new_nseg += sec.nseg

		#TODO: potentially change to warnings.warn()
		if initial_nseg != new_nseg:
			print('Model nseg changed from', initial_nseg, 'to', new_nseg)
		else:
			print('Model nseg did not change')
