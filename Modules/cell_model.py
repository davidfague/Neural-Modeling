'''Note that this code is dependent on morphology reconstructions utilizing the section-type names: soma, axon, dend, apic. 
additionally, the code assumes that the morphology is reconstructed starting from the soma i.e. the soma is the first 'parent' section'''
import numpy as np
import pandas as pd
import os, h5py

from neuron import h

from recorder import SegmentRecorder, SynapseRecorder, SpikeRecorder, EmptySegmentRecorder
from recorder import SynapseRecorderList, SegmentRecorderList
from synapse import Synapse
from logger import Logger

from dataclasses import dataclass

from adjacency import find_branching_seg_with_most_branching_descendants_in_subset_y, get_divergent_children_of_branching_segments

@dataclass
class SegmentData:
	L: float
	membrane_surface_area: int
	coords: pd.DataFrame
	section: str
	index_in_section: int
	seg_half_seg_RA: float
	seg: str
	pseg: str

class CellModel:

	FREQS = {'delta': 1, 'theta': 4, 'alpha': 8, 'beta': 12, '25':25, 'gamma': 30}

	def __init__(
			self, 
			skeleton_cell: object,
			random_state: np.random.RandomState,
			neuron_r: h.Random,
			logger: Logger):
		self.skeleton_cell = skeleton_cell
		self.random_state = random_state
		self.neuron_r = neuron_r
		self.logger = logger
   
		# Morphology & Geometry (parse the hoc model)
		self.all = []
		self.soma = None
		self.apic = None
		self.dend = None
		self.axon = None
		self.update_section_lists()

		# Adjust the number of soma segments
		if self.soma[0].nseg != 1:
			self.logger.log(f"CellModel: changed soma nseg from {self.soma[0].nseg} to 1.")
			self.soma[0].nseg = 1

		# Adjust coordinates
		self._assign_sec_coords(random_state)

		# Connectivity
		self.synapses = []

		# Current Injection
		self.current_injection = None

		# Recorders
		self.recorders = []

	# ---------- HOC PARSING ----------
 
#	def update_section_lists(self):
#		for model_part in ["all", "soma", "apic", "dend", "axon"]:
#			setattr(self, model_part, self._convert_section_list(getattr(self.skeleton_cell, model_part)))

	def update_section_lists(self):
    # Initialize an empty list for 'all' to update self.all
		self.all = []
    
		for model_part in ["soma", "dend", "apic", "axon"]:
				# Retrieve the current part list using the existing method
				current_part_list = self._convert_section_list(getattr(self.skeleton_cell, model_part))
        
				# Update the specific model part attribute with the converted list
				setattr(self, model_part, current_part_list)
        
        # Extend the 'all' list with the current part list
				self.all.extend(current_part_list)

		self.all = self._convert_section_list(self.all) # may not be needed since self.all should not be hoc.SectionList

	def _convert_section_list(self, section_list: object) -> list:
		# If the section list is a hoc object, add its sections to the python list
		if str(type(section_list)) == "<class 'hoc.HocObject'>":
			new_section_list = [sec for sec in section_list]

		# Else, the section list is actually one section, add it to the list
		elif str(type(section_list)) == "<class 'nrn.Section'>":
			new_section_list = [section_list]

		# Python lists can also be passed
		elif str(type(section_list)) == "<class 'list'>":
			new_section_list = section_list
		
		else:
			raise TypeError(f"Expected input 'section_list' to be either of type hoc.HocObject, nrn.Section, or list, but got {type(section_list).__name__}")

		return new_section_list

	# ---------- COORDINATES ----------
	
	def _assign_sec_coords(self, random_state: np.random.RandomState) -> None:

		for sec in self.all:
			# Do only for sections without already having 3D coordinates
			if sec.n3d() != 0: continue

			# Store for a check later
			old_length = sec.L

			if sec is self.soma:
				new_length = self._assign_coordinates_to_soma_sec(sec)
			else:
				# Get the parent segment, sec
				pseg = sec.parentseg()
				if pseg is None: raise RuntimeError("Section {sec} is attached to None.")
				psec = pseg.sec

				# Process and get the new length
				new_length = self._assign_coordinates_to_non_soma_sec(sec, psec, pseg, random_state)
			
			if np.abs(new_length - old_length) >= 1: # Otherwise, it is a precision issue
				self.logger.log(f"Generation of 3D coordinates resulted in change of section length for {sec} from {old_length} to {sec.L}")

	def _assign_coordinates_to_soma_sec(self, sec: h.Section) -> float:
		sec.pt3dclear()
		sec.pt3dadd(*[0., -1 * sec.L / 2., 0.], sec.diam)
		sec.pt3dadd(*[0., sec.L / 2., 0.], sec.diam)
		return sec.L

	def _assign_coordinates_to_non_soma_sec(
			self, 
			sec: h.Section, 
			psec: h.Section, 
			pseg: object, 
			random_state: np.random.RandomState) -> float:
		
		# Get random theta and phi values for apical tuft and basal dendrites
		theta, phi = self._generate_phi_theta_for_apical_tuft_and_basal_dendrites(sec, random_state)

		# Find starting position using parent segment coordinates
		pt0 = self._find_starting_position_for_a_non_soma_sec(psec, pseg)

		# Calculate new coordinates using spherical coordinates
		xyz = [sec.L * np.sin(theta) * np.cos(phi), 
			   sec.L * np.cos(theta), 
			   sec.L * np.sin(theta) * np.sin(phi)]
		
		pt1 = [pt0[k] + xyz[k] for k in range(3)]

		sec.pt3dclear()
		sec.pt3dadd(*pt0, sec.diam)
		sec.pt3dadd(*pt1, sec.diam)

		return sec.L

	def _generate_phi_theta_for_apical_tuft_and_basal_dendrites(
			self, 
			sec: h.Section, 
			random_state: np.random.RandomState) -> tuple:
		
		if sec in self.apic:
			if sec != self.apic[0]: # Trunk
				theta, phi = random_state.uniform(0, np.pi / 2), random_state.uniform(0, 2 * np.pi)
			else:
				theta, phi = 0, np.pi/2
		elif sec in self.dend:
			theta, phi = random_state.uniform(np.pi / 2, np.pi), random_state.uniform(0, 2 * np.pi)
		else:
			theta, phi = 0, 0
		
		return theta, phi
	
	def _find_starting_position_for_a_non_soma_sec(self, psec: h.Section, pseg: object) -> list:
		for i in range(psec.n3d() - 1):
			arc_length = (psec.arc3d(i), psec.arc3d(i + 1)) # Before, After
			if (arc_length[0] / psec.L) <= pseg.x <= (arc_length[1] / psec.L):
				# pseg.x is between 3d coordinates i and i+1
				psec_x_between_coordinates = (pseg.x * psec.L - arc_length[0]) / (arc_length[1] - arc_length[0])

				#  Calculate 3d coordinates at psec_x_between_coordinates
				xyz_before = [psec.x3d(i), psec.y3d(i), psec.z3d(i)]
				xyz_after = [psec.x3d(i + 1), psec.y3d(i+1), psec.z3d(i + 1)]
				xyz = [xyz_before[k] + (xyz_after[k] - xyz_before[k]) * psec_x_between_coordinates for k in range(3)]
				break

		return xyz
	
	def get_coords_of_segments_in_section(self, sec) -> pd.DataFrame:
     
		for i in range(sec.n3d() - 1, 0, -1):
			if (sec.x3d(i) == sec.x3d(i-1)) and (sec.y3d(i) == sec.y3d(i-1)) and (sec.z3d(i) == sec.z3d(i-1)):
				print(f"Removing duplicate coordinate at index {i} in section {sec.name()}")
				h.pt3dremove(i, sec=sec)

		seg_coords = np.zeros((sec.nseg, 13))

		seg_length = sec.L / sec.nseg
		arc_lengths = [sec.arc3d(i) for i in range(sec.n3d())]
		coords = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])
  
		seg_idx_in_sec = 0
		for seg in sec:
			start = seg.x * sec.L - seg_length / 2
			end = seg.x * sec.L + seg_length / 2
			mid = seg.x * sec.L
		
			for i in range(len(arc_lengths) - 1):
				# Check if segment's middle is between two 3D coordinates
				if (arc_lengths[i] <= mid < arc_lengths[i+1]) == False:
					continue

				t = (mid - arc_lengths[i]) / (arc_lengths[i+1] - arc_lengths[i])
				pt = coords[i] + (coords[i+1] - coords[i]) * t
	
				# Calculate the start and end points of the segment
				norm = np.linalg.norm(coords[i+1] - coords[i])
				if norm == 0:
					print(f"Zero norm encountered in section {sec.name()}, coords[{i}] = {coords[i]}, coords[{i+1}] = {coords[i+1]}")
				direction = (coords[i+1] - coords[i]) / norm

				# p0
				seg_coords[seg_idx_in_sec, 0:3] = pt - direction * seg_length / 2
				# Correct the start point if it goes before 3D coordinates
				while (i > 0) and (start < arc_lengths[i]):  # Added boundary check i > 0
					i -= 1
					norm = np.linalg.norm(coords[i+1] - coords[i])
					if norm == 0:
						print(f"Zero norm encountered in section {sec.name()}, coords[{i}] = {coords[i]}, coords[{i+1}] = {coords[i+1]}")
					direction = (coords[i+1] - coords[i]) / norm
					seg_coords[seg_idx_in_sec, 0:3] = coords[i] + direction * (start - arc_lengths[i])

				# p05
				seg_coords[seg_idx_in_sec, 3:6] = pt

				# p1
				seg_coords[seg_idx_in_sec, 6:9] = pt + direction * seg_length / 2
	
				# Correct the end point if it goes beyond 3D coordinates
				while (end > arc_lengths[i+1]) and (i+2 < len(arc_lengths)):
					i += 1
					norm = np.linalg.norm(coords[i+1] - coords[i])
					if norm == 0:
						print(f"Zero norm encountered in section {sec.name()}, coords[{i}] = {coords[i]}, coords[{i+1}] = {coords[i+1]}")
					direction = (coords[i+1] - coords[i]) / norm
					seg_coords[seg_idx_in_sec, 6:9] = coords[i] + direction * (end - arc_lengths[i])
	
				seg_coords[seg_idx_in_sec, 9] = seg.diam / 2
				seg_idx_in_sec += 1

		# Compute length (dl)
		seg_coords[:, 10:13] = seg_coords[:, 6:9] - seg_coords[:, 0:3]

		# Create a dataframe
		colnames = [f'p0_{x}' for x in range(3)] + [f'pc_{x}' for x in range(3)] + [f'p1_{x}' for x in range(3)]
		colnames = colnames + ['r'] + [f'dl_{x}' for x in range(3)]
		seg_coords = pd.DataFrame(seg_coords, columns = colnames)

		return seg_coords
	
	# ---------- SEGMENTS ----------

	def get_segments(self, section_names: list) -> tuple:
		segments = []
		datas = []

		for sec in self.all:
			if (sec.name().split(".")[-1].split("[")[0] in section_names) or ("all" in section_names):
				for index_in_section, seg in enumerate(sec):
					data = SegmentData(
						L = seg.sec.L / seg.sec.nseg,
						membrane_surface_area = np.pi * seg.diam * (seg.sec.L / seg.sec.nseg),
						coords = self.get_coords_of_segments_in_section(sec).iloc[index_in_section, :].to_frame(1).T,
						section = sec.name(),
						index_in_section = index_in_section,
						seg_half_seg_RA = 0.01 * seg.sec.Ra * (sec.L / 2 / seg.sec.nseg) / (np.pi * (seg.diam / 2) ** 2),
            			seg = str(seg),
            			pseg = str(sec.parentseg()) if index_in_section==0 else str(sec((index_in_section-0.5)/seg.sec.nseg)) # x = middle of previous segment
					)
					segments.append(seg)
					datas.append(data)

		return segments, datas

	def get_segments_without_data(self, section_names: list) -> tuple:
		segments = []
		for sec in self.all:
			if (sec.name().split(".")[-1].split("[")[0] in section_names) or ("all" in section_names):
				for index_in_section, seg in enumerate(sec):
					segments.append(seg)

		return segments

	def get_segment_length(self, seg_idx: int, segments: list):
		seg = segments[seg_idx]
		return seg.sec.L / seg.sec.nseg
	
	def get_seg_index(self, segment: object):
		indx = 0
		for sec in self.all:
			for seg in sec:
				if seg == segment: return indx
				indx += 1

	def calculate_furcation_level(self):
		"""
		Calculate the furcation level of each segment and align results with all_segments.
		
		Returns:
			list: A list of furcation levels, where the order matches the rows of all_segments.
		"""
		all_segments = self.get_segments("all")
		furcation_levels = [0] * len(all_segments)  # Initialize furcation levels with default 0

		# Helper function to get all parent sections
		def get_parent_sections(section):
			"""Recursively gather all parent sections up to the root."""
			parents = []
			current_section = section.parent
			while current_section is not None:
				parents.append(current_section)
				current_section = current_section.parent
			return parents

		# Iterate over all segments and compute furcation levels
		for i, seg in enumerate(all_segments):
			section = seg.sec
			parent_sections = get_parent_sections(section)
			furcation_level = 0
			for parent in parent_sections:
				if len(parent.children()) > 1:  # Branching point
					furcation_level += 1
			furcation_levels[i] = furcation_level

		return furcation_levels

	# ---------- SYNAPSES ----------
				
	def add_synapses_over_segments(
			self, 
			segments,
			nsyn,
			syn_mod,
			syn_params,
			gmax,
			name,
			density=False,
			seg_probs=None,
			release_p=None,
			specific_segments=None) -> None:
		
		# Check if specific segments are provided
		if specific_segments is not None:
			segments = specific_segments
			nsyn = len(specific_segments)

		total_length = sum([seg.sec.L / seg.sec.nseg for seg in segments])
		if density and specific_segments is None:
			nsyn = int(total_length * nsyn)

		if seg_probs is None:
			seg_probs = [seg_length / total_length for seg_length in [seg.sec.L / seg.sec.nseg for seg in segments]]

		for _ in range(nsyn):
			if specific_segments is None:
				segment = self.random_state.choice(segments, 1, True, seg_probs / np.sum(seg_probs))[0]
			else:
				segment = segments[_]

			if release_p is not None:
				if isinstance(release_p, dict):
					sec_type = segment.sec.name().split(".")[-1].split("[")[0]
					p = release_p[sec_type](size=1)
				else:
					p = release_p(size=1)  # release_p is partial

				pu = self.random_state.uniform(low=0, high=1, size=1)

				# Drop synapses with too low release probability unless syn_mod is 'int2pyr' or 'pyr2pyr'
				if p < pu and 'int2pyr' not in syn_mod and 'pyr2pyr' not in syn_mod:
					continue

			# Create synapse
			segment_distance = h.distance(segment, self.soma[0](0.5))
			if isinstance(syn_params, (tuple, list)):  # select a syn_param dictionary from the options
				# Excitatory
				if 'AMPA' in syn_mod or 'pyr2pyr' in syn_mod:
					syn_params = np.random.choice(syn_params, p=(0.9, 0.1))
				# Inhibitory
				elif 'GABA' in syn_mod or 'int2pyr' in syn_mod:
					# Second option is for > 100 um from soma, else first option
					syn_params = syn_params[1] if segment_distance > 100 else syn_params[0]
				else:
					raise NotImplementedError(f"syn_param selection for syn_mod: {syn_mod} not implemented")
			elif isinstance(syn_params, dict):
				pass  # its okay
			else:
				raise NotImplementedError(f"syn_param selection of type {type(syn_params)} for syn_mod: {syn_mod}")

			if 'int2pyr' in syn_mod or 'pyr2pyr' in syn_mod:  # these modfiles do release probability computation as spikes arrive during simulation instead of before
				syn_params["P_0"] = p

			self.synapses.append(Synapse(
				segment=segment, 
				syn_mod=syn_mod, 
				syn_params=syn_params, 
				gmax=gmax(size=1) if callable(gmax) else gmax,
				neuron_r=self.neuron_r,
				name=name))


	def get_synapses(self, synapse_names: list):
		return [syn for syn in self.synapses if (syn.name in synapse_names) or ('all' in synapse_names)]
	
	# ---------- CURRENT INJECTION ----------

	def set_injection(self, amp: float = 0, dur: float = 0, delay: float = 0, target='soma'):
		"""
		Add current injection to soma.
		"""
		if target == 'soma':
			self.current_injection = h.IClamp(self.soma[0](0.5))
		elif target == 'nexus':
			nexus_segment = self.find_nexus_seg()
			segments, _ = self.get_segments(['all'])
			self.current_injection = h.IClamp(segments[nexus_segment])
		self.current_injection.amp = amp
		self.current_injection.dur = dur
		self.current_injection.delay = delay

	# ---------- MORPHOLOGY ----------
 
	def find_nexus_seg(self): # TODO: implement for reducing apic to single cable (in this case nexus is not a branching point and will need to use the seg_to_seg mapping)
		all_seg_list, seg_data = self.get_segments(['all'])
		adjacency_matrix = self.compute_directed_adjacency_matrix()
		#print(f"seg_data[379].coords['p1_1']: {seg_data[379].coords['p1_1']}")
		y_coords = []
		for i, seg in enumerate(all_seg_list):
			y_coord = seg_data[i].coords["p1_1"].iloc[0] if not seg_data[i].coords["p1_1"].empty else None
			y_coords.append(y_coord)
		#print(f"y_coords: {y_coords}")
		#print(f"all_seg_list: {all_seg_list}")
		apical_segment_indices = [i for i, seg in enumerate(all_seg_list) if 'apic' in str(seg)]
		#print(f"apical_segment_indices: {apical_segment_indices}")
		nexus_index_in_all_list, _ = find_branching_seg_with_most_branching_descendants_in_subset_y(adjacency_matrix, apical_segment_indices, y_coords)
		#print(f"The found apical nexus segment is: {all_seg_list[nexus_index_in_all_list]}")
		return nexus_index_in_all_list
 
	def get_tuft_root_sections(self):
		# all_segments, _ = self.get_segments(['all'])
		# nexus_seg_index = self.find_nexus_seg()
		# nexus_seg = all_segments[nexus_seg_index]
		# return nexus_seg.sec.children()
		# return self.get_segments(['all'])[0][self.find_nexus_seg()].sec.children()
		NotImplementedError(f"DEPRECATED: use cell_model.get_root_sections('tuft') instead")

	def get_basal_root_sections(self):
		# basal_root_sections = [sec for sec in self.soma[0].children() if sec in self.dend]
		# return basal_root_sections
		NotImplementedError(f"DEPRECATED: use cell_model.get_root_sections('basal') instead")

	def get_basal_secondary_sections(self):
		# basal_secondary_sections = []
		# for sec in self.soma[0].children():
		# 	if sec in self.dend:
		# 		for second_child in sec.children():
		# 			if second_child is not None:
		# 				basal_secondary_sections.append(second_child)
		# 			else:
		# 				basal_secondary_sections.append(sec)
		# return basal_secondary_sections
		NotImplementedError(f"DEPRECATED: use cell_model.get_sections_at_branching_level('basal', 2)")

	def get_basal_sections(self, level=1): #@MARK  - Check. Potential
		# '''Function for getting basal sections at a section depth (i.e. the last sections upto n sections from the soma)'''
		# if level < 1:
		# 	raise(ValueError(f"level {level} must be less than 1"))
		# def get_children_at_level(sections, current_level, target_level):
		# 	if not sections or current_level == target_level:
		# 		return sections, current_level
		# 	next_level_sections = []
		# 	for sec in sections:
		# 		next_level_sections.extend(sec.children())
		# 	return get_children_at_level(next_level_sections, current_level + 1, target_level)
		
		# initial_sections = [sec for sec in self.soma[0].children() if sec in self.dend]
		# sections, reached_level = get_children_at_level(initial_sections, 1, level)
		
		# while not sections and level > 1:
		# 	level -= 1
		# 	sections, reached_level = get_children_at_level(initial_sections, 1, level)
		
		# return sections
		NotImplementedError(f"DEPRECATED: use cell_model.get_sections_at_branching_level('basal', level)")
	
	# @MARK Check that this one works as intended; check if level = inf returns terminal sections.
	def get_sections_at_branching_level(self, sec_type_to_get, level=1, exact_level=False):
		'''
		'sec_type_to_get' possible inputs: 'dend', 'basal', 'apic', 'trunk', 'oblique', 'tuft'
		Function for getting sec_type_to_get sections at a section depth 
		(i.e., the last sections up to n sections from the soma).
		exact_level = False will return the other terminal branches if they do not branch to the specified depth.
		
		Ex:
		Suppose the basal tree has the following structure basal1 is the root, basal2 is child to the root, 
		basal3 and basal4 are child to basal2, basal5 is child to basal4.
		In this scenario:
		level=1 returns: basal1, 1
		level=2 returns: basal2, 2
		level=3 returns [basal3, basal4], [3, 3]
		(level=4, false) returns [basal3, basal5], [3,4] since basal3 and basal4 are the terminal children and they have 2 and 3 generations above them, respectively
		(level=4, true) returns [basal5], [4] basal5 is the only section with exactl 4 ascendants
		'''
		
		if level < 1:
			raise ValueError(f"level {level} must be greater than or equal to 1")
		
		def get_children_at_level(sections, current_level, target_level):
			if not sections or current_level == target_level:
				return sections, [current_level] * len(sections)
			next_level_sections = []
			next_level_reached = []
			for sec in sections:
				children = sec.children()
				next_level_sections.extend(children)
				next_level_reached.extend([current_level + 1] * len(children))
			child_sections, child_levels = get_children_at_level(next_level_sections, current_level + 1, target_level)
			return child_sections, next_level_reached[:len(child_sections)]
		
		# Get the root sections of the specified type
		initial_sections = self.get_root_sections(sec_type_to_get)
		sections, reached_levels = get_children_at_level(initial_sections, 1, level)
		
		# Adjust the level downwards if no sections are found at the target level
		while not sections and level > 1:
			level -= 1
			sections, reached_levels = get_children_at_level(initial_sections, 1, level)

		if not sections:
			raise ValueError(f"sections returned from get_sections_at_branching_level is {sections}")
		
		if exact_level:
			sections, reached_levels = zip(*[
				(sec, lvl) for sec, lvl in zip(sections, reached_levels) if lvl == level
			])

		return sections, reached_levels

	def get_oblique_root_sections(self):
		# all_segments = self.get_segments_without_data(['all'])
		# nexus_seg_index = self.find_nexus_seg()
		# adjacency_matrix = self.compute_directed_adjacency_matrix()
		# apic_trunk_root_seg_index = all_segments.index(all_segments[0].sec.children()[1](0.0001))
		# oblique_root_seg_indices = get_divergent_children_of_branching_segments(adjacency_matrix, start=apic_trunk_root_seg_index, end=nexus_seg_index)
		# oblique_root_sections = [all_segments[seg_index].sec for seg_index in oblique_root_seg_indices]
        
		# oblique_roots_with_children = [sec for sec in oblique_root_sections if len(sec.children()) > 0]
		# oblique_roots_with_children_seg_indices = [all_segments.index(seg) for sec in oblique_roots_with_children for seg in sec]
        
		# return oblique_roots_with_children
		NotImplementedError(f"DEPRECATED: use oblique_roots_with_children = [sec for sec in cell_model.get_root_sections('oblique') if len(sec.children()) > 0]")
    
    # @DEPRACATING
	def get_apic_root_sections(self):
	# 	soma_apical_children = [sec for sec in self.soma[0].children() if sec in self.apic]
	# 	return soma_apical_children
		NotImplementedError(f"DEPRECATED: use cell_model.get_root_sections('apic')")

	# newer function merging get_tuft_root_sections, get_basal_root_sections, get_oblique_root_sections
	def get_root_sections(self, sec_type_to_get: str) -> list:
		''' 
		possible inputs: 'dend', 'basal', 'apic', 'trunk', 'oblique', 'tuft'
		'''
		actual_root_sec_types = self.get_actual_sec_types(sec_type_to_get)
		if sec_type_to_get in ['dend','basal','apic','trunk']:
			parent_sec = self.soma[0]
		elif sec_type_to_get in ['tuft']:
			parent_sec = self.get_segments(['all'])[0][self.find_nexus_seg()].sec
		elif sec_type_to_get in ['oblique']:
			return [self.get_segments_without_data(['all'])[i].sec for i in get_divergent_children_of_branching_segments(self.compute_directed_adjacency_matrix(), start=self.get_segments_without_data(['all']).index(self.get_root_sections('trunk')[0](0.0001)), end=self.find_nexus_seg())]
		else:
			NotImplementedError(f"{sec_type_to_get}")
		root_sections = [sec for sec in parent_sec.children() if sec in getattr(self, actual_root_sec_types)]
		return root_sections
	
	def get_segments_of_type(self, sec_type_to_get: str):
		def gather_segments_recursively(section, stop_segments=None):
			"""
			Recursively gather all segments from the given section and its descendants,
			stopping if a segment is in the `stop_segments` list.
			"""
			if stop_segments is None:
				stop_segments = set()
			
			segments = []
			for seg in section:
				if seg in stop_segments:
					return segments  # Stop recursion if we hit a stopping segment
				segments.append(seg)
			
			for child_section in section.children():
				segments.extend(gather_segments_recursively(child_section, stop_segments))
			
			return segments

		# Handle the 'trunk' special case
		if sec_type_to_get == 'trunk':
			# Get the stopping segments: oblique and tuft root sections' first segments
			oblique_roots = self.get_root_sections("oblique")
			tuft_roots = self.get_root_sections("tuft")

			# Include the first segment (index == 0) from each oblique and tuft root
			stop_segments = {
				seg for root in oblique_roots for idx, seg in enumerate(root) if idx == 0
			}.union(
				{seg for root in tuft_roots for idx, seg in enumerate(root) if idx == 0}
			)

			# # include nexus_seg (not needed since tuft is used.)
			# nexus_seg = self.get_nexus_segment()  # Assume this is a method to retrieve the nexus segment
			# if nexus_seg is not None:
			# 	stop_segments.add(nexus_seg)

			# Gather segments for trunk, stopping at `stop_segments`
			all_segments = []
			for root_section in self.get_root_sections("trunk"):
				all_segments.extend(gather_segments_recursively(root_section, stop_segments))
			return all_segments
		elif sec_type_to_get == 'soma':
			return [seg for seg in self.soma[0]]

		# General case: Gather all segments for the specified section type
		all_segments = []
		for root_section in self.get_root_sections(sec_type_to_get):
			all_segments.extend(gather_segments_recursively(root_section))
		
		return all_segments
	
	def get_actual_sec_types(self, sec_type_to_get):
		'''converts 'basal' to 'dend', 'trunk', 'oblique', 'tuft' to 'apic' (the 'actual' names that are the conventional attributes of cell_model and templates.)'''
		return 'dend' if sec_type_to_get in ['dend','basal'] else 'apic' if sec_type_to_get in ['apic','trunk','oblique','tuft'] else NotImplementedError(f"{sec_type_to_get}")
	
	# @MARK deprecate--possibly only used for counting total number of terminal branches (get_basals, get_tufts_obliques, get_nbranch)
	# if so then this can be just get_nbranch repeatedly calling new function: get_terminal_sections(type_to_get)

	def get_basals(self) -> list:
		# return self.find_terminal_sections(self.dend)
		NotImplementedError(f"DEPRECATING: use find_terminal_sections('basal')")
	
	def get_tufts_obliques(self) -> tuple:
		# '''only gathers terminal sections'''
		# tufts = []
		# obliques = []
		# nexus_path_distance = h.distance(self.get_segments(self.soma[0](0.5), ['all'])[0][self.find_nexus_seg()])
		# for sec in self.find_terminal_sections(self.apic):
		# 	if h.distance(self.soma[0](0.5), sec(0.5)) > nexus_path_distance:
		# 		tufts.append(sec)
		# 	else:
		# 		obliques.append(sec)

		# return tufts, obliques
		NotImplementedError(f"DEPRECATING: use find_terminal_sections('tuft'), find_terminal_sections('oblique')")
	
	def get_nbranch(self) -> int:
		'''counts tuft and basal terminal branches
		@DEPRECATING: can probably be deprecated or replaced with utilizing: len(get_sections_at_branching_level(sec_type_to_get, inf)) '''
		# tufts, _ = self.get_tufts_obliques()
		# basals = self.get_basals()
		# return len(tufts) + len(basals) if len(tufts) == 1 else len(tufts) - 1 + len(basals)
		return len(self.find_terminal_sections('tuft')) + len(self.find_terminal_sections('basal'))
	
	def find_terminal_sections(self, sec_type_to_get: str) -> list:
		'''
		possible inputs: 'all', 'apic', 'dend', 'soma', 'axon' (attributes of cell_model)
		Can be modified to allow 'tuft', 'oblique', 'trunk' inputs like get_root_sections
		Finds all terminal sections by iterating over all sections and returning those which are not parent sections.
		(it is probably faster to check if sec.children() is empty)
		@DEPRECATING: Can probably be depracated or replaced with utilizing: get_sections_at_branching_level(sec_type_to_get, inf)
		'''
		actual_sec_type = self.get_actual_sec_types(sec_type_to_get)
		# Find non-terminal sections (list of sections that are parent)
		# parent_sections = [sec.parentseg().sec for sec in self.all if sec.parentseg() is not None] # check ''' '''
		# terminal_sections = [sec for sec in getattr(self, actual_sec_type) if sec not in parent_sections]

		terminal_sections = [sec for sec in getattr(self, actual_sec_type) if sec.children() is None] # @MARK check 'is None' works

		if sec_type_to_get == 'tuft':
			nexus_path_distance = h.distance(self.get_segments(self.soma[0](0.5), ['all'])[0][self.find_nexus_seg()])
			terminal_sections = [sec for sec in terminal_sections if h.distance(self.soma[0](0.5), sec(0.5)) > nexus_path_distance]
		elif sec_type_to_get == 'oblique':
			nexus_path_distance = h.distance(self.get_segments(self.soma[0](0.5), ['all'])[0][self.find_nexus_seg()])
			terminal_sections = [sec for sec in terminal_sections if h.distance(self.soma[0](0.5), sec(0.5)) < nexus_path_distance]

		if terminal_sections == []:
			ValueError(f"no terminal sections {terminal_sections} returned for sec_type_to_get:{sec_type_to_get}") # might happen if 'is None' does not work
		return terminal_sections
	
	def compute_electrotonic_distance(self, from_segment) -> pd.DataFrame:
		passive_imp = h.Impedance()
		passive_imp.loc(from_segment)
		active_imp = h.Impedance()
		active_imp.loc(from_segment)
		
		segments, _ = self.get_segments(["all"])
		elec_distance = np.zeros((len(segments), 2 * len(self.FREQS.items())))

		colnames = []
		col_idx = 0
		for freq_name, freq_hz in self.FREQS.items():
			# 9e-9 is a Segev's value
			passive_imp.compute(freq_hz + 9e-9, 0)
			active_imp.compute(freq_hz + 9e-9, 1)
			for i, seg in enumerate(segments):
				elec_distance[i, col_idx] = active_imp.ratio(seg.sec(seg.x))
				elec_distance[i, col_idx + 1] = passive_imp.ratio(seg.sec(seg.x))

				# potential alternative to impedance_ratio:
				# input_impedance = imp_obj.input(sec=subtree_root_section) * 1000000
				# input_phase = imp_obj.input_phase(CLOSE_TO_SOMA_EDGE, sec=subtree_root_section)
				# creates a complex impedance value out of the given polar coordinates
				# input_impedance = cmath.rect(input_impedance, input_phase)
			colnames.append(f"{freq_name}_active")
			colnames.append(f"{freq_name}_passive")
			col_idx = col_idx + 2

		return pd.DataFrame(elec_distance, columns = colnames)
	
	def compute_directed_adjacency_matrix(self) -> None:
		'''
		(i, j) = 1 means i is the parent of j
		'''

		segments, _ = self.get_segments(['all'])

		adj_matrix = np.zeros((len(segments), len(segments)))

		for i, seg in enumerate(segments):

			idx = int(np.floor(seg.x * seg.sec.nseg))

			# The segment is not the first one in the section => the parent is the previous segment
			if idx != 0:
				adj_matrix[i - 1, i] = 1
				continue

			# The segment is the first one in the section
			pseg = seg.sec.parentseg()

			# Soma, do nothing
			if pseg is None: continue

			# Not soma
			pidx = int(np.floor(pseg.x * pseg.sec.nseg))
			if pseg.x == 1: pidx -= 1
			counter = 0
			for j, pot_seg in enumerate(segments):
				if str(pseg).split("(")[0] == str(pot_seg).split("(")[0]:
					if counter == pidx: 
						pseg_id = j
						break
					counter += 1

			adj_matrix[pseg_id, i] = 1

		return adj_matrix
	
	# ---------- RECORDERS ----------

	def add_spike_recorder(self, sec: object, var_name: str, spike_threshold: float):
		self.recorders.append(SpikeRecorder(sec = sec, var_name = var_name, spike_threshold = spike_threshold))
	
	def add_synapse_recorders(self, var_name: str, synapse=None) -> None:
		rec_list = SynapseRecorderList(var_name)
		if synapse is None:
			for syn in self.synapses:
				try: rec_list.add(SynapseRecorder(syn.h_syn, var_name))
				except:
						if str(syn.h_syn) == 'int2pyr' and 'gaba' in var_name.lower():
							print(f'failed: {print(syn.h_syn)}, {var_name}')
						elif str(syn.h_syn) == 'pyr2pyr' and 'nmda' in var_name.lower():
							print(f'failed: {print(syn.h_syn)}, {var_name}')
		else:
			rec_list.add(SynapseRecorder(synapse.h_syn, var_name))
		self.recorders.append(rec_list)

	def add_segment_recorders(self, var_name: str, segment_to_record=None) -> None:
		rec_list = SegmentRecorderList(var_name)
		if segment_to_record is None:
			segments, _ = self.get_segments(["all"])
			for seg in segments:
				try: rec_list.add(SegmentRecorder(seg, var_name))
				except: rec_list.add(EmptySegmentRecorder())
			self.recorders.append(rec_list)
		else:
			try: rec_list.add(SegmentRecorder(seg, var_name))
			except: rec_list.add(EmptySegmentRecorder())
			self.recorders.append(rec_list)
	
	def write_recorder_data(self, path: str, step: int) -> None:
		os.mkdir(path)

		for recorder in self.recorders:
			if type(recorder) == SpikeRecorder:
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.vec.as_numpy().reshape(1, -1))

			elif (type(recorder) == SegmentRecorder) or (type(recorder) == SynapseRecorder):
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.vec.as_numpy()[::step].reshape(1, -1))

			elif (type(recorder) == SegmentRecorderList):
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.get_combined_data()[:, ::step])

			elif (type(recorder) == SynapseRecorderList):
				segments, _ = self.get_segments(["all"])
				self._write_datafile(os.path.join(path, f"{recorder.var_name}.h5"), recorder.get_combined_data(segments, self.synapses)[:, ::step])
	
	def _write_datafile(self, reportname, data):
		with h5py.File(reportname, 'w') as file:
			file.create_dataset("data", data = data)
