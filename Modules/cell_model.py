import numpy as np
import pandas as pd
import os, h5py, csv

from neuron import h

from Modules.recorder import Recorder
from Modules.logger import Logger

class CellModel:

	FREQS = {'delta': 1, 'theta': 4, 'alpha': 8, 'beta': 12, 'gamma': 30}

	def __init__(
			self, 
			skeleton_cell: object,
			random_state: np.random.RandomState,
			logger: Logger,
			spike_threshold: float = 10):
		
		self.logger = logger
	
		# Morphology & Geometry (parse the hoc model)
		self.all = []
		self.soma = None
		self.apic = None
		self.dend = None
		self.axon = None
		for model_part in ["all", "soma", "apic", "dend", "axon"]:
			setattr(self, model_part, self.convert_section_list(getattr(skeleton_cell, model_part)))

		# Adjust the number of soma segments
		if self.soma[0].nseg != 1:
			self.soma[0].nseg = 1
			self.logger.log(f"CellModel: changed soma nseg from {self.soma[0].nseg} to 1.")

		# Adjust coordinates
		self.assign_sec_coords(random_state)

		#TODO: make empty on init, generate as needed?
		# ----------
		# Regions
		self.basals = self.find_terminal_sections(self.dend)
		self.tufts = []
		self.obliques = []
		for sec in self.find_terminal_sections(self.apic):
			if h.distance(self.soma[0](0.5), sec(0.5)) > 800:
				self.tufts.append(sec)
			else:
				self.obliques.append(sec)

		# Trunk is a branch vs not a branch
		self.nbranch = len(self.tufts) + len(self.basals) if len(self.tufts == 1) else len(self.tufts) - 1 + len(self.basals)
		# ----------

		# Connectivity
		self.synapses = []
		self.netcons = []
		self.spike_trains = []

		# Current Injection
		self.current_injection = None

		# Recorders
		self.recorders = []
		# By default, record spikes and membrane voltage
		self.recorders.append(self.set_spike_recorder(self.soma[0], spike_threshold))
		self.recorders.append(self.set_spike_recorder(self.axon[0], spike_threshold))

		# Angles and rotations that were used to branch the cell
		# Store to use for geometry file generation
		self.sec_angs = [] 
		self.sec_rots = []

		#self.recompute_parent_segment_ids()
		#self.recompute_segment_elec_distance(segment = self.soma[0](0.5), seg_name = "soma")
		self.recompute_netcons_per_seg()
		self.compute_adjacent_segments()

		self.get_channels_from_var_names() # Get channel and attribute names from recorded channel name
		self.errors_in_setting_params = self.insert_unused_channels() # Need to update with var_names

		# Temporary fix for reference issues
		for sec in self.all:
			try: sec.insert("ursadonny")
			except Exception as e: self.errors_in_setting_params.append(e)

	def get_segments(self) -> tuple:
		'''
		Returns:
		'''
		nseg = 0
		sec_id_in_seg = []
		segments = []

		for sec in self.all:
			sec_id_in_seg.append(nseg)
			nseg += sec.nseg
			for seg in sec: segments.append(seg)

		return nseg, sec_id_in_seg, segments

	def assign_sec_coords(self, random_state: np.random.RandomState) -> None:

		for sec in self.all:
			# Do only for sections without already having 3D coordinates
			if sec.n3d() != 0: continue

			# Store for a check later
			old_length = sec.L

			if sec is self.soma:
				new_length = self.assign_coordinates_to_soma_sec(sec)
			else:
				# Get the parent segment, sec
				pseg = sec.parentseg()
				if pseg is None: raise RuntimeError("Section {sec} is attached to None.")
				psec = pseg.sec

				# Process and get the new length
				new_length = self.assign_coordinates_to_non_soma_sec(sec, psec, pseg, random_state)
			
			if np.abs(new_length - old_length) >= 1: # Otherwise, it is a precision issue
				self.logger.log(f"Generation of 3D coordinates resulted in change of section length for {sec} from {old_length} to {sec.L}")

	def assign_coordinates_to_soma_sec(self, sec: h.Section) -> float:
		self.sec_angs.append(0)
		self.sec_rots.append(0)
		sec.pt3dclear()
		sec.pt3dadd(*[0., -1 * sec.L / 2., 0.], sec.diam)
		sec.pt3dadd(*[0., sec.L / 2., 0.], sec.diam)
		return sec.L

	def assign_coordinates_to_non_soma_sec(
			self, 
			sec: h.Section, 
			psec: h.Section, 
			pseg: object, 
			random_state: np.random.RandomState) -> float:
		
		# Get random theta and phi values for apical tuft and basal dendrites
		theta, phi = self.generate_phi_theta_for_apical_tuft_and_basal_dendrites(sec, random_state)

		# Find starting position using parent segment coordinates
		pt0 = self.find_starting_position_for_a_non_soma_sec(psec, pseg)

		# Calculate new coordinates using spherical coordinates
		xyz = [sec.L * np.sin(theta) * np.cos(phi), 
			   sec.L * np.cos(theta), 
			   sec.L * np.sin(theta) * np.sin(phi)]
		
		pt1 = [pt0[k] + xyz[k] for k in range(3)]

		sec.pt3dclear()
		sec.pt3dadd(*pt0, sec.diam)
		sec.pt3dadd(*pt1, sec.diam)

		return sec.L

	def generate_phi_theta_for_apical_tuft_and_basal_dendrites(
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
	
	def find_starting_position_for_a_non_soma_sec(self, psec: h.Section, pseg: object) -> list:
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

	def set_spike_recorder(self, sec, spike_threshold) -> object:
		nc = h.NetCon(sec(0.5)._ref_v, None, sec = sec)
		nc.threshold = spike_threshold
		vec = h.Vector()
		nc.record(vec)
		return vec
	
	def get_spike_time(self, index: object = 0) -> np.ndarray:
		  """
		  Return soma spike time of the cell by index (indices), ndarray (list of ndarray)
		  Parameters
		  index: index of the cell to retrieve the spikes from
		  """
		  if self.spike_threshold is None:
			  raise ValueError("Spike recorder was not set up.")
		  if type(index) is str and index == 'all':
			  index = range(self.ncell)
		  if not hasattr(index, '__len__'):
			  spk = self.spikes.as_numpy().copy()
		  else:
			  index = np.asarray(index).ravel()
			  spk = np.array([self.spikes.as_numpy().copy() for i in index], dtype=object)
		  return spk
		
	def get_channels_from_var_names(self):
		# Identifying unique channels
		channels_set = set()
		for var_name in self.var_names:
			if (var_name not in ['i_pas', 'ik', 'ica', 'ina']) and ('ion' not in var_name):
				split_name = var_name.split('_')
				if var_name.startswith('g'):
					channels_set.add('_'.join(split_name[2:]))
				elif var_name.startswith('i'):
					channels_set.add('_'.join(split_name[1:]))
		Neymotin_channels = ['nax', 'kdmc', 'kap', 'kdr', 'hd'] # have different attribute structure as a result of the modfile
		self.CHANNELS = [
				(channel, f'gbar') if channel in Neymotin_channels else (channel, f'g{channel}bar') 
				for channel in channels_set
			]
	
	def insert_unused_channels(self):
		'''
		Method for allowing recording of channels in sections that do not have the current.
		'''
		errors_in_setting_params = []
		for channel, conductance in self.CHANNELS:
			if not (str(channel) == ''): # for some reason '' was getting added? Need to check how self.channels is formed.
			  for sec in self.all:
				  if not hasattr(sec(0.5), channel):
					  try: 
						  # Insert this channel into
						  sec.insert(channel)
					  except Exception as e:
						  errors_in_setting_params.append((sec.name(), e))
					  
					  try:
						  for seg in sec:
							  setattr(getattr(seg, channel), conductance, 0) # set the maximum conductance to zero
					  except Exception as e:
						  errors_in_setting_params.append((sec.name(), e))
		return errors_in_setting_params

	def write_seg_info_to_csv(self, path, seg_info=None, title_prefix:str = None):
		if seg_info is  None:
		  seg_info=self.seg_info
		if title_prefix:
		  csv_file_path = os.path.join(path, title_prefix+'seg_info.csv')
		else:
		  csv_file_path = os.path.join(path, 'seg_info.csv')
		with open(csv_file_path, mode = 'w') as file:
			writer = csv.DictWriter(file, fieldnames = seg_info[0].keys())
			writer.writeheader()
			for row in seg_info:
				writer.writerow(row)
			
	def compute_adjacent_segments(self):
		# getting adjacent segments using parent segment IDs
		# getting children segments
		segs = self.seg_info
		for i in range(len(segs)):  # iterate through segment index
		  psegid = segs[i]['pseg_index']
		  if psegid is not None:# if not np.isnan(psegid):
			  psegid = int(psegid)
			  for seg_index,seg in enumerate(segs):  # check segIDs
				  if psegid == seg_index:  # find parent seg from parent seg id
					  segs[psegid]['adjacent_segments'].append(self.segments[i])  # add child seg to this seg's adj_segs list
					  # cell.segments[seg_index].child_segs.append(self.segments[i])  # add child seg to this seg's child_segs list
					  segs[i]['adjacent_segments'].append(self.segments[psegid])  # add parent seg to this seg's adj_segs
					  # cell.segments[i].parent_segs.append(self.segments[psegid])  # add parent seg to thisparent probes

	
	def recompute_netcons_per_seg(self):
		NetCon_per_seg = [0] * len(self.seg_info)
		inh_NetCon_per_seg = [0] * len(self.seg_info)
		exc_NetCon_per_seg = [0] * len(self.seg_info)

		v_rest = -60 # Used to determine exc/inh may adjust or automate
	
		# Calculate number of synapses for each segment (may want to divide by segment length afterward to get synpatic density)
		for netcon in self.netcons:
			syn = netcon.syn()
			syn_type = syn.hname().split('[')[0]
			syn_seg_id = self.seg_info.index(next((s for s in self.seg_info if s['seg'] == syn.get_segment()), None))
			seg_dict = self.seg_info[syn_seg_id]
			if syn in seg_dict['seg'].point_processes():
				NetCon_per_seg[syn_seg_id] += 1 # Get synapses per segment
				if (syn_type == 'pyr2pyr') | ('AMPA_NMDA' in syn_type): # | (syn.e > v_rest): TODO: check syn.e
					exc_NetCon_per_seg[syn_seg_id] += 1
				elif (syn_type == 'int2pyr') | ('GABA_AB' in syn_type):
					inh_NetCon_per_seg[syn_seg_id] += 1
				else: # To clearly separate the default case
					inh_NetCon_per_seg[syn_seg_id] += 1
			else:
				raise(ValueError("Synapse not in designated segment's point processes."))
	
		for i, seg in enumerate(self.seg_info):
			seg['netcons_per_seg'] = {
				'exc': exc_NetCon_per_seg[i],
				'inh': inh_NetCon_per_seg[i],
				'total': NetCon_per_seg[i]
			  }
			seg['netcon_density_per_seg'] = {
				'exc': exc_NetCon_per_seg[i] / seg['seg_L'],
				'inh': inh_NetCon_per_seg[i] / seg['seg_L'],
				'total': NetCon_per_seg[i] / seg['seg_L']
			  }
			seg['netcon_SA_density_per_seg'] = {
				'exc': exc_NetCon_per_seg[i] / seg['seg_SA'],
				'inh': inh_NetCon_per_seg[i] / seg['seg_SA'],
				'total': NetCon_per_seg[i] / seg['seg_SA']
			}
		
	def convert_section_list(self, section_list: object) -> list:

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

	# PRAGMA MARK: Data manipulation

	# TODO: CHECK
	def setup_recorders(self, vector_length: int = None):
		self.recorders = {var_name: Recorder(obj_list = self.segments, var_name = var_name, vector_length = vector_length) for var_name in self.var_names}
	
	def get_output_folder_name(self) -> str:
		nbranches = len(self.apic) - 1
		nc_count = len(self.netcons)
		syn_count = len(self.synapses)
		seg_count = len(self.segments)
	
		output_folder_name = (
			str(self.hoc_model) + "_" + 
			str(seg_count) + "nseg_" + 
			str(nbranches) + "nbranch_" + 
			str(nc_count) + "NCs_" + 
			str(syn_count) + "nsyn"
		)
	
		return output_folder_name
	
	#TODO: clean
	def generate_recorder_data(self, vector_length: int = None) -> None: # TODO: add check for synapse.current_type
		'''
		Method for calculating net synaptic currents and getting data after simulation
		'''
		numTstep = vector_length
		i_NMDA_bySeg = [[0] * (numTstep)] * len(self.segments)
		i_AMPA_bySeg = [[0] * (numTstep)] * len(self.segments)
		i_GABA_bySeg = [[0] * (numTstep)] * len(self.segments)
		# i_bySeg = [[0] * (numTstep+1)] * len(self.segments)
		for synapse in self.synapses: # Record nmda and ampa synapse currents
			try:
				if ('nmda' in synapse.current_type) or ('NMDA' in synapse.current_type):
					i_NMDA = np.array(synapse.rec_vec[0])
					i_AMPA = np.array(synapse.rec_vec[1])
					seg = self.segments.index(synapse.segment)

					# Match shapes
					if len(i_NMDA) > len(i_NMDA_bySeg[seg]):
						i_NMDA = i_NMDA[:-1]
					if len(i_AMPA) > len(i_AMPA_bySeg[seg]):
						i_AMPA = i_AMPA[:-1]  

					i_NMDA_bySeg[seg] = i_NMDA_bySeg[seg] + i_NMDA
					i_AMPA_bySeg[seg] = i_AMPA_bySeg[seg] + i_AMPA
					
				elif ('gaba' in synapse.syn_type) or ('GABA' in synapse.syn_type): # GABA_AB current is 'i' so use syn_mod
					i_GABA = np.array(synapse.rec_vec[0])
					seg = self.segments.index(synapse.segment)

					if len(i_GABA) > len(i_GABA_bySeg[seg]):
						i_GABA = i_GABA[:-1]

					i_GABA_bySeg[seg] = i_GABA_bySeg[seg] + i_GABA
			except:
				continue
		
		i_NMDA_df = np.array(pd.DataFrame(i_NMDA_bySeg) )#* 1000) # conversion was for old modfile
		i_AMPA_df = np.array(pd.DataFrame(i_AMPA_bySeg) )#* 1000)
		i_GABA_df = np.array(pd.DataFrame(i_GABA_bySeg) )#* 1000)
		
		self.data_dict = {}
		# Dynamically add recorded data to data_dict
		for var_name, recorder in self.recorders.items():
				self.data_dict[var_name + '_data'] = recorder.as_numpy()
		self.data_dict['spikes'] = self.get_spike_time()
		# self.data_dict['ih_data'] = self.ihcn_Ih.as_numpy()
		# self.data_dict['gNaTa_T_data'] = self.gNaTa_T.as_numpy()
		# self.data_dict['ina_NaTa_t_data'] = self.ina_NaTa_t.as_numpy()
		# self.data_dict['ina_Nap_Et2_data'] = self.ina_Nap_Et2.as_numpy()
		# self.data_dict['ik_K_Pst_data'] = self.ik_K_Pst.as_numpy()
		# self.data_dict['ik_K_Tst_data'] = self.ik_K_Tst.as_numpy()
		# self.data_dict['ik_SK_E2_data'] = self.ik_SK_E2.as_numpy()
		# self.data_dict['ik_SKv3_1_data'] = self.ik_SKv3_1.as_numpy()
		# self.data_dict['ica_Ca_HVA_data'] = self.ica_Ca_HVA.as_numpy()
		# self.data_dict['ica_Ca_LVAst_data'] = self.ica_Ca_LVAst.as_numpy()
		# self.data_dict['i_pas_data'] = self.i_pas.as_numpy()
		self.data_dict['Vm'] = self.Vm.as_numpy()
		self.data_dict['i_NMDA'] = i_NMDA_df
		self.data_dict['i_AMPA'] = i_AMPA_df
		self.data_dict['i_GABA'] = i_GABA_df
		# self.data_dict['i'] = i_bySeg
	
	def write_data(self, full_path):
		os.makedirs(full_path)
		for name, data in self.data_dict.items():
			self.write_datafile(os.path.join(full_path, f"{name}_report.h5"), data)

		self.write_seg_info_to_csv(full_path)
	
	def write_datafile(self, reportname, data):
		with h5py.File(reportname, 'w') as file:
			# Check if the data is a DataFrame, and convert to numpy array if true
			if isinstance(data, pd.DataFrame):
						data = data.values
			file.create_dataset("report/biophysical/data", data = data)

	def get_seg_info(self) -> list:
		bmtk_index = 0
		seg_index_global = 0
		seg_info = []

		sec_coords = self.get_seg_coords()

		for sec in self.all:
			seg_id_in_sec = 0
			for seg in sec:
				seg_info.append(self.generate_info_for_a_segment(
					seg, 
					sec, 
					sec_coords.iloc[seg_index_global, :], 
					seg_index_global, 
					bmtk_index, 
					seg_id_in_sec))
				seg_index_global += 1
				seg_id_in_sec += 1
			bmtk_index += 1

		# Postprocess seg_info
		seg_info = self.recompute_parent_segment_ids(seg_info)
		seg_info = self.recompute_segment_elec_distance(self, seg_info, self.soma[0](0.5), "soma")

		return seg_info
	
	def recompute_segment_elec_distance(self, seg_info, segment, seg_name) -> list:
		if not all('seg_elec_distance' in seg for seg in seg_info):
			for seg in self.seg_info:
				seg['seg_elec_distance'] = {}
		
		passive_imp = h.Impedance()
		passive_imp.loc(segment)

		active_imp = h.Impedance()
		active_imp.loc(segment)
	
		for freq_name, freq_hz in self.FREQS.items():
			passive_imp.compute(freq_hz + 1 / 9e9, 0) 
			active_imp.compute(freq_hz + 1 / 9e9, 1) 
			for i, seg in enumerate(self.segments):
				elec_dist_info = {
					f'{seg_name}_active': active_imp.ratio(seg.sec(seg.x)),
					f'{seg_name}_passive': passive_imp.ratio(seg.sec(seg.x))
				}
				if freq_name in seg_info[i]['seg_elec_distance']:
					seg_info[i]['seg_elec_distance'][freq_name].update(elec_dist_info)
				else:
					seg_info[i]['seg_elec_distance'][freq_name] = elec_dist_info
	
	def recompute_parent_segment_ids(self, seg_info) -> list:
		for seg in seg_info: seg['pseg_index'] = None

		for i, seg in enumerate(seg_info):
			idx = int(np.floor(seg['x'] * seg['sec_nseg']))

			if idx != 0:
				seg_info[i]['pseg_index'] = i - 1
				continue
			
			pseg = seg['seg'].sec.parentseg()
			if pseg is None:
				seg_info[i]['pseg_index'] = None
				continue

			psec = pseg.sec
			nseg = psec.nseg
			pidx = int(np.floor(pseg.x * nseg))
			if pseg.x == 1: pidx -= 1
			try:
				pseg_id = next(idx for idx, info in enumerate(seg_info) if info['seg'] == psec((pidx + 0.5) / nseg))
			except StopIteration:
				pseg_id = None

			seg_info[i]['pseg_index'] = pseg_id
		
		return seg_info

	def generate_info_for_a_segment(
			self, 
			seg, 
			sec,
			seg_coord, 
			seg_index_global, 
			bmtk_index, 
			seg_id_in_sec) -> dict:
		
		# Extract section index from section name
		sec_name_parts = sec.name().split('.')
		sec_type = sec_name_parts[1][:4]
		last_part = sec_name_parts[-1]
	
		if '[' in last_part:
			sec_index = int(last_part.split('[')[1].split(']')[0])
		else:
			sec_index = 0

		info = {
			'seg': seg,
			'seg_index_global': seg_index_global,
			'p0_x3d': seg_coord['p0_0'],
			'p0_y3d': seg_coord['p0_1'],
			'p0_z3d': seg_coord['p0_2'],
			'p0.5_x3d': seg_coord['pc_0'],
			'p0.5_y3d': seg_coord['pc_1'],
			'p0.5_z3d': seg_coord['pc_2'],
			'p1_x3d': seg_coord['p1_0'],
			'p1_y3d': seg_coord['p1_1'],
			'p1_z3d': seg_coord['p1_2'],
			'seg_diam': seg.diam,
			'bmtk_index': bmtk_index,
			'x': seg.x,
			'sec': seg.sec,
			'type': sec_type,
			'sec_index': sec_index,
			'sec_diam': sec.diam,
			'sec_nseg': seg.sec.nseg,
			'sec_Ra': seg.sec.Ra,
			'seg_L': sec.L / sec.nseg,
			'sec_L': sec.L,
			'seg_SA': (sec.L / sec.nseg) * (np.pi * seg.diam),
			'seg_h_distance': h.distance(self.soma[0](0.5), seg),
			'seg_half_seg_RA': 0.01 * seg.sec.Ra * (sec.L / 2 / seg.sec.nseg) / (np.pi * (seg.diam / 2)**2),
			'pseg_index': None,
			'seg_elec_distance': {},
			'adjacent_segments': [],
			'seg_id_in_sec': seg_id_in_sec,
			'seg_gcanbar': seg.can.gcanbar if hasattr(seg, 'can') else 0,
			'seg_gcalbar': seg.cal.gcalbar if hasattr(seg, 'cal') else 0
		}
		return info
		
	def find_terminal_sections(self, region: list) -> list:
		'''
		Finds all terminal sections by iterating over all sections and returning those which are not parent sections.
		'''
		# Find non-terminal sections
		parent_sections = []
		for sec in self.all:
			if sec.parentseg() is None:
				continue
			
			if sec.parentseg().sec not in parent_sections:
				parent_sections.append(sec.parentseg().sec)
			
		terminal_sections = []
		for sec in region:
			if (sec not in parent_sections):
				terminal_sections.append(sec)

		return terminal_sections

	def set_injection(self, amp: float = 0, dur: float = 0, delay: float = 0):
		"""
		Add current injection to soma.
		"""
		self.current_injection = h.IClamp(self.soma(0.5))
		self.current_injection.amp = amp
		self.current_injection.ci.dur = dur
		self.current_injection.delay = delay

	
	def get_seg_coords(self) -> pd.DataFrame:
		nseg_total = sum(sec.nseg for sec in self.all)
		p0, p05, p1 = np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3)), np.zeros((nseg_total, 3))
		r = np.zeros(nseg_total)

		seg_idx = 0
		for sec in self.all:
			seg_length = sec.L / sec.nseg

			arc_lengths = [sec.arc3d(i) for i in range(sec.n3d())]
			coords = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())])

			for seg in sec:
				start = seg.x * sec.L - seg_length / 2
				end = seg.x * sec.L + seg_length / 2
				mid = seg.x * sec.L
			
				for i in range(len(arc_lengths) - 1):
					# Check if segment's middle is between two 3D coordinates
					if arc_lengths[i] <= mid < arc_lengths[i+1]:
						t = (mid - arc_lengths[i]) / (arc_lengths[i+1] - arc_lengths[i])
						pt = coords[i] + (coords[i+1] - coords[i]) * t
			
						# Calculate the start and end points of the segment
						direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
						p0[seg_idx] = pt - direction * seg_length / 2
						p1[seg_idx] = pt + direction * seg_length / 2
			
						# Correct the start point if it goes before 3D coordinates
						while (i > 0) and (start < arc_lengths[i]):  # Added boundary check i > 0
							i -= 1
							direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
							p0[seg_idx] = coords[i] + direction * (start - arc_lengths[i])
			
						# Correct the end point if it goes beyond 3D coordinates
						while (end > arc_lengths[i+1]) and (i+2 < len(arc_lengths)):
							i += 1
							direction = (coords[i+1] - coords[i]) / np.linalg.norm(coords[i+1] - coords[i])
							p1[seg_idx] = coords[i] + direction * (end - arc_lengths[i])
			
						p05[seg_idx] = pt
						r[seg_idx] = seg.diam / 2
						seg_idx += 1
						break

		# Create a dataframe
		colnames = [f'p0_{x}' for x in range(3)] + [f'pc_{x}' for x in range(3)] + [f'p1_{x}' for x in range(3)]
		colnames = colnames + ['r'] + [f'dl_{x}' for x in range(3)]

		seg_coords = pd.DataFrame(
			data = np.hstack((p0, p05, p1, r.reshape((-1, 1)), p1 - p0)),
			columns = colnames
		)

		return seg_coords