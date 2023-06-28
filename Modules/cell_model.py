import numpy as np
import pandas as pd
import warnings
from neuron import h, nrn
from Modules.recorder import Recorder
from Modules.cell_utils import calc_seg_coords
import os, shutil, h5py, csv

# Global Constants
FREQS = {'delta': 1, 'theta': 4, 'alpha': 8, 'beta': 12, 'gamma': 30}

CHANNELS = [('NaTa_t', 'gNaTa_t_NaTa_t', 'gNaTa_tbar'),
            ('Ca_LVAst', 'ica_Ca_LVAst', 'gCa_LVAstbar'),
            ('Ca_HVA', 'ica_Ca_HVA', 'gCa_HVAbar'),
            ('Ih', 'ihcn_Ih', 'gIhbar')]

class CellModel:

    def __init__(self, hoc_model: object, synapses: list = [], netcons: list = [], spike_trains: list = [], 
                 spike_threshold: list = None):

        # Parse the hoc model
        self.all, self.soma, self.apic, self.dend, self.axon = None, None, None, None, None
        for model_part in ["all", "soma", "apic", "dend", "axon"]:
            setattr(self, model_part, self.convert_section_list(getattr(hoc_model, model_part)))
        self.hoc_model = hoc_model
        self.synapses = synapses
        self.netcons = netcons
        self.spike_trains = spike_trains
        self.spike_threshold = spike_threshold
        self.injection = []
        
        # Angles and rotations that were used to branch the cell
        # Store to use for geometry file generation
        self.sec_angs = [] 
        self.sec_rots = []

        # Segments
        self.segments = []
        self.sec_id_in_seg = []
        self.nseg = self._nceg = None # _nceg is for compatibility and staged for deprecation
        self.seg_info = []

        # Spikes
        self.spikes = None

        self.generate_sec_coords()
        self.seg_coords = calc_seg_coords(self)

        self.init_segments()
        self.set_spike_recorder()

        self.init_segment_info()
        self.recompute_parent_segment_ids()
        self.recompute_segment_elec_distance()
        self.recompute_netcons_per_seg()


        self.insert_unused_channels()
        self.setup_recorders()

    # PRAGMA MARK: Section Generation

    # TODO: CHECK
    def generate_sec_coords(self, verbose = True) -> None:

        for sec in self.all:
            # Do only for sections without already having 3D coordinates
            if sec.n3d() != 0: continue

            if verbose: print(f"Generating 3D coordinates for {sec}")

            # Store for a check later
            old_length = sec.L

            if sec is self.soma:
                new_length = self.process_soma_sec(sec, verbose)
            else:
                # Get the parent segment, sec
                pseg = sec.parentseg()
                if pseg is None: raise RuntimeError("Section {sec} is attached to None.")
                psec = pseg.sec

                # Process and get the new length
                new_length = self.process_non_soma_sec(sec, psec, pseg)
            
            if np.abs(new_length - old_length) >= 1: # Otherwise, it is a precision issue
                warnings.warn(f"Generation of 3D coordinates resulted in change of section length for {sec} from {old_length} to {sec.L}",
                              RuntimeWarning)

    def process_soma_sec(self, sec: h.Section, verbose: bool) -> float:
        self.sec_angs.append(0)
        self.sec_rots.append(0)

        if sec.nseg != 1:
            if verbose:
                print(f'Changing soma nseg from {sec.nseg} to 1')
            sec.nseg = 1

        sec.pt3dclear()
        sec.pt3dadd(*[0., -1 * sec.L / 2., 0.], sec.diam)
        sec.pt3dadd(*[0., sec.L / 2., 0.], sec.diam)

        return sec.L

    def process_non_soma_sec(self, sec: h.Section, psec: h.Section, pseg: nrn.Segment) -> float:
        # Get random theta and phi values for apical tuft and basal dendrites
        theta, phi = self.generate_phi_theta_for_apical_tuft_and_basal_dendrites(sec)

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

    def generate_phi_theta_for_apical_tuft_and_basal_dendrites(self, sec: h.Section, 
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
    
    def find_starting_position_for_a_non_soma_sec(self, psec: h.Section, pseg: nrn.Segment) -> list:
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
    
    # PRAGMA MARK: Spike Recording

    #TODO: CHECK
    #TODO: check if can transfer to the Recorder class
    def set_spike_recorder(self) -> None:
        if self.spike_threshold:
            vec = h.Vector()
            nc = h.NetCon(self.soma[0](0.5)._ref_v, None, sec = self.soma[0])
            nc.threshold = self.spike_threshold
            nc.record(vec)
            self.spikes = vec
    
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
    
    # PRAGMA MARK: Segment Recording

    #TODO: CHECK
    def init_segments(self):
        nseg = 0
        for sec in self.all:
            self.sec_id_in_seg.append(nseg)
            nseg += sec.nseg
            for seg in sec: self.segments.append(seg)
        self.nseg = self._nseg = nseg
    
    def insert_unused_channels(self):
        for channel, attr, conductance in CHANNELS:
            for sec in self.all:
                if not hasattr(sec(0.5), attr):
                    sec.insert(channel)
                    for seg in sec:
                        setattr(getattr(seg, channel), conductance, 0)

    def write_seg_info_to_csv(self):
        csv_file_path = os.path.join(self.output_folder_name, 'seg_info.csv')
        if os.path.exists(csv_file_path):
            print('Updating csv ', csv_file_path)
            os.remove(csv_file_path)
        else:
            print('Creating csv ', csv_file_path)
              
        with open(csv_file_path, mode='w') as file:
            writer = csv.DictWriter(file, fieldnames=self.seg_info[0].keys())
            writer.writeheader()
            for row in self.seg_info:
                writer.writerow(row)
    
    def init_segment_info(self) -> None:
          bmtk_index = 0
          seg_index_global = 0
          for sec in self.all:
              sec_type = sec.name().split('.')[1][:4]
              for seg in sec:
                  self.seg_info.append(self.genreate_seg_info(seg, sec, sec_type, seg_index_global, bmtk_index))
                  seg_index_global += 1
              bmtk_index += 1
        
    def recompute_parent_segment_ids(self) -> None:
        for seg in self.seg_info: seg['pseg_index'] = None

        for i, seg in enumerate(self.seg_info):
            idx = int(np.floor(seg['x'] * seg['sec_nseg']))

            if idx != 0:
                pseg_id = i - 1
            else:
                pseg = seg['seg'].sec.parentseg()
                if pseg is None:
                    pseg_id = None
                else:
                    psec = pseg.sec
                    nseg = psec.nseg

                    pidx = int(np.floor(pseg.x * nseg))
                    if pseg.x == 1: pidx -= 1

                    try:
                        pseg_id = next(idx for idx, info in enumerate(self.seg_info) if info['seg'] == psec((pidx + 0.5) / nseg))
                    except StopIteration:
                        pseg_id = None
                          
            self.seg_info[i]['pseg_index'] = pseg_id
    
    #TODO: implement calculate nexus elec distance (need nexus seg)
    def recompute_segment_elec_distance(self) -> None:
        for seg in self.seg_info: seg['seg_elec_distance'] = {}
     
        soma_passive_imp = h.Impedance()
        soma_active_imp = h.Impedance()
        # nexus_passive_imp = h.Impedance()
        # nexus_active_imp = h.Impedance()
        soma_passive_imp.loc(self.soma[0](0.5))
        soma_active_imp.loc(self.soma[0](0.5))

        for freq_name, freq_hz in FREQS.items():
            soma_passive_imp.compute(freq_hz + 1 / 9e9, 0) # Passive from soma
            soma_active_imp.compute(freq_hz + 1 / 9e9, 1) # Active from soma
            # nexus_passive_imp.compute(freq_hz + 1 / 9e9, 0) # Passive from nexus
            # nexus_active_imp.compute(freq_hz + 1 / 9e9, 1) # Active from nexus
            for i, seg in enumerate(self.segments):
                elec_dist_info = {
                    'active_soma': soma_active_imp.ratio(seg.sec(seg.x)),
                    # 'active_nexus': nexus_active_imp.ratio(seg.sec(seg.x)),
                    'passive_soma': soma_passive_imp.ratio(seg.sec(seg.x)) #,
                    # 'passive_nexus': nexus_passive_imp.ratio(seg.sec(seg.x))
                }
                self.seg_info[i]['seg_elec_distance'][freq_name] = elec_dist_info
    
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
        
    # PRAGMA MARK: Utils

    # TODO: CHECK
    def convert_section_list(self, section_list: object) -> list:

        # If the section list is a hoc object, add its sections to the python list
        if str(type(section_list)) == "<class 'hoc.HocObject'>":
            new_section_list = [sec for sec in section_list]

        # Else, the section list is actually one section, add it to the list
        elif str(type(section_list)) == "<class 'nrn.Section'>":
            new_section_list = [section_list]

        else:
            raise TypeError

        return new_section_list

    # PRAGMA MARK: Data manipulation

    # TODO: CHECK
    def setup_recorders(self):
      self.gNaTa_T = Recorder(obj_list = self.segments, var_name = 'gNaTa_t_NaTa_t')
      self.ina = Recorder(obj_list = self.segments, var_name = 'ina_NaTa_t')
      self.ical = Recorder(obj_list = self.segments, var_name = 'ica_Ca_LVAst')
      self.icah = Recorder(obj_list = self.segments, var_name = 'ica_Ca_HVA')
      self.ih = Recorder(obj_list = self.segments, var_name = 'ihcn_Ih')
      self.Vm = Recorder(obj_list = self.segments)
    
    def create_output_folder(self) -> str:
        nbranches = len(self.apic) - 1
        nc_count = len(self.netcons)
        syn_count = len(self.synapses)
        seg_count = len(self.segments)
        firing_rate = len(self.spikes) / (h.tstop / 1000)
    
        self.output_folder_name = (
            str(self.hoc_model) + "_" + 
            str(int(firing_rate*10)) + "e-1Hz_" + 
            str(seg_count) + "nseg_" + 
            str(int(h.tstop))+ "ms_" +
            str(nbranches) + "nbranch_" + 
            str(nc_count) + "NCs_" + 
            str(syn_count) + "nsyn" #+ '_'
          # + str(self.runtime_in_minutes) + 'min'
        )
    
        if os.path.exists(self.output_folder_name):
            print(f'Updating data folder {self.output_folder_name}')
            shutil.rmtree(self.output_folder_name)
        else:
            print(f'Outputting data to {self.output_folder_name}')
        
        os.makedirs(self.output_folder_name)

        return self.output_folder_name
    
    def get_recorder_data(self) -> dict: # TODO: add check for synapse.current_type
      '''
      Method for calculating net synaptic currents and getting data after simulation
      '''
      numTstep = int(h.tstop/h.dt)
      i_NMDA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      i_AMPA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      i_GABA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      # i_bySeg = [[0] * (numTstep+1)] * len(self.segments)
    
      for synapse in self.synapses: # Record nmda and ampa synapse currents
          if ('nmda' in synapse.current_type) or ('NMDA' in synapse.current_type):
              i_NMDA = np.array(synapse.rec_vec[0])
              i_AMPA = np.array(synapse.rec_vec[1])
              seg = self.segments.index(synapse.segment)

              i_NMDA_bySeg[seg] = i_NMDA_bySeg[seg] + i_NMDA
              i_AMPA_bySeg[seg] = i_AMPA_bySeg[seg] + i_AMPA
              
          elif ('gaba' in synapse.syn_type) or ('GABA' in synapse.syn_type): # GABA_AB current is 'i' so use syn_mod
              i_GABA = np.array(synapse.rec_vec[0])
              seg = self.segments.index(synapse.segment)

              i_GABA_bySeg[seg] = i_GABA_bySeg[seg] + i_GABA
    
      i_NMDA_df = pd.DataFrame(i_NMDA_bySeg) * 1000
      i_AMPA_df = pd.DataFrame(i_AMPA_bySeg) * 1000
      i_GABA_df = pd.DataFrame(i_GABA_bySeg) * 1000
    
      self.data_dict = {}
      self.data_dict['spikes'] = self.get_spike_time()
      self.data_dict['ih_data'] = self.ih.as_numpy()
      self.data_dict['gNaTa_T_data'] = self.gNaTa_T.as_numpy()
      self.data_dict['ina_data'] = self.ina.as_numpy()
      self.data_dict['icah_data'] = self.icah.as_numpy()
      self.data_dict['ical_data'] = self.ical.as_numpy()
      self.data_dict['Vm'] = self.Vm.as_numpy()
      self.data_dict['i_NMDA'] = i_NMDA_df
      self.data_dict['i_AMPA'] = i_AMPA_df
      self.data_dict['i_GABA'] = i_GABA_df
      # self.data_dict['i'] = i_bySeg
      self.write_data(self.create_output_folder())
      
      return self.data_dict
    
    def write_data(self, output_folder_name):
        for name, data in self.data_dict.items():
            self.write_datafile(f"{output_folder_name}/{name}_report.h5", data)
    
    def write_datafile(self, reportname, data):
        if os.path.isfile(reportname):
            os.remove(reportname)
            print(f"Removed old {reportname}")
    
        with h5py.File(reportname, 'w') as file:
            file.create_dataset("report/biophysical/data", data = data)

    def genreate_seg_info(self, seg, sec, sec_type, seg_index_global, bmtk_index) -> dict:
        info = {
            'seg': seg,
            'seg_index_global': seg_index_global,
            'p0_x3d': self.seg_coords['p0'][seg_index_global][0],
            'p0_y3d': self.seg_coords['p0'][seg_index_global][1],
            'p0_z3d': self.seg_coords['p0'][seg_index_global][2],
            'p0.5_x3d': self.seg_coords['pc'][seg_index_global][0],
            'p0.5_y3d': self.seg_coords['pc'][seg_index_global][1],
            'p0.5_z3d': self.seg_coords['pc'][seg_index_global][2],
            'p1_x3d': self.seg_coords['p1'][seg_index_global][0],
            'p1_y3d': self.seg_coords['p1'][seg_index_global][1],
            'p1_z3d': self.seg_coords['p1'][seg_index_global][2],
            'seg_diam': seg.diam,
            'bmtk_index': bmtk_index,
            'x': seg.x,
            'sec': seg.sec,
            'type': sec_type,
            'sec_index': int(sec.name().split('[')[2].split(']')[0]),
            'sec_diam': sec.diam,
            'sec_nseg': seg.sec.nseg,
            'sec_Ra': seg.sec.Ra,
            'seg_L': sec.L / sec.nseg,
            'sec_L': sec.L,
            'seg_SA': (sec.L / sec.nseg) * (np.pi * seg.diam),
            'seg_h.distance': h.distance(self.soma[0](0.5), seg),
            'seg_half-seg RA': 0.01 * seg.sec.Ra * (sec.L / 2 / seg.sec.nseg) / (np.pi * (seg.diam / 2)**2),
            'pseg': seg.sec.parentseg(),
            'pseg_index': None,
            'seg_elec_distance': {}
        }
        return info
