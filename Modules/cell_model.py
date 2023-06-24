import numpy as np
import pandas as pd
import warnings
from typing import Union, Tuple, List, Optional, Any, TYPE_CHECKING
from neuron import h
from Modules.recorder import Recorder
from Modules.cell_utils import calc_seg_coords
import shutil
import os
import h5py
import csv

# Typing
from typing import TypeVar
NeuronHocTemplate = TypeVar("NeuronHocTemplate")
NeuronAnyType = TypeVar("NeuronAnyType")
NeuronSection = TypeVar("NeuronSection")
NeuronSegment = TypeVar("NeuronSegment")
NeuronAnyNumber = TypeVar("NeuronAnyNumber")

class CellModel:

    def __init__(self, hoc_model: NeuronHocTemplate, synapses: list = [], netcons: list = [], spike_trains: list = [], spike_threshold: Optional[float] = None) -> None:

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

        self.generate_sec_coords()
        self.seg_coords = calc_seg_coords(self)

        self.__store_segments()
        self.__set_spike_recorder()
        #self.__store_synapses_list() #store and record synapses from the synapses_list used to initialize the cell
        self.__get_segment_info__()
        self.__insert_unused_channels()
        self.__setup_recorders()

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

    def process_soma_sec(self, sec: NeuronSection, verbose: bool) -> NeuronAnyNumber:
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

    def process_non_soma_sec(self, sec: NeuronSection, psec: NeuronSection, pseg: NeuronSegment) -> NeuronAnyNumber:
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

    def generate_phi_theta_for_apical_tuft_and_basal_dendrites(self, sec: NeuronSection) -> tuple:
        if sec in self.apic:
            if sec != self.apic[0]: # Trunk
                theta, phi = np.random.uniform(0, np.pi / 2), np.random.uniform(0, 2 * np.pi)
            else:
                theta, phi = 0, np.pi/2
        elif sec in self.dend:
            theta, phi = np.random.uniform(np.pi / 2, np.pi), np.random.uniform(0, 2 * np.pi)
        else:
            theta, phi = 0, 0
        
        return theta, phi
    
    def find_starting_position_for_a_non_soma_sec(self, psec: NeuronSection, pseg: NeuronSegment) -> list:
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
    
    # PRAGMA MARK: Segment Generation
    
        
    # PRAGMA MARK: Spike Recording

    # TODO: CHECK

    def __set_spike_recorder(self, threshold: Optional = None):
          if threshold is not None:
              self.spike_threshold = threshold
          if self.spike_threshold is None:
              self.spikes = None
          else:
              vec = h.Vector()
              nc = h.NetCon(self.soma[0](0.5)._ref_v, None, sec=self.soma[0])
              nc.threshold = self.spike_threshold
              nc.record(vec)
              self.spikes = vec
    
    def get_spike_time(self, index: Union[np.ndarray, List[int], int, str] = 0) -> np.ndarray:
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

    # TODO: CHECK

    def __store_segments(self):
        self.segments = []
        self.sec_id_in_seg = []
        nseg = 0
        for sec in self.all:
            self.sec_id_in_seg.append(nseg)
            nseg += sec.nseg
            for seg in sec:
                self.segments.append(seg)
    #             self.__store_point_processes(seg) #may be outdated (was storing netcons from netcons list into self.injection
        self._nseg = nseg
    
    def __insert_unused_channels(self):
      channels = [('NaTa_t', 'gNaTa_t_NaTa_t', 'gNaTa_tbar'),
                  ('Ca_LVAst', 'ica_Ca_LVAst', 'gCa_LVAstbar'),
                  ('Ca_HVA', 'ica_Ca_HVA', 'gCa_HVAbar'),
                  ('Ih', 'ihcn_Ih', 'gIhbar')]
      for channel, attr, conductance in channels:
          for sec in self.all:
              if not hasattr(sec(0.5), attr):
                  sec.insert(channel)
                  for seg in sec:
                      setattr(getattr(seg, channel), conductance, 0)
                  # print(channel, sec) # empty sections

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
    
    def __get_segment_info__(self):
          self.seg_info = []
          k = 0
          j = 0
          for sec in self.all:
              sec_type = sec.name().split('.')[1][:4]
              for i, seg in enumerate(sec):
                  self.seg_info.append({ #update to have consistent naming scheme (will then need to debug plotting functions too, but should be easy)
                      'seg': seg,
                      'seg index': j,
                      'p0 x3d': self.seg_coords['p0'][i][0],
                      'p0 y3d': self.seg_coords['p0'][i][1],
                      'p0 z3d': self.seg_coords['p0'][i][2],
                      'p0.5 x3d': self.seg_coords['pc'][i][0],
                      'p0.5 y3d': self.seg_coords['pc'][i][1],
                      'p0.5 z3d': self.seg_coords['pc'][i][2],
                      'p1 x3d': self.seg_coords['p1'][i][0],
                      'p1 y3d': self.seg_coords['p1'][i][1],
                      'p1 z3d': self.seg_coords['p1'][i][2],
                      'seg diam': seg.diam,
                      'bmtk index': k,
                      'x': seg.x,
                      'sec': seg.sec,
                      'Type': sec_type,
                      'sec index': int(sec.name().split('[')[2].split(']')[0]),
                      'sec diam': sec.diam,
                      'sec nseg': seg.sec.nseg,
                      'sec Ra': seg.sec.Ra,
                      'seg L': sec.L/sec.nseg,
                      'sec L': sec.L,
                      'seg SA': (sec.L/sec.nseg)*(np.pi*seg.diam),
                      'seg h.distance': h.distance(self.soma[0](0.5), seg)
                  })
                  j += 1
              k += 1
          return self.__get_parent_segment_ids()
        
    def __get_parent_segment_ids(self):
          for seg in self.seg_info:
              seg['pseg index'] = None
          pseg_ids = []
          for i, seg in enumerate(self.seg_info):
              idx = int(np.floor(seg['x'] * seg['sec nseg']))
              if idx != 0:
                  pseg_id = i-1
              else:
                  pseg = seg['seg'].sec.parentseg()
                  if pseg is None:
                      pseg_id = None
                  else:
                      psec = pseg.sec
                      nseg = psec.nseg
                      pidx = int(np.floor(pseg.x * nseg))
                      if pseg.x == 1.:
                          pidx -= 1
                      try:
                          pseg_id = next(idx for idx, info in enumerate(self.seg_info) if info['seg'] == psec((pidx + .5) / nseg))
                      except StopIteration:
                          pseg_id = "Segment not in segments"
                  self.seg_info[i]['pseg index'] = pseg_id
              # pseg_ids.append(pseg_id)
          return self.__get_segment_elec_distance()
    
    def __get_segment_elec_distance(self):
        #TODO implement calculate nexus elec distance (need nexus seg)
          for seg in self.seg_info:
              seg['seg elec distance'] = {}
          freqs = {'delta': 1, 'theta': 4, 'alpha': 8, 'beta': 12, 'gamma': 30}
     
          soma_passive_imp = h.Impedance()
          soma_active_imp = h.Impedance()
          nexus_passive_imp = h.Impedance()
          nexus_active_imp = h.Impedance()
          soma_passive_imp.loc(self.soma[0](0.5))
          soma_active_imp.loc(self.soma[0](0.5))

          for freq_name, freq_hz in freqs.items():
              soma_passive_imp.compute(freq_hz + 1 / 9e9, 0) #passive from soma
              soma_active_imp.compute(freq_hz + 1 / 9e9, 1) #active from soma
              # nexus_passive_imp.compute(freq_hz + 1 / 9e9, 0) #passive from nexus
              # nexus_active_imp.compute(freq_hz + 1 / 9e9, 1) #active from nexus
              for i, seg in enumerate(self.segments):
                  elec_dist_info = {
                      'active_soma': soma_active_imp.ratio(seg.sec(seg.x)),
                      # 'active_nexus': nexus_active_imp.ratio(seg.sec(seg.x)),
                      'passive_soma': soma_passive_imp.ratio(seg.sec(seg.x)) #,
                      # 'passive_nexus': nexus_passive_imp.ratio(seg.sec(seg.x))
                  }
                  self.seg_info[i]['seg elec distance'][freq_name] = elec_dist_info
          return self.__calculate_netcons_per_seg()
    
    def __calculate_netcons_per_seg(self):
          NetCon_per_seg = [0] * len(self.seg_info)
          inh_NetCon_per_seg = [0] * len(self.seg_info)
          exc_NetCon_per_seg = [0] * len(self.seg_info)
    
          v_rest = -60 #used to determine exc/inh may adjust or automate
    
          # calculate number of synapses for each segment (may want to divide by segment length afterward to get synpatic density)
          for netcon in self.netcons:
              syn = netcon.syn()
              syn_type=syn.hname().split('[')[0]
              syn_seg_id = self.seg_info.index(next((s for s in self.seg_info if s['seg'] == syn.get_segment()), None))
              seg_dict = self.seg_info[syn_seg_id]
              if syn in seg_dict['seg'].point_processes():
                  NetCon_per_seg[syn_seg_id] += 1 # get synapses per segment
                  if syn_type == 'pyr2pyr':
                      exc_NetCon_per_seg[syn_seg_id] += 1
                  elif syn_type == 'int2pyr':
                      inh_NetCon_per_seg[syn_seg_id] += 1
                  elif 'AMPA_NMDA' in syn_type:
                      exc_NetCon_per_seg[syn_seg_id] += 1
                  elif 'GABA_AB' in syn_type:
                      inh_NetCon_per_seg[syn_seg_id] += 1
                  elif syn.e > v_rest:
                      exc_NetCon_per_seg[syn_seg_id] += 1
                  else:
                      inh_NetCon_per_seg[syn_seg_id] += 1
              else:
                  raise(ValueError("synapse not in designated segment's point processes"))
    
          for i, seg in enumerate(self.seg_info):
              seg['netcons_per_seg'] = {
                  'exc': exc_NetCon_per_seg[i],
                  'inh': inh_NetCon_per_seg[i],
                  'total': NetCon_per_seg[i]
              }
              seg['netcon_density_per_seg'] = {
                  'exc': exc_NetCon_per_seg[i]/seg['seg L'],
                  'inh': inh_NetCon_per_seg[i]/seg['seg L'],
                  'total': NetCon_per_seg[i]/seg['seg L']
              }
              seg['netcon_SA_density_per_seg'] = {
                  'exc': exc_NetCon_per_seg[i]/seg['seg SA'],
                  'inh': inh_NetCon_per_seg[i]/seg['seg SA'],
                  'total': NetCon_per_seg[i]/seg['seg SA']
              }
    
          return
        
    # PRAGMA MARK: Utils

    # TODO: CHECK
    def convert_section_list(self, section_list: NeuronAnyType) -> list:

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

    def __setup_recorders(self):
      self.gNaTa_T = Recorder(obj_list=self.segments, var_name='gNaTa_t_NaTa_t')
      self.ina = Recorder(obj_list=self.segments, var_name='ina_NaTa_t')
      self.ical = Recorder(obj_list=self.segments, var_name='ica_Ca_LVAst')
      self.icah = Recorder(obj_list=self.segments, var_name='ica_Ca_HVA')
      self.ih = Recorder(obj_list=self.segments, var_name='ihcn_Ih')
      self.Vm = Recorder(obj_list=self.segments)
    
    def __create_output_folder(self):
      nbranches = len(self.apic)-1
      nc_count = len(self.netcons)
      syn_count = len(self.synapses)
      seg_count = len(self.segments)
      firing_rate = len(self.spikes)/(h.tstop/1000)
    
      self.output_folder_name = (str(self.hoc_model)+"_"+
          str(int(firing_rate*10)) + "e-1Hz_"
          + str(seg_count) + "nseg_"
          + str(int(h.tstop))+ "ms_"
          + str(nbranches) + "nbranch_"
          + str(nc_count) + "NCs_"
          + str(syn_count) + "nsyn" #+ '_'
          # + str(self.runtime_in_minutes) + 'min' 
      )
    

      if os.path.exists(self.output_folder_name):
        print('Updating data folder ', self.output_folder_name)
        shutil.rmtree(self.output_folder_name)
        os.makedirs(self.output_folder_name)
      else:
        print('Outputting data to ', self.output_folder_name)
        os.makedirs(self.output_folder_name)
          
      return self.output_folder_name
    
    def get_recorder_data(self): # TODO: add check for synapse.current_type
      '''
      Method for calculating net synaptic currents and getting data after simulation
      '''
      numTstep = int(h.tstop/h.dt)
      i_NMDA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      i_AMPA_bySeg = [[0] * (numTstep+1)] * len(self.segments)
      # i_bySeg = [[0] * (numTstep+1)] * len(self.segments)
    
      for synapse in self.synapses: # record nmda and ampa synapse currents
          if ('nmda' in synapse.current_type) or ('NMDA' in synapse.current_type):
              i_NMDA = np.array(synapse.rec_vec[0])
              i_AMPA = np.array(synapse.rec_vec[1])
              seg = self.segments.index(synapse.segment)

              i_NMDA_bySeg[seg] = i_NMDA_bySeg[seg] + i_NMDA
              i_AMPA_bySeg[seg] = i_AMPA_bySeg[seg] + i_AMPA
          else:
              continue
    
      i_NMDA_df = pd.DataFrame(i_NMDA_bySeg) * 1000
      i_AMPA_df = pd.DataFrame(i_AMPA_bySeg) * 1000
    
    
      self.data_dict = {}
      self.data_dict['spikes']=self.get_spike_time()
      self.data_dict['ih_data'] = self.ih.as_numpy()
      self.data_dict['gNaTa_T_data'] = self.gNaTa_T.as_numpy()
      self.data_dict['ina_data'] = self.ina.as_numpy()
      self.data_dict['icah_data'] = self.icah.as_numpy()
      self.data_dict['ical_data'] = self.ical.as_numpy()
      self.data_dict['Vm'] = self.Vm.as_numpy()
      self.data_dict['i_NMDA'] = i_NMDA_df
      self.data_dict['i_AMPA'] = i_AMPA_df
      # self.data_dict['i'] = i_bySeg
      self.__create_output_files(self.__create_output_folder())
    
      return self.data_dict
    
    def __create_output_files(self,output_folder_name):
      for name, data in self.data_dict.items():
        try:
          self.__report_data(f"{output_folder_name}/{name}_report.h5", data.T)
        except:
          self.__report_data(f"{output_folder_name}/{name}_report.h5", data)
    
    def __report_data(self,reportname, dataname):
      try:
          os.remove(reportname)
          print("removed old", reportname)
      except FileNotFoundError:
          pass
    
      with h5py.File(reportname, 'w') as f:
          f.create_dataset("report/biophysical/data", data=dataname)
