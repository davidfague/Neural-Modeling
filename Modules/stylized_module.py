from abc import ABC, abstractmethod
from neuron import h
import math
import numpy as np
import pandas as pd
from typing import Optional, Union, List
from enum import Enum

class StylizedCell(ABC):
    def __init__(self, geometry: pd.DataFrame = None,
                 dl: float = 30., vrest: float = -70.0, nbranch: int = 4,
                 record_soma_v: bool = True, spike_threshold: Optional[float] = None,
                 attr_kwargs: dict = {}):
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        dL: maximum segment length
        vrest: reversal potential of leak channel for all segments
        nbranch: number of branches of each non-axial section
        record_soma_v: whether or not to record soma membrane voltage
        spike_threshold: membrane voltage threshold for recording spikes, if not specified, do not record
        attr_kwargs: dictionary of class attribute - value pairs
        """
        self._h = h
        self._dL = dl
        self._vrest = vrest
        self._nbranch = max(nbranch, 2)
        #self._record_soma_v = record_soma_v
        #self.spike_threshold = spike_threshold
        self._nsec = 0
        self._nseg = 0
        self.soma = None
        self.all = []  # list of all sections
        self.dend = [] # MARK need to add to list
        self.apic = [] # MARK need to add to list
        self.axon = [] # MARK need to add to list
        self.segments = []  # list of all segments
        self.sec_id_lookup = {}  # dictionary from section type id to section index
        self.sec_id_in_seg = []  # index of the first segment of each section in the segment list
        self.injection = []  # current injection objects
        self.synapse = []  # synapse objects
        self.spikes = None
        self.geometry = None
        self.biophysical_division()
        for key, value in attr_kwargs.items():
            setattr(self, key, value)
        self.__set_geometry(geometry)
        self.__setup_all()

    #  PRIVATE METHODS
    def __set_geometry(self, geometry: Optional[pd.DataFrame] = None):
        if geometry is None:
            raise ValueError("geometry not specified.")
        else:
            if not isinstance(geometry, pd.DataFrame):
                raise TypeError("geometry must be a pandas dataframe")
            if geometry.iloc[0]['type'] != 1:
                raise ValueError("first row of geometry must be soma")
            self.geometry = geometry.copy()

    def __setup_all(self):
        self.__create_morphology()
        self.__calc_seg_coords()
        self.set_channels()
        #self.v_rec = self.__record_soma_v() if self._record_soma_v else None
        #self.__set_spike_recorder()

    def __calc_seg_coords(self):
        """Calculate segment coordinates for ECP calculation"""
        p0 = np.empty((self._nseg, 3))
        p1 = np.empty((self._nseg, 3))
        p05 = np.empty((self._nseg, 3))
        r = np.empty(self._nseg)
        for isec, sec in enumerate(self.all):
            iseg = self.sec_id_in_seg[isec]
            nseg = sec.nseg
            pt0 = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
            pt1 = np.array([sec.x3d(1), sec.y3d(1), sec.z3d(1)])
            pts = np.linspace(pt0, pt1, 2 * nseg + 1)
            p0[iseg:iseg + nseg, :] = pts[:-2:2, :]
            p1[iseg:iseg + nseg, :] = pts[2::2, :]
            p05[iseg:iseg + nseg, :] = pts[1:-1:2, :]
            r[iseg:iseg + nseg] = sec.diam / 2
        self.seg_coords = {'dl': p1 - p0, 'pc': p05, 'r': r}

    def __create_morphology(self):
        """Create cell morphology"""
        self._nsec = 0
        rot = 2 * math.pi / self._nbranch
        for sec_id, sec in self.geometry.iterrows():
            start_idx = self._nsec
            if sec_id == 0:
                r0 = sec['R']
                pt0 = [0., -2 * r0, 0.]
                pt1 = [0., 0., 0.]
                self.soma = self.__create_section(name=sec['name'], diam=2 * r0)
                self.__set_location(self.soma, pt0, pt1, 1)
            else:
                length = sec['L']
                radius = sec['R']
                ang = sec['ang']
                nseg = math.ceil(length / self._dL)
                pid = self.sec_id_lookup[sec['pid']]
                if sec['axial']:
                    nbranch = 1
                    x = 0
                    y = length*((ang>=0)*2-1)
                else:
                    nbranch = self._nbranch
                    x = length * math.cos(ang)
                    y = length * math.sin(ang)
                    if len(pid) == 1:
                        pid = pid*nbranch
                for i in range(nbranch):
                    psec = self.all[pid[i]]
                    pt0 = [psec.x3d(1), psec.y3d(1), psec.z3d(1)]
                    pt1[1] = pt0[1] + y
                    pt1[0] = pt0[0] + x * math.cos(i * rot)
                    pt1[2] = pt0[2] + x * math.sin(i * rot)
                    section = self.__create_section(name=sec['name'], diam=2 * radius)
                    section.connect(psec(1), 0)
                    self.__set_location(section, pt0, pt1, nseg)
            self.sec_id_lookup[sec_id] = list(range(start_idx, self._nsec))
        self.__set_location(self.soma, [0., -r0, 0.], [0., r0, 0.], 1)
        self.__store_segments()

    def __create_section(self, name: str = 'null_sec', diam: float = 500.0) -> h.Section:
        if ('soma' in name):
          sec_type_index = 0
          name = f'stylized_model[0].soma[{sec_type_index}]'
        elif ('apic' in name) or ('oblique' in name) or ('tuft' in name):
          sec_type_index = len(self.apic) + 1
          name = f'stylized_model[0].apic[{sec_type_index}]'
        elif 'basal' in name:
          sec_type_index = len(self.dend) + 1
          name = f'stylized_model[0].dend[{sec_type_index}]'
        elif 'axon' in name:
          sec_type_index = len(self.axon) + 1
          name = f'stylized_model[0].axon[{sec_type_index}]'
        
        sec = h.Section(name=name, cell=self)
        sec.diam = diam
        self.all.append(sec)
        if ('apic' in name) or ('oblique' in name) or ('tuft' in name):
          self.apic.append(sec)
        elif 'basal' in name:
          self.dend.append(sec)
        elif 'axon' in name:
          self.axon.append(sec)
        self._nsec += 1
        return sec

    def __set_location(self, sec: h.Section, pt0: List[float], pt1: List[float], nseg: int):
        sec.pt3dclear()
        sec.pt3dadd(*pt0, sec.diam)
        sec.pt3dadd(*pt1, sec.diam)
        sec.nseg = nseg

    def __store_segments(self):
        self.segments = []
        self.sec_id_in_seg = []
        nseg = 0
        for sec in self.all:
            self.sec_id_in_seg.append(nseg)
            nseg += sec.nseg
            for seg in sec:
                self.segments.append(seg)
        self._nseg = nseg

#    def __record_soma_v(self) -> Recorder:
#        return Recorder(self.soma(.5), 'v')

#    def __set_spike_recorder(self, threshold: Optional = None):
#        if threshold is not None:
#            self.spike_threshold = threshold
#        if self.spike_threshold is None:
#            self.spikes = None
#        else:
#            vec = h.Vector()
#            nc = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
#            nc.threshold = self.spike_threshold
#            nc.record(vec)
#            self.spikes = vec

    #  PUBLIC METHODS
    @abstractmethod
#    def set_channels(self):
#        """Abstract method for setting biophysical properties, inserting channels"""
#        pass

    def set_channels(self):
        """Define biophysical properties, insert channels"""
        self.define_biophys_entries()
        # common parameters
        for sec in self.all:
            sec.cm = 2.0
            sec.Ra = 100
            sec.insert('pas')
            sec.e_pas = self._vrest
        # fixed parameters
        soma = self.soma
        soma.cm = 1.0           # Originally 1 
        soma.insert('NaTa_t')  # Sodium channel
        soma.insert('SKv3_1')  # Potassium channel
        soma.insert('Ca_HVA')
        soma.insert('Ca_LVAst')
        soma.insert('CaDynamics_E2')
        soma.insert('Ih')
        soma.insert('SK_E2')
        soma.insert('K_Tst')
        soma.insert('K_Pst')
        soma.insert('Nap_Et2')
        soma.ena = 50
        soma.ek = -85
        

        for isec in self.grp_ids[1]:        #prox,mid,dist basal; proxtrunk; oblique
            sec = self.get_sec_by_id(isec) 
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.insert('Ca_HVA')
            sec.insert('Ca_LVAst')
            sec.insert('Im')
            sec.insert('CaDynamics_E2')
            sec.insert('Ih')
            sec.insert('SK_E2')
            sec.ena = 50
            sec.ek = -85

        for isec in self.grp_ids[2]:
            sec = self.get_sec_by_id(isec)  # Mid Trunk
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.insert('Ca_HVA')
            sec.insert('Ca_LVAst')
            sec.insert('Im')
            sec.insert('CaDynamics_E2')
            sec.insert('Ih')
            sec.insert('SK_E2')
            sec.ena = 50
            sec.ek = -85


        for isec in self.grp_ids[3]:
            sec = self.get_sec_by_id(isec)  # Distal Trunk
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.insert('Ca_HVA')
            sec.insert('Ca_LVAst')
            sec.insert('Im')
            sec.insert('CaDynamics_E2')
            sec.insert('Ih')
            sec.insert('SK_E2')
            sec.ena = 50
            sec.ek = -85

        for isec in self.grp_ids[4]:
            sec = self.get_sec_by_id(isec)  # Tuft dendrites
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.insert('Ca_HVA')
            sec.insert('Ca_LVAst')
            sec.insert('Im')
            sec.insert('CaDynamics_E2')
            sec.insert('Ih')
            sec.insert('SK_E2')
            sec.ena = 50
            sec.ek = -85


        for isec in self.grp_ids[5]:
            sec = self.get_sec_by_id(isec)  # axon
            sec.cm = 2.0
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.insert('Ca_HVA')
            sec.insert('Ca_LVAst')
            sec.insert('Im')
            sec.insert('CaDynamics_E2')
            sec.insert('Ih')
            sec.insert('SK_E2')
            sec.insert('K_Tst')
            sec.insert('K_Pst')
            sec.insert('Nap_Et2')
            sec.ena = 50
            sec.ek = -85
		        
        for isec in self.grp_ids[6]:
            sec = self.get_sec_by_id(isec)  # inactive basal dendrites
            sec.cm = 3.0
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.insert('Ca_HVA')
            sec.insert('Ca_LVAst')
            sec.insert('CaDynamics_E2')
            sec.insert('Ih')
            sec.ena = 50
            sec.ek = -85
            # sec.Ra = 100000
            # sec.e_pas = 200

        # variable parameters
        for i,entry in enumerate(self.biophys_entries):
            for sec in self.get_sec_by_id(self.grp_ids[entry[0]]):
                setattr(sec,entry[1],self.biophys[i])
        h.v_init = self._vrest
    

    def define_biophys_entries(self):
        """
        Define list of entries of biophysical parameters.
        Each entry is a pair of group id and parameter reference string.
        Define default values and set parameters in "biophys".
        """
        #update to groups: soma,dend,apic,basal after updating gmax for calcium and Ih to be based on distance for apical.
        self.grp_sec_type_ids = [ # select section id's for each group
                                 [0], #soma
                                 [1,2,3,4,5], #basal group: prox,mid,dist basal; proxtrunk; oblique
                                 [6,7,8], #mid trunk,distal trunk, proxtuft
                                 [9], #nexus: midtuft
                                 [10], #tuft: disttuft
                                 [11], #axon
                                 [12] #passive basal dendrites
                                 ]
        self.grp_ids = []  # get indices of sections for each group
        for ids in self.grp_sec_type_ids:
            secs = []
            for i in ids:
                secs.extend(self.sec_id_lookup[i])
            self.grp_ids.append(secs)
        self.biophys_entries = [
            (0,'g_pas'),(1,'g_pas'),(2,'g_pas'),(3,'g_pas'),(4,'g_pas'),(5,'g_pas'),(6,'g_pas'),  # g_pas of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0,'gNaTa_tbar_NaTa_t'),(1,'gNaTa_tbar_NaTa_t'),(2,'gNaTa_tbar_NaTa_t'),(3,'gNaTa_tbar_NaTa_t'),(4,'gNaTa_tbar_NaTa_t'),(5,'gNaTa_tbar_NaTa_t'),(6,'gNaTa_tbar_NaTa_t'),  # gNaTa_t of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0,'gSKv3_1bar_SKv3_1'),(1,'gSKv3_1bar_SKv3_1'),(2,'gSKv3_1bar_SKv3_1'),(3,'gSKv3_1bar_SKv3_1'),(4,'gSKv3_1bar_SKv3_1'),(5,'gSKv3_1bar_SKv3_1'),(6,'gSKv3_1bar_SKv3_1'),  # gSKv3_1 of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0, 'gCa_HVAbar_Ca_HVA'),(1, 'gCa_HVAbar_Ca_HVA'),(2, 'gCa_HVAbar_Ca_HVA'),(3, 'gCa_HVAbar_Ca_HVA'),(4, 'gCa_HVAbar_Ca_HVA'),(5, 'gCa_HVAbar_Ca_HVA'),(6, 'gCa_HVAbar_Ca_HVA'),  # gCA_HVA of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0, 'gCa_LVAstbar_Ca_LVAst'),(1, 'gCa_LVAstbar_Ca_LVAst'),(2, 'gCa_LVAstbar_Ca_LVAst'),(3, 'gCa_LVAstbar_Ca_LVAst'),(4, 'gCa_LVAstbar_Ca_LVAst'),(5, 'gCa_LVAstbar_Ca_LVAst'),(6, 'gCa_LVAstbar_Ca_LVAst'),# gCA_LVAst of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (1, 'gImbar_Im'),(2, 'gImbar_Im'),(3, 'gImbar_Im'),(4, 'gImbar_Im'),(5, 'gImbar_Im'), # gIm of basal, midTrunk, distTrunk, tuft, axon
            (0,'decay_CaDynamics_E2'),(1,'decay_CaDynamics_E2'),(2,'decay_CaDynamics_E2'),(3,'decay_CaDynamics_E2'),(4,'decay_CaDynamics_E2'),(5,'decay_CaDynamics_E2'),(6,'decay_CaDynamics_E2'), # decay_CaDynamics of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0,'gamma_CaDynamics_E2'),(1,'gamma_CaDynamics_E2'),(2,'gamma_CaDynamics_E2'),(3,'gamma_CaDynamics_E2'),(4,'gamma_CaDynamics_E2'),(5,'gamma_CaDynamics_E2'),(6,'gamma_CaDynamics_E2'), # gamma_CaDynamics of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0,'gIhbar_Ih'),(1,'gIhbar_Ih'),(2,'gIhbar_Ih'),(3,'gIhbar_Ih'),(4,'gIhbar_Ih'),(5,'gIhbar_Ih'),(6,'gIhbar_Ih'), # gIh of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
            (0,'gSK_E2bar_SK_E2'),(1,'gSK_E2bar_SK_E2'),(2,'gSK_E2bar_SK_E2'),(3,'gSK_E2bar_SK_E2'),(4,'gSK_E2bar_SK_E2'),(5,'gSK_E2bar_SK_E2'), # gSk_E2 of soma, basal, midTrunk, distTrunk, tuft, axon
            (0,'gK_Tstbar_K_Tst'),(5,'gK_Tstbar_K_Tst'), # gK_Tst of soma, axon
            (0,'gK_Pstbar_K_Pst'),(5,'gK_Pstbar_K_Pst'), # gK_Pst of soma, axon
            (0,'gNap_Et2bar_Nap_Et2'),(5,'gNap_Et2bar_Nap_Et2') # gNap_Et2 of soma, axon
        ]

        default_biophys = np.array([0.0000338,0.0000467,0.0000489,0.0000589,0.0000589,0.0000325,0.0000100, # g_pas of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
                                    2.04,0.0213,0.0213,0.0213,0.0213,0.0,0.0, # gNaTa_t of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal # axon was 2.89618
                                    0.693,0.000261,0.000261,0.000261,0.000261,0.0,0.0, # gSKv3_1 of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal #axon was 0.473799
                                    0.000992,0.0,0.0000555,0.000555,.0000555,0.0,0.0,  # gCA_HVA of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal #basal was 0.000992
                                    0.00343,0.0,0.000187,0.0187,0.000187,0.0,0.0, # gCA_LVAst of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
                                    0.0000675,0.0000675,0.0000675,0.0000675,0.0, # gIm of soma, basal, midTrunk, distTrunk, tuft, axon
                                    460.0,122,122,122,122,277.300774,122, # decay_CaDynamics of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
                                    0.000501,0.000509,0.000509,0.000509,0.000509,0.000525,0.000509, # gamma_CaDynamics of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
                                    0.0002,0.0002,0.0002,0.00507257227,0.01535011884,0.0001,0.0002, # gIh of soma, basal, midTrunk, distTrunk, tuft, axon, passiveBasal
                                    0.0441,0.0012,0.0012,0.0012,0.0012,0.000047, # gSk_E2 of soma, basal, midTrunk, distTrunk, tuft, axon
                                    0.0812,0.077274, # gK_Tst of soma, axon
                                    0.00223,0.188851, # gK_Pst of soma, axon
                                    0.00172,0.0  # gNap_Et2 of soma, axon
                                    ])

        


        if self.biophys is not None:
            # print('length of default_biophys:',len(default_biophys))
            # print('length of self.biophys:',len(self.biophys))
            for i in range(len(self.biophys)):
                if self.biophys[i]>=0:
                    default_biophys[i]=self.biophys[i]
        self.biophys = default_biophys

    def biophysical_division(self):
        """Define biophysical division in morphology"""
        pass

    def get_sec_by_id(self, index):
        """Get section(s) objects by index(indices) in the section list"""
        if hasattr(index, '__len__'):
            sec = [self.all[i] for i in index]
        else:
            sec = self.all[index]
        return sec

    def get_seg_by_id(self, index):
        """Get segment(s) objects by index(indices) in the segment list"""
        if hasattr(index, '__len__'):
            seg = [self.segments[i] for i in index]
        else:
            seg = self.segments[index]
        return seg

    def set_all_passive(self, gl: float = 0.0003):
        """A use case of 'set_channels', set all sections passive membrane"""
        for sec in self.all:
            sec.cm = 1.0
            sec.insert('pas')
            sec.g_pas = gl
            sec.e_pas = self._vrest

#    def add_injection(self, sec_index, **kwargs):
#        """Add current injection to a section by its index"""
#        self.injection.append(CurrentInjection(self, sec_index, **kwargs))

#    def add_synapse(self, stim: h.NetStim, sec_index: int, **kwargs):
#        """Add synapse to a section by its index"""
#        self.synapse.append(Synapse(self, stim, sec_index, **kwargs))
#
#    def v(self) -> Optional[Union[str, np.ndarray]]:
#        """Return recorded soma membrane voltage in numpy array"""
#        if self.v_rec is None:
#            raise NotImplementedError("Soma membrane voltage has not been recorded")
#        else:
#            return self.v_rec.as_numpy()

class Cell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""
    def __init__(self,geometry=None,dL=30,vrest=-70.0):
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        dL: maximum segment length
        vrest: reversal potential of leak channel for all segments
        """
        super().__init__(geometry,dL,vrest)
        #self.record_soma_v() # uncomment this if want to record soma voltage
    
    def set_channels(self):
        """Define biophysical properties, insert channels"""
#         self.set_all_passive(gl=0.0003)  # soma,dend both have gl
        gl_soma=15e-5
        gl_dend=1e-5
        for sec in self.all:
            sec.cm = 1.0
            sec.insert('pas')
            sec.e_pas = self._vrest
        self.soma.g_pas = gl_soma
        for sec in self.all[1:]:
            sec.g_pas = gl_dend
        h.v_init = self._vrest
    
#    def record_soma_v(self):
#        self.v_rec = Recorder(self.soma(.5),'v')
#    
#    def v(self):
#        """Return recorded soma membrane voltage in numpy array"""
#        if hasattr(self,'v_rec'):
#            return self.v_rec.as_numpy()

class Builder(object):
    def __init__(self,geometry,loc_param=[0.,0.,0.,1.,0.],geo_param=[-1],scale=1.0,ncell=1):
        """
        Initialize simulation object
        geometry: pandas dataframe of cell morphology properties
        electrodes: array of electrode coordinates, n-by-3
        soma_injection: vector of some injection waveform
        loc_param: location parameters, ncell-by-5 array
        geo_param: geometry parameters, ncell-by-k array, if not specified, use default properties in geometry
        scale: scaling factors of lfp magnitude, ncell-vector, if is single value, is constant for all cells
        """
        self.ncell = ncell  # number of cells in this simulation
        self.cells = []  # list of cell object
        self.lfp = []  # list of EcpMod object
        self.define_geometry_entries()  # list of entries to geometry dataframe
        self.geometry = geometry.copy()
        #self.electrodes = electrodes
        #self.soma_injection = soma_injection
        self.set_loc_param(loc_param)  # setup variable location parameters
        self.set_geo_param(geo_param)  # setup variable geometry parameters
        self.set_scale(scale)  # setup scaling factors of lfp magnitude
        self.create_cells()  # create cell objects with properties set up
        self.t_vec = h.Vector( round(h.tstop/h.dt)+1 ).record(h._ref_t)  # record time
    
    def pack_parameters(self,param,ndim,param_name):
        """Pack parameters for the simulation"""
        if ndim==0:
            if not hasattr(param,'__len__'):
                param = [param]
            param = np.array(param).ravel()
            if param.size!=self.ncell:
                if param.size==1:
                    param = np.broadcast_to(param,self.ncell)
                else:
                    raise ValueError(param_name+" size does not match ncell")   
        if ndim==1:
            param = np.array(param)
            if param.ndim==1:
                param = np.expand_dims(param,0)
            if param.shape[0]!=self.ncell:
                if param.shape[0]==1:
                    param = np.broadcast_to(param,(self.ncell,param.shape[1]))
                else:
                    raise ValueError(param_name+" number of rows does not match ncell")
        return param
    
    def set_loc_param(self,loc_param):
        """Setup location parameters. loc_param ncell-by-5 array"""
        loc_param = self.pack_parameters(loc_param,1,"loc_param")
        self.loc_param = [(np.insert(loc_param[i,:2],2,0.),loc_param[i,2:]) for i in range(self.ncell)]
    
    def set_geo_param(self,geo_param):
        """Setup geometry parameters. geo_param ncell-by-k array, k entries of properties"""
        self.geo_param = self.pack_parameters(geo_param,1,"geo_param")
    
    def set_scale(self,scale):
        """setup scaling factors of lfp magnitude"""
        self.scale = self.pack_parameters(scale,0,"scale")
    
    def define_geometry_entries(self):
        """Define list of entries to geometry dataframe. Each entry is a pair of section id and property."""
        self.geo_entries = [
            (0,'R'),  # change soma radius
            (3,'L'),  # change trunk length
            (3,'R'),  # change trunk radius
            ([1,2],'R'),  # change dendrites radius
            (4,'R'),  # change tuft radius
            ([1,2,4],'L') # change dendrite length
        ]
    
    def set_geometry(self,geometry,geo_param):
        """Set property values from geo_param through each entry to geometry. Return dataframe"""
        geom = geometry.copy()
        for i,x in enumerate(geo_param):
            if x>=0:
                geom.loc[self.geo_entries[i]] = x
        return geom
    
    def create_cells(self):
        """Create cell objects with properties set up"""
        self.cells.clear()  # remove cell objects from previous run
        self.lfp.clear()
        for i in range(self.ncell):
            geometry = self.set_geometry(self.geometry,self.geo_param[i,:])
            self.cells.append( Cell(geometry=geometry) )