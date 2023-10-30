import numpy as np
from neuron import h
from neuron import nrn

class CurrentInjection:
    
    def __init__(self, segment: nrn.Segment, pulse: bool = True, pulse_params: dict = None, 
                 current: np.ndarray = None, dt: np.ndarray = None, record: bool = False):
        """
        Parameters:
        ----------
        segment: nrn.Segment
            Target segment.

        pulse: bool
            If True, use pulse injection with keyword arguments in 'pulse_param'
            If False, use waveform resources in vector 'current' as injection

        current: np.ndarray
            ..

        dt: np.ndarray
            Array of time steps.

        record: bool
            If True, enable recording current injection history
        """
        self.segment = segment
        self.neuron_iclamp_object = h.IClamp(self.segment)

        self.inj_vec = None

        if pulse:
            self.setup_pulse(pulse_params)
        else:
            if current is None: current = [0]
            self.setup_current(current, dt)

        self.setup(record)

    def setup_pulse(self, pulse_params: dict) -> None:
        """
        Set IClamp attributes. 
        
        Parameters:
        ----------
        pulse_params: dict
            Parameters to set: {attribute_name: attribute_value}.
        """
        for param, value in pulse_params.items():
            setattr(self.neuron_iclamp_object, param, value)

    def setup(self, record: bool = False) -> None:
        if record: self.setup_recorder()

    #TODO: fix docstring
    def setup_current(self, current: np.ndarray = None, dt: np.ndarray = None) -> None:
        """
        Set current injection with the waveform in the current vector.

        Parameters:
        ----------
        current: np.ndarray
            ...
        
        dt: np.ndarray
            ...
        
        """
        # TODO: Does it make sense to create a copy? 
        ccl = self.neuron_iclamp_object
        ccl.dur = h.tstop

        if dt is None:
            dt = h.dt

        self.inj_vec = h.Vector()
        self.inj_vec.from_python(current)
        self.inj_vec.append(0)
        self.inj_vec.play(ccl._ref_amp, dt)

    def setup_recorder(self):
        size = [round(h.tstop / h.dt) + 1]
        self.rec_vec = h.Vector(*size).record(self.neuron_iclamp_object._ref_i)

    #PRAGMA MARK: Utility

    def get_section(self) -> h.Section:
        return self.neuron_iclamp_object.get_segment().sec

    def get_segment(self):
        return self.neuron_iclamp_object.get_segment()

class Synapse:

    def __init__(self, segment: nrn.Segment = None, syn_mod: str = None, gmax: float = None, record: bool = False,
                 syn_params: dict = None, syn_obj: object = None, vector_length: int = None, ncs: list = None):
        '''
        Parameters:
        ----------
        segment: nrn.Segment
            Segment to add synapse to.

        syn_mod: str = "Exp2Syn"
            Type of synapse.

        gmax: float = 0.01
            Maximum conductance.

        record: bool = False
            Whether to setup a recorder.

        syn_params: dict = None
            Additional parameter for the synapse.

        syn_obj: object = None
            Optional existing hoc synapse point process object to store

        '''
        if syn_obj:
            self.segment = syn_obj.get_segment()
        elif segment:
            self.segment = segment
        else:
            raise ValueError("Need to pass either existing neuron synapse object or segment to create new synapse.")
        
        self.syn_type = syn_mod if syn_obj is None else str(syn_obj).split('[')[0]
        self.syn_params = syn_params
        self.synapse_hoc_obj = syn_obj
        self.gmax_var = None  # Variable name of maximum conductance (uS)
        self.rec_vec = []  # List of vectors for recording
        self.set_params_based_on_synapse_mod(self.syn_type)
        
        if gmax:
            self.gmax = gmax
        elif syn_obj:
            self.gmax = self.get_gmax_from_syn_obj()
        else:
            raise NotImplementedError("Need to pass either existing neuron synapse object or gmax to create new synapse.")
        
        self.setup(record, vector_length)
        self.ncs = ncs or []  # netcons

    # PRAGMA MARK: new
    def update_netcons(self, netcons): # function for updating netcons
      self.ncs=netcons

    #PRAGMA MARK: Synapse Parameter Setup

    def set_params_based_on_synapse_mod(self, syn_mod: str) -> None:
        syn_params_map = {
            'AlphaSynapse1': ({}, 'gmax'),
            'Exp2Syn': ({'e': 0., 'tau1': 1.0, 'tau2': 3.0}, '_nc_weight'),
            'pyr2pyr': ({}, 'initW'),
            'int2pyr': ({}, 'initW'),
            'AMPA_NMDA': ({}, 'initW'),
            'AMPA_NMDA_STP': ({}, 'initW'),
            'GABA_AB': ({}, 'initW'),
            'GABA_AB_STP': ({}, 'initW')
        }
        
        current_type_map = {
            'AlphaSynapse1': "i",
            'Exp2Syn': "i",
            'GABA_AB': "i",
            'GABA_AB_STP': "i",
            'pyr2pyr': "iampa_inmda",
            'AMPA_NMDA': 'i_AMPA_i_NMDA',
            'AMPA_NMDA_STP': 'i_AMPA_i_NMDA',
            'int2pyr': 'igaba'
        }
    
        # set syn_params and gmax_var based on syn_mod
        if syn_mod in syn_params_map:
            self.syn_params, self.gmax_var = syn_params_map[syn_mod]
        else:
            raise ValueError("Synapse type not defined.")
        
        # set current_type based on syn_mod
        if syn_mod in current_type_map:
            self.current_type = current_type_map[syn_mod]
        else:
            raise ValueError("Synapse type not defined.")
        
        if self.synapse_hoc_obj is None: 
            # create new synapse hoc object if not provided one
            self.synapse_hoc_obj = getattr(h, self.syn_type)(self.segment)


    #PRAGMA MARK: Synapse Value Setup

    def setup(self, record, vector_length) -> None:
        self.setup_synapse()
        if record:
            self.setup_recorder(vector_length)
            
    def set_syn_params(self,syn_params) -> None:
        self.syn_params = syn_params # update syn_params if calling outside init
        for key, value in syn_params.items():
            if callable(value):
                setattr(self.synapse_hoc_obj, key, value(size=1))
            else:
                setattr(self.synapse_hoc_obj, key, value)

    def setup_synapse(self) -> None:
        if self.syn_params is not None:
            self.set_syn_params(self.syn_params)
        self.set_gmax()

    def set_gmax(self, gmax: float = None) -> None:
        if gmax is not None:
            self.gmax = gmax
        if self.gmax_var == '_nc_weight':
            self.nc.weight[0] = self.gmax
        else:
            setattr(self.synapse_hoc_obj, self.gmax_var, self.gmax)

    def get_gmax_from_syn_obj(self) -> None:
        gmax = getattr(self.synapse_hoc_obj, self.gmax_var)
        return gmax
    
    def setup_recorder(self, vector_length) -> None:
        size = vector_length
        
        if self.current_type == "i":
            self.rec_vec.append(h.Vector(size).record(self.synapse_hoc_obj._ref_i))
            
        elif self.current_type == "igaba":
            self.rec_vec.append(h.Vector(size).record(self.synapse_hoc_obj._ref_igaba))

        elif self.current_type == "i_AMPA_i_NMDA":
            vec_inmda = h.Vector(size).record(self.synapse_hoc_obj._ref_i_NMDA)
            vec_iampa = h.Vector(size).record(self.synapse_hoc_obj._ref_i_AMPA)
            self.rec_vec.append(vec_inmda)
            self.rec_vec.append(vec_iampa)
        
        elif self.current_type == "iampa_inmda":
            vec_inmda = h.Vector(size).record(self.synapse_hoc_obj._ref_inmda)
            vec_iampa = h.Vector(size).record(self.synapse_hoc_obj._ref_iampa)
            self.rec_vec.append(vec_inmda)
            self.rec_vec.append(vec_iampa)

        else:
            raise(ValueError('current_type not defined'))

    #PRAGMA MARK: Utility

    def get_section(self) -> h.Section:
        return self.synapse_hoc_obj.get_segment().sec

    def get_segment(self) -> nrn.Segment:
        return self.synapse_hoc_obj.get_segment()
        
    def get_exc_or_inh_from_syn_type(self):
        if 'pyr2pyr' in self.syn_type or 'AMPA_NMDA' in self.syn_type:
            return 'exc'
        elif 'int2pyr' in self.syn_type or 'GABA_AB' in self.syn_type:
            return 'inh'
        else:
            raise(ValueError("Cannot determine 'exc' or 'inh' from syn_type:", self.syn_type))
