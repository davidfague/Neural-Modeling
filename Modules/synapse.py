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

    def __init__(self, segment: nrn.Segment, syn_mod: str = 'Exp2Syn', gmax: float = 0.01, record: bool = False, 
                 syn_params: dict = None):
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

        '''
        self.segment = segment
        self.syn_type = syn_mod
        self.gmax = gmax
        self.gmax_var = None # Variable name of maximum conductance (uS)
        self.syn_params = syn_params
        self.synapse_neuron_obj = None
        self.rec_vec = []  # List of vectors for recording

        self.set_params_based_on_synapse_mod(syn_mod)
        self.setup(record)
        self.ncs = []

    #PRAGMA MARK: Synapse Parameter Setup

    def set_params_based_on_synapse_mod(self, syn_mod: str) -> None:
        if syn_mod == 'AlphaSynapse1':
            # Reversal potential (mV); Synapse time constant (ms)
            #TODO: old? # self.syn_params = {'e': 0., 'tau': 2.0}
            self.syn_params = {}
            self.gmax_var = 'gmax'
        elif syn_mod == 'Exp2Syn':
            self.syn_params = {'e': 0., 'tau1': 1.0, 'tau2': 3.0}
            self.gmax_var = '_nc_weight'
        elif syn_mod in ['pyr2pyr', 'int2pyr']: # ampanmda, gaba
            self.syn_params = {}
            self.gmax_var = 'initW'
        elif any(ext in syn_mod for ext in ['AMPA_NMDA', 'GABA_AB']): # ampanmda, gaba
            self.syn_params = {}
            self.gmax_var = 'initW'
        else:
            raise ValueError("Synpase type not defined.")
        
        if syn_mod in ['AlphaSynapse1', 'Exp2Syn']:
            self.current_type = "i"
        elif syn_mod in ['pyr2pyr', 'AMPA_NMDA']:
            self.current_type = "iampa_inmda"
        elif syn_mod in ['int2pyr', 'GABA_AB']:
            self.current_type = 'igaba'
        else:
            raise ValueError
        
        self.synapse_neuron_obj = getattr(h, self.syn_type)(self.segment)

    #PRAGMA MARK: Synapse Value Setup

    def setup(self, record: bool = False) -> None:
        self.setup_synapse()
        if record:
            self.setup_recorder()

    def setup_synapse(self) -> None:
        if self.syn_params is not None:
            for key, value in self.syn_params.items():
                if callable(value):
                    setattr(self.synapse_neuron_obj, key, value(size=1))
                else:
                    setattr(self.synapse_neuron_obj, key, value)
        self.set_gmax()

    def set_gmax(self, gmax: float = None) -> None:
        if gmax is not None:
            self.gmax = gmax
        if self.gmax_var == '_nc_weight':
            self.nc.weight[0] = self.gmax
        else:
            setattr(self.synapse_neuron_obj, self.gmax_var, self.gmax)
    
    def setup_recorder(self) -> None:
        size = [round(h.tstop / h.dt) + 1]
        
        if self.current_type == "i":
            self.rec_vec.append(h.Vector(*size).record(self.synapse_neuron_obj._ref_i))
            
        elif self.current_type == "igaba":
            self.rec_vec.append(h.Vector(*size).record(self.synapse_neuron_obj._ref_igaba))

        elif self.current_type == "iampa_inmda":
            vec_inmda = h.Vector(*size).record(self.synapse_neuron_obj._ref_inmda)
            vec_iampa = h.Vector(*size).record(self.synapse_neuron_obj._ref_iampa)
            self.rec_vec.append(vec_inmda)
            self.rec_vec.append(vec_iampa)

    #PRAGMA MARK: Utility

    def get_section(self) -> h.Section:
        return self.synapse_neuron_obj.get_segment().sec

    def get_segment(self) -> nrn.Segment:
        return self.synapse_neuron_obj.get_segment()
