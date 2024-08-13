from neuron import h

Exp2Syn_syn_params = {
    'e': 0., 
    'tau1': 1.0, 
    'tau2': 3.0
}

# For Segev synapse modfiles
# Inh distal
LTS_syn_params = { # for GABA_AB_STP.mod
    'e_GABAA': -90.,
    'Use': 0.3,
    'Dep': 25.,
    'Fac': 100.
}
# Inh perisomatic
FSI_syn_params = { # for GABA_AB_STP.mod
    'e_GABAA': -90.,
    'Use': 0.3,
    'Dep': 400.,
    'Fac': 0.
}
# Exc choice of two:
CS2CP_syn_params = { # for AMPA_NMDA_STP.mod
    'tau_d_AMPA': 5.2,
    'Use': 0.41,
    'Dep': 532.,
    'Fac': 65.
}
CP2CP_syn_params = { # for AMPA_NMDA_STP.mod
    'tau_d_AMPA': 5.2,
    'Use': 0.37,
    'Dep': 31.7,
    'Fac': 519.
}

# For Ben's synapse modfiles
# Inh perisomatic
PV2PN_syn_params = { # for int2pyr.mod
    # "level_of_detail": "int2pyr",
    "AlphaTmax_gaba": 1.52,
    "Beta_gaba": 0.14,
    "Cdur_gaba": 0.7254,
    "gbar_gaba": 1,
    "Erev_gaba": -75,
    "initW": 1,
    "Wmax": 3,
    "Wmin": 0.25,
    "delay": 2,
    "con_pattern": 1,
    "lambda1": 1,
    "lambda2": 0.01,
    "threshold1": 0.5,
    "threshold2": 0.6,
    "tauD1": 40,
    "d1": 0.7,
    "tauD2": 500,
    "d2": 0.7,
    "tauF": 1,
    "f": 1
}
# Inh dendritic
SOM2PN_syn_params = { # for int2pyr.mod
    # "level_of_detail": "int2pyr",
    "AlphaTmax_gaba": 1.52,
    "Beta_gaba": 0.14,
    "Cdur_gaba": 0.7254,
    "gbar_gaba": 0.006,
    "Erev_gaba": -75,
    "initW": 1,
    "Wmax": 3,
    "Wmin": 0.25,
    "delay": 2,
    "con_pattern": 1,
    "lambda1": 1,
    "lambda2": 0.01,
    "threshold1": 0.5,
    "threshold2": 0.6,
    "tauD1": 200,
    "d1": 0.8,
    "tauD2": 1,
    "d2": 1,
    "tauF": 1,
    "f": 1
}
# Exc
PN2PN_syn_params = { # for pyr2pyr.mod
    # "level_of_detail": "pyr2pyr",
    "AlphaTmax_ampa": 5,
    "Beta_ampa": 0.5882,
    "Cdur_ampa": .2,
    "gbar_ampa": 0.001,
    "Erev_ampa": 0,
    "AlphaTmax_nmda": 3.4483,
    "Beta_nmda": 0.0233,
    "Cdur_nmda": 0.29,
    "gbar_nmda": 0.0005,
    "Erev_nmda": 0,
    "initW": 5,
    "delay": 0.9,
    "tauD1": 35,
    "d1": 0.95,
    "tauD2": 250,
    "d2": 0.8,
    "tauF": 1,
    "f": 1
}


class Synapse:

    def __init__(self, segment, syn_mod, syn_params, gmax, neuron_r, name, hoc_syn=None):
        '''providing hoc_syn creates a python Synapse object from the existing hoc object; otherwise, create a new hoc object'''
        
        self.gmax_var = None # Variable name of maximum conductance (uS)
        self.current_type = None        
        self.syn_params = None
        self.gmax_val = None
        self.random_generator = None
        
        if hoc_syn != None:
          self.h_syn = hoc_syn
          self.syn_mod = str(hoc_syn).split('[')[0]
          self.set_gmax_var_and_current_type_based_on_syn_mod(self.syn_mod)
        else:
          self.syn_mod = syn_mod
          self.h_syn = getattr(h, self.syn_mod)(segment)
          self.set_gmax_var_and_current_type_based_on_syn_mod(syn_mod)
          self.set_gmax_val(gmax)
          self.set_syn_params(syn_params)
          self.set_random_generator(neuron_r)

        self.name = name

        # Presynaptic cell
        self.pc = None
        self.netcons = []
        
    
    
    def set_spike_train_for_pc(self, mean_fr, spike_train):
        self.pc.set_spike_train(mean_fr, spike_train)
        nc = h.NetCon(self.pc.vecstim, self.h_syn, 1, 0, 1)
        self.netcons.append(nc)

    def set_random_generator(self, r: h.Random) -> None:				 
        if self.syn_mod in ['pyr2pyr', 'int2pyr']:
            r.uniform(0, 1)
            self.h_syn.setRandObjRef(r)
            self.random_generator = r

    def set_syn_params(self, syn_params) -> None:
        self.syn_params = syn_params
        for key, value in syn_params.items():
            if key in ['delay', 'con_pattern', 'initW']: # these variables are parameters that do not get set to the h_syn: "delay' is not a defined hoc variable name.""
                continue # initW from the syn params is not used.
            elif key in ['Wmax', 'Wmin']: # bound the plastic weight around its original value
                if self.gmax_var == 'initW':
                    setattr(self.h_syn, key, value * self.gmax_val)
                else:
                    raise(f"gmax_var must be 'initw' for syn_param {key}")
            elif callable(value): # set syn params
                setattr(self.h_syn, key, value(size = 1))
            else:
                setattr(self.h_syn, key, value)

    def set_gmax_val(self, gmax: float) -> None:
        self.gmax_val = gmax
        setattr(self.h_syn, self.gmax_var, self.gmax_val)

    def set_gmax_var_and_current_type_based_on_syn_mod(self, syn_mod: str) -> None:

        syn_params_map = {
            'AlphaSynapse1': 'gmax',
            'Exp2Syn': '_nc_weight',
            'pyr2pyr': 'initW',
            'int2pyr': 'initW',
            'AMPA_NMDA': 'initW',
            'AMPA_NMDA_STP': 'initW',
            'GABA_AB': 'initW',
            'GABA_AB_STP': 'initW'
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
    
        # Set gmax_var based on syn_mod
        if syn_mod in syn_params_map:
            self.gmax_var = syn_params_map[syn_mod]
        else:
            raise ValueError("Synapse type not defined.")
        
        # Set current_type based on syn_mod
        if syn_mod in current_type_map:
            self.current_type = current_type_map[syn_mod]
        else:
            raise ValueError("Synapse type not defined.")