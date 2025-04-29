from neuron import h
def load_hay_cell(conn_type_settings):
    for cell_type,items in conn_type_settings.items():
        conn_type_settings[cell_type]['spec_settings']['post_cell'] = 'L5PCtemplate'
        conn_type_settings[cell_type]['spec_settings']['sec_id'] = 0 # 0 will be soma; 1 would be a basal dendrite


    # hoc_files_to_load = ['stdrun.hoc', "../../../Neural-Modeling/cells/templates/L5PCbiophys3.hoc", 'import3d.hoc', "../../../Neural-Modeling/cells/templates/L5PCtemplate.hoc"]

    # needed for h.load_file("import3d.hoc")
    h.load_file('stdrun.hoc')

    # load procedure L5PCbiophys() for distributing biophys in h.L5PCtemplate()
    # h.load_file("../../../Neural-Modeling/cells/templates/L5PCbiophys3.hoc") # cannot be loaded without loading mechanisms first
    h.load_file("../../../../cells/templates/L5PCbiophys3.hoc")

		# # load needed procedure for importing 3d coordinates
    h.load_file("import3d.hoc")

		# # # Load h.L5PCtemplate()
    # h.load_file("../../../Neural-Modeling/cells/templates/L5PCtemplateMediumRes.hoc") # load template that gets biophys and establishes sectioning
    h.load_file("../../../../cells/templates/L5PCtemplateMediumRes.hoc") 
    # called 'MediumRes' because semgentation is reverted back to original

    # path to 3d coords file that will be passed to "cell = h.L5PCtemplate(template_arg)" in SynapseTuner.set_up_cell(self)
    # template_arg = "../../../Neural-Modeling/cells/templates/cell1.asc" # contains 3d coordinates
    template_arg = "../../../../cells/templates/cell1.asc" # contains 3d coordinates
    return template_arg

from bmtool.synapses import SynapseTuner
def InitializeSysnapseTuner(connection, template_arg, current_name='i', other_vars_to_record=[], sliders_to_use=['initW']):
  mechanisms_dir = 'modfiles'
  templates_file = 'templates.hoc'

  tuner = SynapseTuner(mechanisms_dir=mechanisms_dir, # where x86_64 is located
                      templates_dir=templates_file, # where the neuron templates are located
                      conn_type_settings=conn_type_settings, # dict of connection settings
                      general_settings = general_settings, # dict of general settings
                      connection = connection, # key in connection settings for which connection you want to tune
                      #json_folder_path=json_folder_path, # If your network uses json files the path can be set to update the connection settings based on the keys and values in the json
                      current_name = current_name, # name of current variable in synapase
                      other_vars_to_record = other_vars_to_record, # Other synaptic variables you wish to record besides the normal ones
                      slider_vars=sliders_to_use,
                       template_arg=template_arg) # Range variables you want to tune to adjust synaptic response.
  return tuner

general_settings = {
    'vclamp': False, # if vclamp should start on or off used mostly for singleEventv
    'rise_interval': (0.1, 0.9), #10-90%
    'tstart': 100.,#200., #500., # when the singleEvent should start
    'tdur': 25.,#100.,    # Dur of sim after single synaptic event has occured
    'threshold': -15., #threshold for spike in mV
    'delay': 1.3, # netcon delay
    'weight': 1., # netcon weight
    'dt': 0.025, # simulation dt (ms)
    'celsius': 20 # temp of sim
}

conn_type_settings = {
    'blank': {
        'spec_settings': {

        },
        'spec_syn_param': {

        }
    },
    # -------------- Ziao Synapses --------------
        'LTS': { # inhibitory perisomatic
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "GABA_AB_STP",
        },
        'spec_syn_param': {
                'e_GABAA': -75.0,#-90.,
              'Use': 0.3,
              'Dep': 25.,
              'Fac': 100.
        }
    },
    'FSI': { # inhibitory perisomatic
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "GABA_AB_STP",
        },
        'spec_syn_param': {
                'e_GABAA': -75.0,#-90.,
                'Use': 0.3,
                'Dep': 400.,
                'Fac': 0.
        }
    },
    'CS2CP': { # excitatory choice
        'spec_settings': {
                                   'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "AMPA_NMDA_STP",
        },
        'spec_syn_param': {
              'tau_d_AMPA': 5.2,
              'Use': 0.41,
              'Dep': 532.,
              'Fac': 65.
        }
    },
    'CP2CP': { # excitatory choice
        'spec_settings': {
                                   'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "AMPA_NMDA_STP",
        },
        'spec_syn_param': {
                'tau_d_AMPA': 5.2,
                'Use': 0.37,
                'Dep': 31.7,
                'Fac': 519.
        }
    },
    # -------------- BEN SYNAPSES -------------- https://github.com/latimerb/L5NeuronSimulation/tree/master/L5NeuronSimulation/biophys_components/synaptic_models
    'PN2PN': { # exc
        'spec_settings': {
                        'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "pyr2pyrSUM",
        },
        'spec_syn_param': {
                "AlphaTmax_ampa": 5,
                "Beta_ampa": 0.5882,
                "Cdur_ampa": .2,
                "gbar_ampa": 0.001592,# :after tuning; intially: 0.001,
                "Erev_ampa": 0,
                "AlphaTmax_nmda": 3.4483,
                "Beta_nmda": 0.0233,
                "Cdur_nmda": 0.29,
                "gbar_nmda": 0.001868, #:after tuning; initially: 0.0005,
                "Erev_nmda": 0,
                "initW": .4375, #.4375 mean computed from lognormal distribution # default listed in a file was: 5,
                # "delay": 0.9,
                "tauD1": 35,
                "d1": 0.95,
                "tauD2": 250,
                "d2": 0.8,
                "tauF": 1,
                "f": 1
        }
    },
    'PV2PN': { # inhibitory perisomatic
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "int2pyr",
        },
        'spec_syn_param' : {
            "AlphaTmax_gaba": 1.52,
            "Beta_gaba": 0.14,
            "Cdur_gaba": 0.7254,
            "gbar_gaba": 0.05, #1,
            "Erev_gaba": -75,
            "initW": 1.323877, # default in ben file:1,
            "Wmax": 3,
            "Wmin": 0.25,
            # "delay": 2,
            # "con_pattern": 1,
            "lambda1": 1,
            "lambda2": 0.01,
            "threshold1": 0.5,
            "threshold2": 0.6,
            "tauD1": 51.339317,#40, #12.231593, #5.123180, #40,
            "d1": 1.275173,#0.7, #0.854166, #0.829285, #0.7,
            "tauD2":  539.965797,#500, #3603.370825, #720,#500,
            "d2": 0.763990,#0.7,#0.866754, #0.831071,#0.7,
            "tauF": 1,
            "f": 1
        }
    },
    'SOM2PN': { # inh dendritic
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "int2pyr",
        },
        'spec_syn_param': {
            "AlphaTmax_gaba": 1.52,
            "Beta_gaba": 0.14,
            "Cdur_gaba": 0.7254,
            "gbar_gaba": 0.006,
            "Erev_gaba": -75,
            "initW": 1.40353,
            "Wmax": 3,
            "Wmin": 0.25,
            # "delay": 2,
            # "con_pattern": 1,
            "lambda1": 1,
            "lambda2": 0.01,
            "threshold1": 0.5,
            "threshold2": 0.6,
            "tauD1": 251.671893,#200,
            "d1": 0.892762,#0.8,
            "tauD2": 1,
            "d2": 1,
            "tauF": 1,
            "f": 1
        }
    },
    # --------------- Greg's -----------------------
    'Fac2FSI': { # facilitating synapse
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp' : -70.,
            'sec_x': 0.5,
            'sec_id': 1,
            "level_of_detail": "AMPA_NMDA_STP",
        },
        'spec_syn_param': {
            'initW': 0.76,
            'tau_r_AMPA': 0.45,
            'tau_d_AMPA': 7.5,
            'Use': 0.13,
            'Dep': 0.,
            'Fac': 200.
        },
    },
    'Dep2FSI': { # depressing synapse
        'spec_settings': {
            'post_cell': 'FSI_Cell',
            'vclamp_amp': -55,
            'sec_x': 0.5,
            'sec_id':0,
            "level_of_detail": "GABA_A_STP",
        },
        'spec_syn_param': {
            'initW': 20,
            'tau_r_GABAA': 0.9,
            'tau_d_GABAA': 15,
            'e_GABAA':-75,
            'Use': 0.4,
            'Dep': 190.,
            'Fac': 0.
        },
    },

}

# define all of your tuner “recipes” in a single dict
tuner_configs = {
    True: {
        'tunerForEXC': {
            'connection': 'PN2PN',
            'current_name': 'i',
            'sliders_to_use': ['initW', 'gbar_nmda', 'gbar_ampa'],
        },
        'tunerForInhPerisomatic': {
            'connection': 'PV2PN',
            'current_name': 'igaba',
            'sliders_to_use': [
                'initW','gbar_gaba','lambda1','lambda2',
                'threshold1','threshold2','tauD1','d1',
                'tauD2','d2','tauF','f'
            ],
        },
        'tunerForInhDendritic': {
            'connection': 'SOM2PN',
            'current_name': 'igaba',
            'sliders_to_use': [
                'initW','gbar_gaba','lambda1','lambda2',
                'threshold1','threshold2','tauD1','d1',
                'tauD2','d2','tauF','f'
            ],
        },
    },
    False: {
        'tunerForEXC': {
            'connection': 'CS2CP',
            'current_name': 'i',
            'sliders_to_use': ['initW', 'tau_d_AMPA', 'Use', 'Dep', 'Fac'],
        },
        'tunerForEXC2': {
            'connection': 'CP2CP',
            'current_name': 'i',
            'sliders_to_use': ['initW', 'tau_d_AMPA', 'Use', 'Dep', 'Fac'],
        },
        'tunerForInhPerisomatic': {
            'connection': 'LTS',
            'current_name': 'i',
            'sliders_to_use': ['initW', 'tau_d_GABAA', 'Use', 'Dep', 'Fac'],
        },
        'tunerForInhDendritic': {
            'connection': 'FSI',
            'current_name': 'i',
            'sliders_to_use': ['initW', 'tau_d_GABAA', 'Use', 'Dep', 'Fac'],
        },
    }
}

target_metrics = { # automatically selects based on tuner_type (need to be updated to match in vivo)
    'exc': {
            'induction': -0.44, #somewhere around -0.425-0.45@20Hz#-0.6@50Hz, # from -0.75
            'ppr': 0.8333, # from 0.8
            'recovery': 0.0,
            'magnitude':{ #pA
                      'mean': 30.6,
                      'std': 29.9
                              }
    },
    'inhPerisomatic': {
            'induction': -0.55, #somewhere between -0.5 and -0.575 @20Hz. somewhere between -0.5 and -0.575 #-0.7 @50Hz, # from -0.75
            'ppr': 0.8666,# from 0.8
            'recovery': 0.0,
            'magnitude':{ #pA
                      'mean': 208.3,
                      'std': 58.7
                              }
    },
    'inhDendritic': {
            'induction': -0.275,# somewhere between -0.3 and -0.2 @20Hz. #-0.4 @50Hz, # from -0.75
            'ppr': 0.9, # from 0.8
            'recovery': 0.0,
            'magnitude':{ #pA
                      'mean': 26.5,
                      'std': 1.6
                              }
    }
}

location_types_by_synapse_type = {
    'inhPerisomatic': ['perisomatic'],
    'exc': ['distal_basal', 'distal_apic'],
    'inhDendritic': ['distal_basal', 'distal_apic']
}

import numpy as np
def log_norm_dist(gmax_mean, gmax_std, size, clip, gmax_scalar): # exc
  val = np.random.lognormal(gmax_mean, gmax_std, size)[0]
  # print(val)
  # print(np.clip(val, clip[0], clip[1]))
  # print(float(np.clip(val, clip[0], clip[1])))
  s = np.clip(val, clip[0], clip[1])
  s = gmax_scalar * float(np.clip(val, clip[0], clip[1]))
  return s

def norm_dist(gmax_mean, gmax_std, size, clip): # inh
  val = np.random.normal(gmax_mean, gmax_std, size)[0]
  s = np.clip(val, clip[0], clip[1])
  return s

distributions_to_test = {
    'inhPerisomatic': {
        'perisomatic': {
          'mean': 1.324*2*1.5*1.16*1.5*1.16,#1.323877*3,
          'std': 0.0#0.373*1.25*0.75*0.5*0.5*0.33*0.33*0.25*0.25*0.25*0.25*0.25#0.3730754,
        }
    },
    'inhDendritic': {
        'distal_basal': {
          'mean':1.4035*2.2*1.36*1.25*1.2*1.05,#1.4035*1.4,# 1.40353*1.5,
          'std': 0.0#0.08474*0.916*0.5*.16*0.25*0.2*0.1*0.1 #0.0847416/8
        },
        'distal_apic': {
          'mean':1.40353*1.25*1.065,#1.4035*10,# 1.40353*1.5
          'std': 0.0#0.08474*0.2*0.66*0.1*0.25*0.1*0.1*0.1#0.08474*4#0.0847416/8
        }
    },
    'exc': {
        'distal_basal': {
          'mean': 0.44*0.9*1,#0.45,
          'std': 0.0,#0.43*0.9*1.38*1.3*1.5*1.5*1.2*1.33*1.2*1.33,#0.35,
          'exc_scalar': 1#1.6
        },
        'distal_apic': {
          'mean': 0.44*1.003*0.95,#0.45,
          'std': 0.0,#0.43*1.2*1.25*1.5*1.5*1.2*1.2*1.33,#0.35,
          'exc_scalar': 1#1.05
        }
    }
}