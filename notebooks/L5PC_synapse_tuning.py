import sys
import os
import pickle

sys.path.append("../../../")
sys.path.append("../../../bmtool/bmtool/")

import neuron
from neuron import h
import bmtool

save_name = "no_std"

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
# neuron.load_mechanisms('modfiles/') # have to load the mechanisms to load L5PCbiophys3.hoc

use_hay_cell = True # overwrite the target cell
if use_hay_cell:
    for cell_type,items in conn_type_settings.items():
        conn_type_settings[cell_type]['spec_settings']['post_cell'] = 'L5PCtemplate'
        conn_type_settings[cell_type]['spec_settings']['sec_id'] = 0 # 0 will be soma; 1 would be a basal dendrite


    # hoc_files_to_load = ['stdrun.hoc', "../../../Neural-Modeling/cells/templates/L5PCbiophys3.hoc", 'import3d.hoc', "../../../Neural-Modeling/cells/templates/L5PCtemplate.hoc"]

    # needed for h.load_file("import3d.hoc")
    h.load_file('stdrun.hoc')

    # load procedure L5PCbiophys() for distributing biophys in h.L5PCtemplate()
    h.load_file("../../../Neural-Modeling/cells/templates/L5PCbiophys3.hoc") # cannot be loaded without loading mechanisms first

		# # load needed procedure for importing 3d coordinates
    h.load_file("import3d.hoc")

		# # # Load h.L5PCtemplate()
    h.load_file("../../../Neural-Modeling/cells/templates/L5PCtemplateMediumRes.hoc") # load template that gets biophys and establishes sectioning
    # called 'MediumRes' because semgentation is reverted back to original

    # path to 3d coords file that will be passed to "cell = h.L5PCtemplate(template_arg)" in SynapseTuner.set_up_cell(self)
    template_arg = "../../../Neural-Modeling/cells/templates/cell1.asc" # contains 3d coordinates
else:
    template_arg=None
    hoc_files_to_load = None

import subprocess
subprocess.run(["rm", "-rf", "x86_64"])
subprocess.run(["rm", "-rf", "modfiles/x86_64"])

def InitializeSysnapseTuner(connection, current_name='i', other_vars_to_record=[], sliders_to_use=['initW'], template_arg=template_arg):
  from synapses import SynapseTuner
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

ben_synapses = True
if ben_synapses:
  tunerForEXC = InitializeSysnapseTuner(connection='PN2PN',
                                      current_name ='i',
                                      sliders_to_use=['initW', 'gbar_nmda','gbar_ampa'])
  tunerForInhPerisomatic =  InitializeSysnapseTuner(connection='PV2PN',
                                      current_name ='igaba',
                                      sliders_to_use=['initW', 'gbar_gaba', "lambda1", "lambda2", "threshold1", "threshold2", "tauD1", "d1", "tauD2", "d2", "tauF", "f"])
  tunerForInhDendritic =  InitializeSysnapseTuner(connection='SOM2PN',
                                      current_name ='igaba',
                                      sliders_to_use=['initW', 'gbar_gaba', "lambda1", "lambda2", "threshold1", "threshold2", "tauD1", "d1", "tauD2", "d2", "tauF", "f"])
else: # ziao synapses
  tunerForEXC = InitializeSysnapseTuner(connection='CS2CP',
                                      current_name ='i',
                                      sliders_to_use=['initW', 'tau_d_AMPA','Use','Dep','Fac'])
  tunerForEXC2 = InitializeSysnapseTuner(connection='CP2CP',
                                      current_name ='i',
                                      sliders_to_use=['initW', 'tau_d_AMPA','Use','Dep','Fac'])
  tunerForInhPerisomatic =  InitializeSysnapseTuner(connection='LTS',
                                      current_name ='i',
                                      sliders_to_use=['initW', 'tau_d_GABAA','Use','Dep','Fac'])
  tunerForInhDendritic =  InitializeSysnapseTuner(connection='FSI',
                                      current_name ='i',
                                      sliders_to_use=['initW', 'tau_d_GABAA','Use','Dep','Fac'])

synapses = {
      'exc': tunerForEXC,
      'inhPerisomatic': tunerForInhPerisomatic,
      'inhDendritic': tunerForInhDendritic
      }

# could make this into a function that just computes a dictionary so we only call it once and only creating/deleting a cell once.
# Could even pass the cell from the tuner instead of creating/deleting a new one.
def get_sec_ids_from_type(section_type):
  cell = h.L5PCtemplate("../../../Neural-Modeling/cells/templates/cell1.asc")

  if section_type == 'distal_apic': # distal apic (>100 microns from soma)
    sec_ids_to_use = [idx for idx,sec in enumerate(cell.all) if (sec in cell.apic) and (h.distance(cell.soma[0](0.5), sec(0.5)) > 100)]
  elif section_type == 'distal_basal': # distal basal dendrites (>100 microns from soma)
    sec_ids_to_use = [idx for idx,sec in enumerate(cell.all) if (sec in cell.dend) and (h.distance(cell.soma[0](0.5), sec(0.5)) > 100)]
  elif section_type == 'perisomatic': # proximal dendrites and soma  (within 100 microns of soma)
    sec_ids_to_use = [idx for idx,sec in enumerate(cell.all) if ((h.distance(cell.soma[0](0.5), sec(0.5)) < 100) and (sec not in list(cell.axon)))]
  else:
    del cell
    NotImplementedError(f"{section_type} not implemented for get_sec_ids_from_type")

  del cell
  return sec_ids_to_use

# might need to tune distal_apic more specifically

import pandas as pd
import numpy as np
# gather PSC across different synaptic weights (weights only for exc; inh are fixed) & regions
measure_PSCs = True

num_weights_per_loc = 2 # number of tests per segment (although each test will have synapse move to random seg with probability seg_length)

numpy_random_state = 4277176
random_state = np.random.RandomState(numpy_random_state)

exc_mean = (np.log(0.45) - 0.5 * np.log((0.35/0.45)**2+1))
exc_std = np.sqrt(np.log((0.35/0.45)**2 + 1))
exc_clip = (0,5)

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

location_types_by_synapse_type = {
    'inhPerisomatic': ['perisomatic'],
    'exc': ['distal_basal', 'distal_apic'],
    'inhDendritic': ['distal_basal', 'distal_apic']
}

target_metrics = { # automatically selects based on tuner_type (need to be updated to match in vivo)
    'exc': {
            'max_amplitude':{ #pA
                      'mean': 30.6,
                      'std': 29.9
                              }
    },
    'inhPerisomatic': {
            'max_amplitude':{ #pA
                      'mean': 208.3,
                      'std': 58.7
                              }
    },
    'inhDendritic': {
            'max_amplitude':{ #pA
                      'mean': 26.5,
                      'std': 1.6
                              }
    }
}

# ballpark soma approximations
# exc:
#  initW_mean:0.44 init_Wstd:0.43
#  PSC_mean:30.60 PSC_std:29.90 29.90
#  target_PSC_mean:30.600000 target_PSC_std:29.90
#  error_PSC_mean: 0.000000 error_PSC_std:0.00
# inhPerisomatic:
#  initW_mean:1.32 init_Wstd:0.37
#  PSC_mean:208.30 PSC_std:58.70 58.70
#  target_PSC_mean:208.300000 target_PSC_std:58.70
#  error_PSC_mean: -0.000090 error_PSC_std:0.00
# inhDendritic:
#  initW_mean:1.40 init_Wstd:0.08
#  PSC_mean:26.50 PSC_std:1.60 1.60
#  target_PSC_mean:26.500000 target_PSC_std:1.60
#  error_PSC_mean: 0.000050 error_PSC_std:0.00

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

# inhDendritic:(0.16842415731075505, 0.08474158821873527)
# Initialize an empty list to store results
segments = [seg for sec in synapses['exc'].cell.all for seg in sec]
results = []
resulting_PSCs_by_segment = {}
for synapse_type in location_types_by_synapse_type.keys():
  resulting_PSCs_by_segment[synapse_type] = {} # make 1 for each tuner/synapse type
  resulting_PSCs_by_segment[synapse_type]['segments'] = [seg for sec in synapses[synapse_type].cell.all for seg in sec] # list segments for each tuner Segments are the same, but identified differently bc each tuner built its own cell.
  for location_type in location_types_by_synapse_type[synapse_type]: # make one for each location type
    resulting_PSCs_by_segment[synapse_type][location_type] = [[] for seg in resulting_PSCs_by_segment[synapse_type]['segments']] # initialize list of PSCs that were used for each segment

if measure_PSCs:
  # gather distribution of PSC magnitudes for each synapse_type and location types
  for synapse_type, location_types in location_types_by_synapse_type.items():
    tuner = synapses[synapse_type]
    if hasattr(synapses[synapse_type], 'dynamic_sliders'):
      del synapses[synapse_type].dynamic_sliders # this var would replace the values even when we try to update them
    if 'inh' in synapse_type:
      num_weights_per_loc_to_use = num_weights_per_loc
      resample_weight = True
      use_norm_dist = True
    else:
      num_weights_per_loc_to_use = num_weights_per_loc
      resample_weight = True
      use_norm_dist = False

    for location_type in location_types:
      # get the sections we need
      sec_ids_to_use = get_sec_ids_from_type(location_type) # select the type using integar

      if synapse_type == 'exc':
        exc_gmax_scalar = distributions_to_test[synapse_type][location_type]['exc_scalar']

      # gather PSC magnitudes across locations
      magnitudes = []

      possible_segments = [seg for sec_id in sec_ids_to_use for seg in list(tuner.cell.all)[sec_id]]
      seg_probs = [(seg.sec.L / seg.sec.nseg) for seg in possible_segments]

      n_tests = max(len(possible_segments)*num_weights_per_loc, 200)

      for i in range(n_tests):
        # randomly pick segment based on probabilities
        seg_to_place_syn_on = random_state.choice(possible_segments, 1, True, seg_probs / np.sum(seg_probs))[0]

        # add PSC result for this segment
        segment_index = resulting_PSCs_by_segment[synapse_type]['segments'].index(seg_to_place_syn_on)

        # move the synapse to the target location
        tuner.syn.loc(seg_to_place_syn_on)

        for i_weight in range(num_weights_per_loc_to_use):
          if resample_weight:
            if use_norm_dist:
              new_weight = norm_dist(distributions_to_test[synapse_type][location_type]['mean'], distributions_to_test[synapse_type][location_type]['std'], 1, (0, 10*distributions_to_test[synapse_type][location_type]['mean']))
            else:
              new_weight = log_norm_dist(exc_mean, exc_std, 1, exc_clip, exc_gmax_scalar)
            tuner.syn.initW = new_weight

          #record magnitude of PSC
          PSC_mag = max(abs(tuner.SingleEvent(plot_and_print=False))) # NOTE: have to update to return
          magnitudes.append(PSC_mag) # have to fix this line
          resulting_PSCs_by_segment[synapse_type][location_type][segment_index].append(PSC_mag) # track by segment

      # print(f"distributions_to_test {synapse_type}: {distributions_to_test[synapse_type]")

      # calc mean, std
      psc_mean = np.mean(magnitudes)
      psc_std = np.std(magnitudes)
      print(f"{synapse_type} {location_type}")
      print(f" target_metrics: {target_metrics[synapse_type]}")
      print(f" actual: mean:{psc_mean:.2f}, std:{psc_std:.2f}")

      #show error
      print(f" error: mean:{target_metrics[synapse_type]['max_amplitude']['mean'] - psc_mean:.2f} std:{target_metrics[synapse_type]['max_amplitude']['std'] - psc_std:.2f}\n")

      # Store results in the list
      results.append({
          "Synapse Type": (synapse_type),
          "Location Type": location_type,
          "initW_mean": round(distributions_to_test[synapse_type][location_type]['mean'], 3),
          "initW_std": round(distributions_to_test[synapse_type][location_type]['std'], 3),
          "PSC Mean": round(psc_mean, 3),
          "PSC Std": round(psc_std, 3),
          "PSC_mean_error": round(target_metrics[synapse_type]['max_amplitude']['mean'] - psc_mean, 3),
          "PSC_std_error": round(target_metrics[synapse_type]['max_amplitude']['std'] - psc_std, 3),
          "n_tests": n_tests,
      })

# Convert the list to a DataFrame
results_df = pd.DataFrame(results)
# import ace_tools as tools
# tools.display_dataframe_to_user(name="PSC Data", dataframe=results_df)

for synapse_type,psc_data in resulting_PSCs_by_segment.items():
  resulting_PSCs_by_segment[synapse_type]['segments'] = str(psc_data['segments'])

# Save to a file
with open(f'{save_name}_PSCs_by_segment.pkl', 'wb') as f:
    pickle.dump(resulting_PSCs_by_segment, f)

results_df.to_csv(f"{save_name}_PSC_loctype_results.csv")