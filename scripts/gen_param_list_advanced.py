N_workers =1
import sys

# Add paths for module imports
sys.path.append("../")
sys.path.append("../Modules/")

import os, pickle, shutil, itertools
import numpy as np
from neuron import h
from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell, CellBuilder
from Modules.constants import HayParameters
import math


#########################################
# User‐specified selections and options #
#########################################

# Simulation type: choose one of: 'sta', 'fi_ci', 'fi_exc', 'check_synapses', 'tuning'
sim_type = ['sta', 'fi_ci', 'fi_exc', 'check_synapses', 'tuning'][-1]

# Synapse numbers: choose one of: 'density', 'full_number', '1000', '10000'
syn_numbers_to_use = ['density', 'full_number', '1000', '10000'][0]
use_SA_probs = False

# Synapse reduction options: supply a list (even one value works)
syn_reductions_to_use = ['None']
# Example for multiple:
# syn_reductions_to_use = ['None', 'NoMapping', 'Merging', 'MappingMerging']

# Morphology options: supply a list of keys from the morphologies dictionary
morphologies_to_use = ['Complex']
# Example for multiple:
# morphologies_to_use = ['Complex', 'Branches', 'Trees']

# Replace dendrites with current injection options: supply a list of keys from the ci_replacements dictionary
ci_replacements_to_use = ['None']
# Example for multiple:
# ci_replacements_to_use = ['None', 'Basals', '1Basal', 'Tufts', '1Tuft', 'Basals&Tufts']

# Seeds: provide lists of seeds. Use [None] if not applicable.
numpy_random_states = [5000] 
                    #    4444444, 999999999, 77777, 33333333, 
                    #    12346456, 98654721, 54367, 213904, 465982, 
                    #    20184657, 3333, 901, 43656, 888888, 
                    #    12435, 678534, 99835, 7865134, 9812763]
neuron_random_states = [None]

default_inh_syn_properties = HayParameters('').inh_syn_properties  # get a copy of the default
default_exc_syn_properties = HayParameters('').exc_syn_properties

# select_parameters_to_vary: parameters to vary across simulations.
# Each key maps to a dict containing:
#   - "values": a list of values (if length==1, no suffix is added to sim_name)
#   - "sim_name_suffix": a string that will prefix the (optionally rounded) value
select_parameters_to_vary = {
    'rhyth_depth_inh_perisomatic': {
         'values': [0.015], # add more values to vary this parameter if desired
         'sim_name_suffix': 'DepthPeriInh',
         'always_include_suffix': True
    },
    'rhyth_depth_inh_distal': {
         'values': [0.015],  # next [0.1, 0.25] # add more values here if needed
         'sim_name_suffix': 'DepthDistalInh',
         'always_include_suffix': True
    },
    # for L5
    'inh_syn_properties': {
         'values': [ # copy default and override the nested value:
            { **default_inh_syn_properties, 
                'tuft': { **default_inh_syn_properties['tuft'], 'syn_density': 0.22*1.5*1.5/1.5/1.5}, # *0.5 to test seeding
                'nexus': { **default_inh_syn_properties['nexus'],'syn_density': 0.22*1.75},
                'trunk': { **default_inh_syn_properties['trunk'],'syn_density': 0.22*0.25/0.25*0.26},
                'oblique': { **default_inh_syn_properties['oblique'],'syn_density': 0.22*2},
                'distal_basal': { **default_inh_syn_properties['distal_basal'],'syn_density': 0.22*1},
                'perisomatic': { **default_inh_syn_properties['perisomatic'],'syn_density': 0.22*0.75*0.5},
            },
         ],
         'sim_name_suffix': 'InhTuftDensity',
         'nested_keys': ['tuft.syn_density', 'oblique.syn_density', 'nexus.syn_density', 'trunk.syn_density', 'distal_basal.syn_density', 'perisomatic.syn_density'], # give the key to extract the value for simulation naming purposes
         'always_include_suffix': True
    },
    'exc_syn_properties': {
         'values': [ # copy default and override the nested value:
            { **default_exc_syn_properties, 
                # detailed 1.5% rhyth
                # 'tuft': { **default_exc_syn_properties['tuft'], 'syn_density': 2.16*0.45*0.5 },
                # 'oblique': { **default_exc_syn_properties['oblique'],'syn_density': 2.16*0.3*0.5},
                # 'nexus': { **default_exc_syn_properties['nexus'],'syn_density': 2.16*0.05 },
                # 'trunk': { **default_exc_syn_properties['trunk'],'syn_density':2.16*0.25 },
                # 'distal_basal': { **default_exc_syn_properties['distal_basal'],'syn_density':2.16*0.601}

                # # reduced 1.5% rhyth
                # 'tuft': { **default_exc_syn_properties['tuft'], 'syn_density': 2.16*0.45*0.5 }, # 60% decrease for detailed
                # 'oblique': { **default_exc_syn_properties['oblique'],'syn_density': 2.16*0.3*0.5},
                # 'nexus': { **default_exc_syn_properties['nexus'],'syn_density': 2.16*0.05 },
                # 'trunk': { **default_exc_syn_properties['trunk'],'syn_density':2.16*0.25 },
                # 'distal_basal': { **default_exc_syn_properties['distal_basal'],'syn_density':2.16*0.601*0.5} # half for reduced

                # # decreased mean fr stds. # Complex
                'tuft': { **default_exc_syn_properties['tuft'], 'syn_density': 2.16*0.45*0.5 /0.5/0.45}, # 60% decrease for detailed
                'nexus': { **default_exc_syn_properties['nexus'],'syn_density': 2.16*0.05 /0.05},
                'trunk': { **default_exc_syn_properties['trunk'],'syn_density':2.16*0.25 /0.25},
                'oblique': { **default_exc_syn_properties['oblique'],'syn_density': 2.16*0.3*0.5/0.5/0.3},
                'distal_basal': { **default_exc_syn_properties['distal_basal'],'syn_density':2.16*0.601*0.5/0.5/0.601}

                                # # decreased mean fr stds. # reduced
                # 'tuft': { **default_exc_syn_properties['tuft'], 'syn_density': 2.16*0.45*0.5 }, # 60% decrease for detailed
                # # 'oblique': { **default_exc_syn_properties['oblique'],'syn_density': 2.16*0.3*0.5},
                # 'nexus': { **default_exc_syn_properties['nexus'],'syn_density': 2.16*0.05 },
                # 'trunk': { **default_exc_syn_properties['trunk'],'syn_density':2.16*0.25 },
                # 'distal_basal': { **default_exc_syn_properties['distal_basal'],'syn_density':2.16*0.601*0.5*0.95} # 0.9x for reduced
            },
         ],
         'sim_name_suffix': 'ExcTuftDensity',
         'nested_keys': ['tuft.syn_density', 'oblique.syn_density', 'nexus.syn_density', 'trunk.syn_density', 'distal_basal.syn_density'], # give the key to extract the value for simulation naming purposes
         'always_include_suffix': True
    },
    # 'exc_scalar_basal': {
    #     'values': [1],
    #     'sim_name_suffix': 'BasalExcScale'
    # },
    # 'exc_scalar_apical': {
    #     'values': [1],
    #     'sim_name_suffix': 'ApicalExcScale'
    # },
    # 'exc_gmax_clip': {
    #     'values': [(0,3), (0,5)],
    #     'sim_name_suffix': 'ExcClip'
    # },
    # 'inh_dendritic_gmax_dist': {
    #     'values':[
    #         #{
    #         # 'distal_apic': (1.4035*2, 0.08474*2),
    #         # 'distal_basal': (1.4035*1.5, 0.08474)
    #         # },
    #         # {
    #         # 'distal_apic': (1.4035*4, 0.08474*4),
    #         # 'distal_basal': (1.4035*1.5, 0.08474)
    #         # },
    #         {
    #         'distal_apic': (1.4035*10, 0.08474*4),
    #         'distal_basal': (1.4035*1.4, 0.08474)
    #         }],
    #     'sim_name_suffix':
    #         'InhGmax',
    # },

    # ########### FOR L23
    #     'inh_syn_properties': {
    #      'values': [ # copy default and override the nested value:
    #         { **default_inh_syn_properties, 
    #             'apic': { **default_inh_syn_properties['apic'], 'syn_density': 0.22*1},
    #             'dend': { **default_inh_syn_properties['dend'],'syn_density': 0.22*1},
    #             'perisomatic': { **default_inh_syn_properties['perisomatic'],'syn_density': 0.22*1},
    #         },
    #      ],
    #      'sim_name_suffix': 'InhDens',
    #      'nested_keys': ['apic.syn_density', 'dend.syn_density', 'perisomatic.syn_density'], # give the key to extract the value for simulation naming purposes
    #      'always_include_suffix': True
    # },
    # 'exc_syn_properties': {
    #      'values': [ # copy default and override the nested value:
    #         { **default_exc_syn_properties, 
    #             'apic': { **default_exc_syn_properties['apic'], 'syn_density': 2.16*0.75}, # 60% decrease for detailed
    #             'dend': { **default_exc_syn_properties['dend'],'syn_density':2.16*1} # 0.9x for reduced
    #         },
    #      ],
    #      'sim_name_suffix': 'ExcDens',
    #      'nested_keys': ['apic.syn_density', 'dend.syn_density'], # give the key to extract the value for simulation naming purposes
    #      'always_include_suffix': True
    # },

}

#######################################
# Predefined simulation parameter sets
#######################################

# Simulation type parameters
sim_type_params_all = {
    'sta': {  # in vivo simulation with recording currents/conductances
        'h_tstop': 5000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': True,
        'record_all_synapses': True,
        # 'record_spike_trains': True,
        # 'record_synapse_distributions': True 
    },
    'fi_ci': {  # FR/I - ramp current injection
        'h_tstop': 5000,
        'save_every_ms': 5000,
        'all_synapses_off': False,
        'CI_on': True,
        'h_i_duration': 4950,
        'h_i_delay': 50,
    },
    'fi_exc': {  # FR/I - ramps excitatory firing rates
        'h_tstop': 5000,
        'save_every_ms': 5000,
        'all_synapses_off': False,
        'exc_constant_fr': True,
        'h_i_duration': 4950,
        'h_i_delay': 50,
    },
    'check_synapses': {  # short in vivo simulation recording synapse distributions
        'h_tstop': 1000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': False,
        'record_all_synapses': False,
        'record_spike_trains': True,
        'record_synapse_distributions': True
    },
    'tuning': {  # in vivo simulation
        'h_tstop': 5000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': False,
        'record_all_synapses': True,
        'record_spike_trains': True,
        'record_synapse_distributions': True
    },
}
# Select the simulation type parameters for the chosen simulation type.
sim_type_params = sim_type_params_all[sim_type]

# Synapse number definitions
syn_numbers = {
    'density': {'inh_syn_number': 0.22, 'exc_syn_number': 2.16},
    'full_number': {'inh_syn_number': 2650, 'exc_syn_number': 26100},
    '1000': {'inh_syn_number': int(1000 * (2650 / (26100 + 2650))),
             'exc_syn_number': int(1000 * (26100 / (26100 + 2650)))},
    '10000': {'inh_syn_number': int(10000 * (2650 / (26100 + 2650))),
              'exc_syn_number': int(10000 * (26100 / (26100 + 2650)))}
}
# Add synapse number settings to the simulation type parameters.
sim_type_params['exc_use_density'] = syn_numbers_to_use.lower() == 'density'
sim_type_params['inh_use_density'] = syn_numbers_to_use.lower() == 'density'
sim_type_params['inh_syn_number'] = syn_numbers[syn_numbers_to_use]['inh_syn_number']
sim_type_params['exc_syn_number'] = syn_numbers[syn_numbers_to_use]['exc_syn_number']
sim_type_params['use_SA_probs'] = use_SA_probs

# Synapse reduction options dictionary
syn_reductions = {
    'None': {'sim_name_add_suffix': ''},
    'NoMapping': {'sim_name_add_suffix': 'NoMapping', 'synapse_mapping': False},
    'Merging': {'sim_name_add_suffix': 'Merging', 'merge_synapses': True},
    'MappingMerging': {'sim_name_add_suffix': 'MappingMerging', 'synapse_mapping': True, 'merge_synapses': True}
}

# Morphology options dictionary
morphologies = {
    'Complex': {'base_sim_name': 'Complex'},
    'ReduceBranches': {'base_sim_name': 'ReduceBranches', 'reduce_obliques': 2, 'reduce_tufts': 2, 'reduce_basals': 4},
    'ReduceTrees': {'base_sim_name': 'ReduceTrees', 'reduce_apic': 1, 'reduce_basals': 1}
}

# Current-injection (CI) replacement options dictionary
ci_replacements = {
    'None': {'sim_name_add_suffix': ''},
    'Basals': {'sim_name_add_suffix': 'REPBasals', 'num_basal_to_replace_with_CI': 8},
    '1Basal': {'sim_name_add_suffix': 'REP1Basal', 'num_basal_to_replace_with_CI': 1},
    'Tufts': {'sim_name_add_suffix': 'REPTufts', 'num_tuft_to_replace_with_CI': 2},
    '1Tuft': {'sim_name_add_suffix': 'REP1Tuft', 'num_tuft_to_replace_with_CI': 1},
    'Basals&Tufts': {'sim_name_add_suffix': 'REPBasals&Tufts', 'num_basal_to_replace_with_CI': 8, 'num_tuft_to_replace_with_CI': 2}
}

#########################################
# Helper functions for generating sets #
#########################################

def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, tuple):
        # Customize tuple formatting, e.g., join elements with a dash:
        return '-'.join(str(x) for x in value)
    elif isinstance(value, dict):
        # Order keys and format nicely
        return '_'.join(f"{k}-{value[k]}" for k in sorted(value))
    else:
        return str(value)

import itertools

def get_parameter_combinations(param_dict):
    """
    Build combinations from param_dict where values can be scalars or dictionaries.
    If a dictionary is passed and a 'nested_key' is specified in the parameter spec,
    that nested value is used to generate the suffix.
    """
    keys = list(param_dict.keys())
    value_lists = [param_dict[k]['values'] for k in keys]
    combinations = []
    for values in itertools.product(*value_lists):
        combo = {}
        suffix_parts = []
        for key, value in zip(keys, values):
            combo[key] = value
            # Only add suffix if more than one value is provided or forced.
            if len(param_dict[key]['values']) > 1 or param_dict[key].get('always_include_suffix', False):
                # If value is a dict and a nested key is provided, extract that nested value.
                if isinstance(value, dict) and 'nested_keys' in param_dict[key]:
                    suffix_values = []
                    for nested_key in param_dict[key]['nested_keys']:
                        keys_list = nested_key.split('.')
                        nested_value = value
                        for nk in keys_list:
                            nested_value = nested_value[nk]
                        suffix_values.append(format_value(nested_value))
                    formatted = '_'.join(suffix_values)
                else:
                    formatted = format_value(value)
                suffix_parts.append(f"{param_dict[key]['sim_name_suffix']}{formatted}")
        combo['sim_name_suffix'] = '_'.join(suffix_parts) if suffix_parts else ''
        combinations.append(combo)
    return combinations

def get_index_matched_parameter_combinations(param_dict):
    """
    Build index-matched combinations from param_dict where values can be scalars or dictionaries.
    If a dictionary is passed and a 'nested_key' is specified in the parameter spec,
    that nested value is used to generate the suffix.
    """
    keys = list(param_dict.keys())
    max_len = max(len(param_dict[k]['values']) for k in keys)
    
    # Validate: all lists with len > 1 must have the same length.
    for k in keys:
        val_len = len(param_dict[k]['values'])
        if val_len not in [1, max_len]:
            raise ValueError(f"Parameter '{k}' has {val_len} values, but others have {max_len}. All varying parameters must match in length.")
    
    combinations = []
    for i in range(max_len):
        combo = {}
        suffix_parts = []
        for key in keys:
            values = param_dict[key]['values']
            value = values[i] if len(values) > 1 else values[0]
            combo[key] = value
            if len(values) > 1 or param_dict[key].get('always_include_suffix', False):
                if isinstance(value, dict) and 'nested_keys' in param_dict[key]:
                    suffix_values = []
                    for nested_key in param_dict[key]['nested_keys']:
                        keys_list = nested_key.split('.')
                        nested_value = value
                        for nk in keys_list:
                            nested_value = nested_value[nk]
                        suffix_values.append(format_value(nested_value))
                    formatted = '_'.join(suffix_values)
                else:
                    formatted = format_value(value)
                suffix_parts.append(f"{param_dict[key]['sim_name_suffix']}{formatted}")
        combo['sim_name_suffix'] = '_'.join(suffix_parts) if suffix_parts else ''
        combinations.append(combo)
    
    return combinations


def create_parameters(numpy_seed, neuron_seed, common_params, morphology_params,
                      syn_reduction_params, ci_replacement_params, varied_params,
                      amp=None, excFR_increase=None):
    """
    Merge all parameter dictionaries and add extra simulation‐specific values.
    Builds a simulation name that reflects:
      - The morphology base name,
      - The synapse reduction and CI replacement suffixes,
      - The suffix from the varied parameters (if more than one value exists),
      - And seed (and optionally amplitude or excitatory FR increase) information.
    """
    sim_name_parts = []
    # Add simulation type at the beginning.
    sim_name_parts.append(sim_type)
    # Morphology base name
    sim_name_parts.append(morphology_params.get('base_sim_name', ''))
    # Append synapse reduction suffix if defined
    if syn_reduction_params.get('sim_name_add_suffix'):
        sim_name_parts.append(syn_reduction_params['sim_name_add_suffix'])
    # Append CI replacement suffix if defined
    if ci_replacement_params.get('sim_name_add_suffix'):
        sim_name_parts.append(ci_replacement_params['sim_name_add_suffix'])
    # Append the varied parameter suffix (if any)
    if varied_params.get('sim_name_suffix'):
        sim_name_parts.append(varied_params['sim_name_suffix'])
    # Append seed information
    sim_name_parts.append(f"Np{numpy_seed}")
    if neuron_seed is not None:
        sim_name_parts.append(f"Neu{neuron_seed}")
    # Append amplitude or excitatory FR increase if provided
    if amp is not None:
        sim_name_parts.append(f"amp{round(amp, 1)}")
    if excFR_increase is not None:
        sim_name_parts.append(f"EXCinc{round(excFR_increase, 1)}")
        
    sim_name = '_'.join([part for part in sim_name_parts if part])
    
    # Merge dictionaries (later entries override earlier ones if keys conflict)
    params = {}
    params.update(common_params)
    params.update(morphology_params)
    params.update(syn_reduction_params)
    params.update(ci_replacement_params)
    params.update(varied_params)
    params['sim_name'] = sim_name
    params['numpy_random_state'] = numpy_seed
    params['morphology_name'] = morphology_params.get('base_sim_name', 'no_morphology_name_provided')
    params['sim_type'] = sim_type
    if neuron_seed is not None:
        params['neuron_random_state'] = neuron_seed
    if amp is not None:
        params['h_i_amplitude'] = round(amp, 1)
    if excFR_increase is not None:
        params['excFR_increase'] = round(excFR_increase, 1)
    
    # Filter to only the keys that HayParameters accepts.
    valid_keys = HayParameters.__init__.__code__.co_varnames
    valid_params = {k: v for k, v in params.items() if k in valid_keys}

    print(f"generating simulation with params: {valid_params}\n\n")
    
    return HayParameters(**valid_params)

def generate_simulations(neuron_random_states, numpy_random_states, select_params, common_params):
    """
    Generates a list of HayParameters objects by iterating over:
      - All numpy and neuron seeds,
      - All choices in morphologies_to_use, syn_reductions_to_use, and ci_replacements_to_use,
      - All combinations of the varied parameters from select_parameters_to_vary, and
      - Simulation-specific loops (e.g. varying current injection amplitude or excitatory FR increase).
    """
    all_parameters = []
    
    # Iterate over seeds and the multiple user-specified options.
    for numpy_seed in numpy_random_states:
        for neuron_seed in neuron_random_states:
            for morph_choice in morphologies_to_use:
                for syn_red_choice in syn_reductions_to_use:
                    for ci_choice in ci_replacements_to_use:
                        morphology_params = morphologies[morph_choice]
                        syn_reduction_params = syn_reductions[syn_red_choice]
                        ci_replacement_params = ci_replacements[ci_choice]
                        
                        # Get all combinations of varied parameters.
                        # varied_params_list = get_parameter_combinations(select_params)
                        varied_params_list = get_index_matched_parameter_combinations(select_params)
                        
                        for varied in varied_params_list:
                            # If the simulation uses current injection, iterate over amplitudes.
                            if 'CI_on' in common_params and common_params['CI_on']:
                                for amp in np.arange(0, 2.1, 0.5):
                                    param_obj = create_parameters(numpy_seed, neuron_seed, common_params,
                                                                  morphology_params, syn_reduction_params,
                                                                  ci_replacement_params, varied, amp=amp)
                                    all_parameters.append(param_obj)
                            # If the simulation uses a constant excitatory FR, iterate over increases.
                            elif 'exc_constant_fr' in common_params and common_params['exc_constant_fr']:
                                for excFR_increase in np.arange(0, 8.1, 2):
                                    param_obj = create_parameters(numpy_seed, neuron_seed, common_params,
                                                                  morphology_params, syn_reduction_params,
                                                                  ci_replacement_params, varied, excFR_increase=excFR_increase)
                                    all_parameters.append(param_obj)
                            else:
                                param_obj = create_parameters(numpy_seed, neuron_seed, common_params,
                                                              morphology_params, syn_reduction_params,
                                                              ci_replacement_params, varied)
                                all_parameters.append(param_obj)
    
    return all_parameters

#########################
# Main execution script #
#########################

if __name__ == "__main__":
    # Generate simulation parameter objects.
    all_parameters = generate_simulations(neuron_random_states, numpy_random_states,
                                          select_parameters_to_vary, sim_type_params)
    
    # Self-check: Warn if the generated number of parameter sets differs from N_workers.
    if len(all_parameters) != N_workers:
        print(f"Warning: Generated simulation parameter sets ({len(all_parameters)}) do not match N_workers ({N_workers}).")
    
    # Create (or recreate) a directory to store the pickled parameter objects.
    params_dir = "params"
    if os.path.exists(params_dir):
        shutil.rmtree(params_dir)
    os.mkdir(params_dir)
    
    # Save each simulation parameter object into its own pickle file.
    for pid, param_obj in enumerate(all_parameters):
        with open(os.path.join(params_dir, f"{pid}.pickle"), 'wb') as file:
            pickle.dump(param_obj, file)
    
    print(f"Generated {len(all_parameters)} simulation parameter sets.")
    # print(f"all_parameters: {all_parameters}")
