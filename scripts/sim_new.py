import sys
import os

# Add paths for module imports
sys.path.append("../")
sys.path.append("../Modules/")

import numpy as np
from neuron import h
from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell, CellBuilder
from Modules.constants import HayParameters
import math

# Configuration parameters and options
synapse_keys = ['None']  # Options: 'None', 'NoMapping', 'MappingMerging', etc.
use_SA_probs = True
syn_numbers_to_use = 'Full'  # Options: '1000', 'Full', etc.
common_attributes_to_use = 'sta'  # Options: 'sta', 'FI', 'FI_ExcFR'
morphology_keys = ['Complex']  # Options: 'Complex', 'Branches', 'Trees'
replace_w_CI_keys = ['None']  # Options: 'None', 'Tufts', 'Basals&Tufts', etc.
numpy_random_states = [5]  # Add more seeds if needed
neuron_random_states = None
sim_title = 'TuningSynapses_reduceNA_shiftExcGmaxBy20Percent'

syn_numbers = {
    'Density': {'inh': None, 'exc': None},
    'Full': {'inh': 2650, 'exc': 26100},
    '1000': {'inh': int(1000 * (2650 / (26100 + 2650))), 'exc': int(1000 * (26100 / (26100 + 2650)))},
    '10000': {'inh': int(10000 * (2650 / (26100 + 2650))), 'exc': int(10000 * (26100 / (26100 + 2650)))}
}

# Define the template for common attributes
common_attributes = {
    'sta': {
        'h_tstop': 5000,
        'merge_synapses': False,
        'record_ecp': True,
        'record_all_channels': True,
        'record_all_synapses': True,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs
    },
    'FI': {
        'h_tstop': 5000,
        'save_every_ms': 5000,
        'all_synapses_off': False,
        'CI_on': True,
        'h_i_duration': 4950,
        'h_i_delay': 50,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs
    },
    'FI_ExcFR': {
        'h_tstop': 5000,
        'save_every_ms': 5000,
        'all_synapses_off': False,
        'exc_constant_fr': True,
        'h_i_duration': 4950,
        'h_i_delay': 50,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs
    }
}

morphology_attributes = {
    'Complex': {'base_sim_name': 'Complex'},
    'Branches': {'base_sim_name': 'Branches', 'reduce_obliques': True, 'reduce_tufts': True, 'reduce_basals': 3},
    'Trees': {'base_sim_name': 'Trees', 'reduce_apic': True, 'reduce_basals': 1}
}

replace_w_CI_attributes = {
    'None': {'sim_name_add_suffix': ''},
    'Basals': {'sim_name_add_suffix': 'REPBasals', 'num_basal_to_replace_with_CI': 8},
    '1Basal': {'sim_name_add_suffix': 'REP1Basal', 'num_basal_to_replace_with_CI': 1},
    'Tufts': {'sim_name_add_suffix': 'REPTufts', 'num_tuft_to_replace_with_CI': 2},
    '1Tuft': {'sim_name_add_suffix': 'REP1Tuft', 'num_tuft_to_replace_with_CI': 1},
    'Basals&Tufts': {'sim_name_add_suffix': 'REPBasals&Tufts', 'num_basal_to_replace_with_CI': 8, 'num_tuft_to_replace_with_CI': 2}
}

varying_syn_attributes = {
    'None': {'sim_name_add_suffix': ''},
    'NoMapping': {'sim_name_add_suffix': 'NoMapping', 'synapse_mapping': False},
    'Merging': {'sim_name_add_suffix': 'Merging', 'merge_synapses': True},
    'MappingMerging': {'sim_name_add_suffix': 'MappingMerging', 'synapse_mapping': True, 'merge_synapses': True}
}

mean = (np.log(0.45) - 0.5 * np.log((0.35 / 0.45) ** 2 + 1))

# Synaptic gmax parameters
synaptic_gmax_params = {
    'inh_gmax_dist': 1,
    'soma_gmax_dist': 1,
    'exc_gmax_mean_0': mean,
    'exc_gmax_std_0': np.sqrt(np.log((0.35 / 0.45) ** 2 + 1)),
    'exc_gmax_clip': (0, 5),
    'exc_scalar': 1
}

# Combine common attributes with synaptic gmax parameters
common_attributes = {**common_attributes[common_attributes_to_use], **synaptic_gmax_params}

# Generate varying attributes by combining morphology, replace_w_CI, and synapse attributes
varying_attributes = []
for morph_key in morphology_keys:
    if morph_key in ['Complex', 'Trees']:  # Complex cell will not have any dendrites replaced with CI
        replace_keys = ['None']
    else:
        replace_keys = replace_w_CI_keys
    for replace_key in replace_keys:
        for syn_key in synapse_keys:
            combined_attrs = {**morphology_attributes[morph_key], **replace_w_CI_attributes[replace_key], **varying_syn_attributes[syn_key]}
            combined_attrs['base_sim_name'] = f"{morphology_attributes[morph_key]['base_sim_name']}{replace_w_CI_attributes[replace_key].get('sim_name_add_suffix', '')}{varying_syn_attributes[syn_key].get('sim_name_add_suffix', '')}"
            combined_attrs.pop('sim_name_add_prefix', None)
            combined_attrs.pop('sim_name_add_suffix', None)
            varying_attributes.append(combined_attrs)

# Function to generate simulations
def generate_simulations(neuron_random_states, numpy_random_states, varying_attributes, common_attributes):
    all_parameters = []

    def create_parameters(numpy_seed, neuron_seed, attributes, amp=None, excFR_increase=None):
        sim_name_parts = [attributes['base_sim_name'], f"Np{numpy_seed}"]
        if neuron_seed is not None:
            sim_name_parts.append(f"Neu{neuron_seed}")
        if amp is not None:
            sim_name_parts.append(f"amp{round(amp, 1)}")
        if excFR_increase is not None:
            sim_name_parts.append(f"EXCinc{round(excFR_increase, 1)}")
        sim_name = '_'.join(sim_name_parts)

        params = {
            **common_attributes,
            'numpy_random_state': numpy_seed,
            'sim_name': sim_name,
            **{k: v for k, v in attributes.items() if k != 'base_sim_name'}
        }

        if neuron_seed is not None:
            params['neuron_random_state'] = neuron_seed
        if amp is not None:
            params['h_i_amplitude'] = round(amp, 1)
        if excFR_increase is not None:
            params['excFR_increase'] = round(excFR_increase, 1)

        return HayParameters(**params)

    if not numpy_random_states:
        numpy_random_states = [None]
    if not neuron_random_states:
        neuron_random_states = [None]

    for numpy_seed in numpy_random_states:
        for neuron_seed in neuron_random_states:
            for attributes in varying_attributes:
                if 'CI_on' in common_attributes:
                    for amp in np.arange(0, 2.1, 0.5):
                        all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, amp=amp))
                elif 'exc_constant_fr' in common_attributes:
                    for excFR_increase in np.arange(0, 8.1, 2):
                        all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, excFR_increase=excFR_increase))
                else:
                    all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes))

    return all_parameters

if __name__ == "__main__":
    # Main execution code
    all_parameters = generate_simulations(neuron_random_states, numpy_random_states, varying_attributes, common_attributes)

    # Define your batch size
    batch_size = 64

    # Check how many batches you will need
    if len(all_parameters) > (batch_size - 1):
        number_of_batches = math.ceil(len(all_parameters) / batch_size)
        print(number_of_batches)
        
        # Create batches of indices
        batches = [all_parameters[i * batch_size:(i + 1) * batch_size] for i in range(number_of_batches)]
        
        # Run each batch
        for i, batch in enumerate(batches):
            sim = Simulation(SkeletonCell.Hay, title=sim_title)
            if i == 0:
                path_to_use = sim.path
            else:
                sim.path = path_to_use
            for parameters in batch:
                sim.submit_job(parameters)
            sim.run()
            
    else:
        # Initialize simulation
        sim = Simulation(SkeletonCell.Hay, title=sim_title)

        # Submit jobs to simulation
        for parameters in all_parameters:
            sim.submit_job(parameters)

        # Remove directory if it exists
        try:
            os.system("rm -r x86_64/")
        except:
            pass

        # Run the simulation
        sim.run()
        
    print(sim.path)
