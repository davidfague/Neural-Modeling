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
use_SA_probs = False
syn_numbers_to_use = 'Full'  # Options: '1000', 'Full', etc.
common_attributes_to_use = 'tuning_synapses' # Options: 'sta', 'FI', 'FI_ExcFR', 'tuning_synapses', 'checking_synapse_distributions'
morphology_keys = ['Complex']  # Options: 'Complex', 'Branches', 'Trees' (can do multiple)
replace_w_CI_keys = ['None']  # Options: 'None', 'Tufts', 'Basals&Tufts', etc. (can do multiple)
numpy_random_states = [1000, 10000000]  # Add more seeds if needed (can do multiple)
neuron_random_states = None
sim_title = 'BenSynapses_testing_depth_of_mod'#'BenSynapses_final_detailed_syn_dist_analysis'

syn_numbers = {
    'Density': {'inh': None, 'exc': None},
    'Full': {'inh': 2650, 'exc': 26100},
    '1000': {'inh': int(1000 * (2650 / (26100 + 2650))), 'exc': int(1000 * (26100 / (26100 + 2650)))},
    '10000': {'inh': int(10000 * (2650 / (26100 + 2650))), 'exc': int(10000 * (26100 / (26100 + 2650)))}
}

# Define the template for common attributes
common_attributes_dict = { # simulation options
    'sta': { # in vivo simulation with recording currents/conductances
        'h_tstop': 30000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': True,
        'record_all_synapses': True,
        'record_spike_trains': False,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs,
        'record_synapse_distributions': False
    },
    'FI': { # ramp current injection
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
    'FI_ExcFR': { # instead of ramp current injection, ramps excitatory firing rates
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
    },
    'checking_synapse_distributions': { # short in vivo simulation recording resulting synapse parameters/distributions (location, weight, etc.)
        'h_tstop': 1000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': False,
        'record_all_synapses': False,
        'record_spike_trains': False,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs,
        'record_synapse_distributions': True
    },
    'tuning_synapses': { # in vivo simulation (not fully implemented yet: needs sim title adjustment)
        'h_tstop': 10000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': False,
        'record_all_synapses': False,
        'record_spike_trains': False,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs,
        'record_synapse_distributions': False
    },
}

morphology_attributes = { #  model morphological reduction options
    'Complex': {'base_sim_name': 'Complex'},
    'Branches': {'base_sim_name': 'Branches', 'reduce_obliques': True, 'reduce_tufts': True, 'reduce_basals': 3},
    'Trees': {'base_sim_name': 'Trees', 'reduce_apic': True, 'reduce_basals': 1}
}

replace_w_CI_attributes = { # replacing dendrites with current injection options
    'None': {'sim_name_add_suffix': ''},
    'Basals': {'sim_name_add_suffix': 'REPBasals', 'num_basal_to_replace_with_CI': 8},
    '1Basal': {'sim_name_add_suffix': 'REP1Basal', 'num_basal_to_replace_with_CI': 1},
    'Tufts': {'sim_name_add_suffix': 'REPTufts', 'num_tuft_to_replace_with_CI': 2},
    '1Tuft': {'sim_name_add_suffix': 'REP1Tuft', 'num_tuft_to_replace_with_CI': 1},
    'Basals&Tufts': {'sim_name_add_suffix': 'REPBasals&Tufts', 'num_basal_to_replace_with_CI': 8, 'num_tuft_to_replace_with_CI': 2}
}

varying_syn_attributes = { # synapse reduction options
    'None': {'sim_name_add_suffix': ''},
    'NoMapping': {'sim_name_add_suffix': 'NoMapping', 'synapse_mapping': False},
    'Merging': {'sim_name_add_suffix': 'Merging', 'merge_synapses': True},
    'MappingMerging': {'sim_name_add_suffix': 'MappingMerging', 'synapse_mapping': True, 'merge_synapses': True}
}

# settling on  '2024-09-16-09-35-21-TuningBenInhSynapses/Complex_InhGmaxApic7.1_InhGmaxDend0.0016_SomaGmax0.0025_ExcGmax-1.0351_Np1000'
inh_gmax_range_apic = [7.1]
inh_gmax_range_dend = [0.0016]
soma_gmax_range = [0.0025]

# for Ziao Synapses (after changing their reversal potentials to match Ben's.)
# inh_gmax_range_apic = [204]
# inh_gmax_range_dend = [7.0]  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = [6.0]  # from 0.8 to 3.0 by 0.1

mean = (np.log(0.45) - 0.5 * np.log((0.35 / 0.45) ** 2 + 1))
exc_gmax_mean_range = [mean]  # Example excitatory gmax mean range

all_depth_of_mod_range = [0, 0.1, 0.25, 0.5, 0.75, 1]

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
def generate_simulations(neuron_random_states, numpy_random_states, varying_attributes, common_attributes_dict, inh_gmax_range_apic, inh_gmax_range_dend, soma_gmax_range, all_depth_of_mod_range=[0]):
    all_parameters = []

    def create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, amp=None, excFR_increase=None, all_depth_of_mod=0):
        sim_name_parts = [attributes['base_sim_name'], f"InhGmaxApic{round(inh_gmax_apic, 4)}", f"InhGmaxDend{round(inh_gmax_dend, 4)}", f"SomaGmax{round(soma_gmax, 4)}", f"ExcGmax{round(exc_gmax_mean,4)}", f"Np{numpy_seed}", f"RhythDepth{all_depth_of_mod}"]
        if neuron_seed is not None:
            sim_name_parts.append(f"Neu{neuron_seed}")
        if amp is not None:
            sim_name_parts.append(f"amp{round(amp, 1)}")
        if excFR_increase is not None:
            sim_name_parts.append(f"EXCinc{round(excFR_increase, 1)}")
        sim_name = '_'.join(sim_name_parts)

        params = {
            **common_attributes_dict,
            'apic_inh_gmax_dist': inh_gmax_apic,
            'basal_inh_gmax_dist': inh_gmax_dend,
            'soma_gmax_dist': soma_gmax,
            'exc_gmax_mean_0': exc_gmax_mean,
            'numpy_random_state': numpy_seed,
            'sim_name': sim_name,
            'rhyth_depth_inh_perisomatic': all_depth_of_mod,
            'rhyth_depth_inh_distal': all_depth_of_mod,
            **{k: v for k, v in attributes.items() if k != 'base_sim_name'}
        }

        # Filter out keys that should not be passed to HayParameters
        valid_params = {k: v for k, v in params.items() if k in HayParameters.__init__.__code__.co_varnames}

        if neuron_seed is not None:
            valid_params['neuron_random_state'] = neuron_seed
        if amp is not None:
            valid_params['h_i_amplitude'] = round(amp, 1)
        if excFR_increase is not None:
            valid_params['excFR_increase'] = round(excFR_increase, 1)

        return HayParameters(**valid_params)

    if not numpy_random_states:
        numpy_random_states = [None]
    if not neuron_random_states:
        neuron_random_states = [None]

    for all_depth_of_mod in all_depth_of_mod_range:
        for inh_gmax_apic in inh_gmax_range_apic:
            for inh_gmax_dend in inh_gmax_range_dend:
                for soma_gmax in soma_gmax_range:
                    for exc_gmax_mean in exc_gmax_mean_range:
                        for numpy_seed in numpy_random_states:
                            for neuron_seed in neuron_random_states:
                                for attributes in varying_attributes:
                                    current_common_attributes = common_attributes_dict
                                    if 'CI_on' in current_common_attributes:
                                        for amp in np.arange(0, 2.1, 0.5):
                                            all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, amp=amp, all_depth_of_mod=all_depth_of_mod))
                                    elif 'exc_constant_fr' in current_common_attributes:
                                        for excFR_increase in np.arange(0, 8.1, 2):
                                            all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, excFR_increase=excFR_increase, all_depth_of_mod=all_depth_of_mod))
                                    else:
                                        all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, all_depth_of_mod=all_depth_of_mod))

    return all_parameters

if __name__ == "__main__":
    # Combine common attributes with synaptic gmax parameters
    common_attributes_dict = {**common_attributes_dict[common_attributes_to_use]}
    # Main execution code to generate simulations
    all_parameters = generate_simulations(neuron_random_states, numpy_random_states, varying_attributes, common_attributes_dict, inh_gmax_range_apic, inh_gmax_range_dend, soma_gmax_range)

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
