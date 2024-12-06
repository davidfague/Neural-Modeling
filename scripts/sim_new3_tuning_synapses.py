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
common_attributes_to_use = 'sta' # Options: 'sta', 'FI', 'FI_ExcFR'
morphology_keys = ['Complex']  # Options: 'Complex', 'Branches', 'Trees'
replace_w_CI_keys = ['None']  # Options: 'None', 'Tufts', 'Basals&Tufts', etc.
numpy_random_states = [1000, 2000, 10000, 20000, 100000, 200000]  # Add more seeds if needed
neuron_random_states = None
sim_title = 'ZiaoSynapses_final_detailed_random_seeding_sta_testing'

syn_numbers = {
    'Density': {'inh': None, 'exc': None},
    'Full': {'inh': 2650, 'exc': 26100},
    '1000': {'inh': int(1000 * (2650 / (26100 + 2650))), 'exc': int(1000 * (26100 / (26100 + 2650)))},
    '10000': {'inh': int(10000 * (2650 / (26100 + 2650))), 'exc': int(10000 * (26100 / (26100 + 2650)))}
}

# Define the template for common attributes
common_attributes_dict = { # simulation options
    'sta': { # in vivo simulation
        'h_tstop': 150000,
        'merge_synapses': False,
        'record_ecp': False,
        'record_all_channels': True,
        'record_all_synapses': True,
        'record_spike_trains': False,
        'exc_use_density': syn_numbers_to_use == 'Density',
        'inh_use_density': syn_numbers_to_use == 'Density',
        'inh_syn_number': syn_numbers[syn_numbers_to_use]['inh'],
        'exc_syn_number': syn_numbers[syn_numbers_to_use]['exc'],
        'use_SA_probs': use_SA_probs
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
    }
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
# New inhibitory gmax ranges
# inh_gmax_range_apic = np.arange(2.4, 3.2, 0.1)  # Apical range: 2.8 ± 0.4 with step 0.1
# inh_gmax_range_dend = np.arange(2.0, 2.8, 0.4)  # Basal/Dendritic range: 2.4 ± 0.4 with step 0.4
# soma_gmax_range = np.arange(0.25, 0.70, 0.15)  # from 0.25 to 0.55 by 0.15
# closest: InhGmaxApic3.0_InhGmaxDend2.0_SomaGmax0.55
# inh_gmax_range_apic = np.arange(2.6, 3.45, 0.05)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(1.8, 2.4, 0.2)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.4, 1, 0.1)  # from 0.40 to 0.9 by 0.1
# cloesest: Complex_InhGmaxApic3.4_InhGmaxDend2.0_SomaGmax0.9_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(3.3, 4.6, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(1.8, 2.3, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.8, 3.1, 0.1)  # from 0.8 to 3.0 by 0.1
# Closest: Complex_InhGmaxApic4.4_InhGmaxDend2.2_SomaGmax2.0_ExcGmax-1.0351_Np1000

# inh_gmax_range_apic = np.arange(4.5, 5.6, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(1.5, 2.4, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(2.0, 3.0, 0.1)  # from 0.8 to 3.0 by 0.1

# Closest: Complex_InhGmaxApic4.6_InhGmaxDend2.4_SomaGmax2.1_ExcGmax-1.0351_Np1000
# Complex_InhGmaxApic4.3_InhGmaxDend2.0_SomaGmax1.8_ExcGmax-1.0351_Np1000

# inh_gmax_range_apic = np.arange(4, 6, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(2.0, 2.8, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(1.8, 2.2, 0.1)  # from 0.8 to 3.0 by 0.1

#Complex_InhGmaxApic5.9_InhGmaxDend2.0_SomaGmax2.1_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(5.9, 8.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(1.0, 2.1, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(2.1, 2.5, 0.1)  # from 0.8 to 3.0 by 0.1
#Complex_InhGmaxApic5.9_InhGmaxDend2.0_SomaGmax2.1_ExcGmax-1.0351_Np1000
#Complex_InhGmaxApic6.2_InhGmaxDend1.0_SomaGmax2.3_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(5.9, 8.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.1, 1.3, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(2.1, 2.5, 0.1)  # from 0.8 to 3.0 by 0.1
#InhGmaxApic7.0_InhGmaxDend0.5_SomaGmax2.3_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(6.9, 7.6, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.1, 1.0, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(2.1, 4.1, 0.1)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic6.9_InhGmaxDend0.1_SomaGmax2.3_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = [6.9]
# inh_gmax_range_dend = [0.1]
# soma_gmax_range = [2.3]

# now retuning inhibitory weights with updated excitatory firing rates.
# inh_gmax_range_apic = np.arange(10.0, 10.5, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(3.0, 3.5, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(4.0, 4.5, 0.1)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic10.1_InhGmaxDend3.2_SomaGmax4.1_ExcGmax-1.0351_Np1000
inh_gmax_range_apic = np.arange(15.0, 24.0, 1.0)  # Apical range: 3.0 ± 0.4 with step 0.05
inh_gmax_range_dend = [3.2]#np.arange(3.2, 3.5, 1.0)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = np.arange(5.0, 14.0, 1.0)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic21.0_InhGmaxDend3.2_SomaGmax13.0_ExcGmax-1.0351_Np1000
# need to raise apic inhibition. dend looks pretty good, soma firing rate pretty good
inh_gmax_range_apic = np.arange(21.0, 35.0, 1.0)  # Apical range: 3.0 ± 0.4 with step 0.05
inh_gmax_range_dend = np.arange(3.1, 3.5, 1.0)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = np.arange(10.0, 16.0, 1.0)  # from 0.8 to 3.0 by 0.1

# Complex_InhGmaxApic27.0_InhGmaxDend3.1_SomaGmax10.0_ExcGmax-1.0351_Np1000
# Complex_InhGmaxApic30.0_InhGmaxDend3.1_SomaGmax12.0_ExcGmax-1.0351_Np1000
# Complex_InhGmaxApic31.0_InhGmaxDend3.1_SomaGmax15.0_ExcGmax-1.0351_Np1000
# Complex_InhGmaxApic39.0_InhGmaxDend3.1_SomaGmax10.0_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(115, 180, 1.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = [3.1]#np.arange(3.1, 3.5, 1.0)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = [10.0]#np.arange(10.0, 15.0, 1.0)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic145.0_InhGmaxDend3.1_SomaGmax10.0_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(140.0, 151.0, 1.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(1.0, 10.0, 1.0)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(1.0, 10.0, 1.0)  # from 0.8 to 3.0 by 0.1

# # Complex_InhGmaxApic150.0_InhGmaxDend3.0_SomaGmax2.0_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(153.0, 177.0, 2.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(2.0, 4.8, 0.2)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(1.0, 3.2, 0.2)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic152.0_InhGmaxDend3.8_SomaGmax1.4_ExcGmax-1.0351_Np1000

# Complex_InhGmaxApic175.0_InhGmaxDend4.6_SomaGmax3.0_ExcGmax-1.0351_Np1000
# about to use the following, but first; testing the old synpase modfiles?
# inh_gmax_range_apic = np.arange(175.0, 198.0, 4.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(4.6, 5.8, 0.2)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(2.8, 4.0, 0.2)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic195.0_InhGmaxDend5.8_SomaGmax4.0_ExcGmax-1.0351_Np1000


# inh_gmax_range_apic = np.arange(195.0, 220.0, 4.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(5.8, 7.0, 0.2)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(4.0, 6.0, 0.2)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic207.0_InhGmaxDend7.0_SomaGmax5.8_ExcGmax-1.0351_Np1000

# inh_gmax_range_apic = np.arange(205.0, 210.0, 1.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(6.8, 8.0, 0.2)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(5.4, 7.0, 0.2)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic205.0_InhGmaxDend7.2_SomaGmax5.4_ExcGmax-1.0351_Np1000

# inh_gmax_range_apic = np.arange(203.0, 207.0, 1.0)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(6.8, 7.6, 0.2)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(5.0, 6.2, 0.2)  # from 0.8 to 3.0 by 0.1
# Complex_InhGmaxApic204.0_InhGmaxDend7.0_SomaGmax6.0_ExcGmax-1.0351_Np1000
# Complex_InhGmaxApic204.0_InhGmaxDend7.0_SomaGmax6.0_ExcGmax-1.0351_Np1000

# for Ziao Synapses (after changing their reversal potentials to match Ben's.)
inh_gmax_range_apic = [204]
inh_gmax_range_dend = [7.0]  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = [6.0]  # from 0.8 to 3.0 by 0.1

# # for Ben Synapses
# inh_gmax_range_apic = np.arange(3.3, 4.6, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(1.8, 2.3, 0.1)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.8, 3.1, 0.1)  # from 0.8 to 3.0 by 0.1

# # for Ben Synapses
# inh_gmax_range_apic = np.arange(1.9, 2.2, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.001, .01, 0.001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.001, .01, 0.001)  # from 0.8 to 3.0 by 0.1Complex_InhGmaxApic0.9_InhGmaxDend0.1_SomaGmax0.1_ExcGmax-1.0351_Np1000
# # Complex_InhGmaxApic2.2_InhGmaxDend0.009_SomaGmax0.003_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(2.2, 3.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.009, .021, 0.001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.001, .006, 0.001)
# # '2024-09-05-09-06-52-TuningBenInhSynapses/Complex_InhGmaxApic2.9_InhGmaxDend0.011_SomaGmax0.001_ExcGmax-1.0351_Np1000'
# inh_gmax_range_apic = np.arange(3.0, 4.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.009, .012, 0.001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.0005, .0020, 0.0005)
# # Complex_InhGmaxApic3.6_InhGmaxDend0.009_SomaGmax0.001_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(3.6, 5.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.007, .012, 0.001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.0008, .0014, 0.0001)
# Complex_InhGmaxApic4.5_InhGmaxDend0.007_SomaGmax0.001_ExcGmax-1.0351_Np1000
# inh_gmax_range_apic = np.arange(3.6, 5.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.001, .010, 0.001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.0008, .0014, 0.0001)
#2024-09-06-10-33-43-TuningBenInhSynapses/Complex_InhGmaxApic4.5_InhGmaxDend0.007_SomaGmax0.001_ExcGmax-1.0351_Np1000
# '2024-09-06-10-33-43-TuningBenInhSynapses/Complex_InhGmaxApic5.0_InhGmaxDend0.005_SomaGmax0.001_ExcGmax-1.0351_Np1000'
# inh_gmax_range_apic = np.arange(5.0, 7.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
# inh_gmax_range_dend = np.arange(0.003, .0055, 0.0005)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
# soma_gmax_range = np.arange(0.0008, .0014, 0.0001)
# 2024-09-09-11-26-53-TuningBenInhSynapses/Complex_InhGmaxApic5.9_InhGmaxDend0.003_SomaGmax0.0011_ExcGmax-1.0351_Np1000
# KEEP: 2024-09-09-11-26-53-TuningBenInhSynapses/Complex_InhGmaxApic6.5_InhGmaxDend0.003_SomaGmax0.0012_ExcGmax-1.0351_Np1000
inh_gmax_range_apic = np.arange(5.9, 7.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
inh_gmax_range_dend = np.arange(0.0005, .0035, 0.0005)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = np.arange(0.00010, .0017, 0.0001)
# 2024-09-10-14-14-59-TuningBenInhSynapses/Complex_InhGmaxApic6.4_InhGmaxDend0.0015_SomaGmax0.0005_ExcGmax-1.0351_Np1000
# 2024-09-10-14-14-59-TuningBenInhSynapses/Complex_InhGmaxApic6.4_InhGmaxDend0.0015_SomaGmax0.0016_ExcGmax-1.0351_Np1000
inh_gmax_range_apic = np.arange(6.0, 7.1, 0.1)  # Apical range: 3.0 ± 0.4 with step 0.05
inh_gmax_range_dend = np.arange(0.001, .0025, 0.0005)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = np.arange(0.0016, .0032, 0.0001)
# 2024-09-11-12-05-06-TuningBenInhSynapses/Complex_InhGmaxApic6.7_InhGmaxDend0.0015_SomaGmax0.0017_ExcGmax-1.0351_Np1000
# '2024-09-11-12-05-06-TuningBenInhSynapses/Complex_InhGmaxApic6.7_InhGmaxDend0.0015_SomaGmax0.0029_ExcGmax-1.0351_Np1000'
inh_gmax_range_apic = np.arange(6.5, 6.95, 0.05)  # Apical range: 3.0 ± 0.4 with step 0.05
inh_gmax_range_dend = np.arange(0.001, .002, 0.0001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = np.arange(0.0020, .0035, 0.0001)
# '2024-09-13-15-53-59-TuningBenInhSynapses/Complex_InhGmaxApic6.95_InhGmaxDend0.0015_SomaGmax0.0028_ExcGmax-1.0351_Np1000'
inh_gmax_range_apic = np.arange(6.95, 7.35, 0.05)  # Apical range: 3.0 ± 0.4 with step 0.05
inh_gmax_range_dend = np.arange(0.001, .002, 0.0001)  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = np.arange(0.0020, .0035, 0.0001)

# settling on  '2024-09-16-09-35-21-TuningBenInhSynapses/Complex_InhGmaxApic7.1_InhGmaxDend0.0016_SomaGmax0.0025_ExcGmax-1.0351_Np1000'
inh_gmax_range_apic = [7.1]
inh_gmax_range_dend = [0.0016]
soma_gmax_range = [0.0025]

# for Ziao Synapses (after changing their reversal potentials to match Ben's.)
inh_gmax_range_apic = [204]
inh_gmax_range_dend = [7.0]  # Basal/Dendritic range: 2.0 ± 0.2 with step 0.2
soma_gmax_range = [6.0]  # from 0.8 to 3.0 by 0.1

mean = (np.log(0.45) - 0.5 * np.log((0.35 / 0.45) ** 2 + 1))
exc_gmax_mean_range = [mean]  # Example excitatory gmax mean range

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
def generate_simulations(neuron_random_states, numpy_random_states, varying_attributes, common_attributes_dict, inh_gmax_range_apic, inh_gmax_range_dend, soma_gmax_range):
    all_parameters = []

    def create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, amp=None, excFR_increase=None):
        sim_name_parts = [attributes['base_sim_name'], f"InhGmaxApic{round(inh_gmax_apic, 4)}", f"InhGmaxDend{round(inh_gmax_dend, 4)}", f"SomaGmax{round(soma_gmax, 4)}", f"ExcGmax{round(exc_gmax_mean,4)}", f"Np{numpy_seed}"]
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
                                        all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, amp=amp))
                                elif 'exc_constant_fr' in current_common_attributes:
                                    for excFR_increase in np.arange(0, 8.1, 2):
                                        all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean, excFR_increase=excFR_increase))
                                else:
                                    all_parameters.append(create_parameters(numpy_seed, neuron_seed, attributes, inh_gmax_apic, inh_gmax_dend, soma_gmax, exc_gmax_mean))

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
