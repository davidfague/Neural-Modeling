"""
simulation_templates.py
-----------------------
Reusable “profiles” / presets used across experiments:
- sim_type_params_all (runtime/recording presets by sim_type)
- morphologies        (tree-reduction or complexity profiles)
- syn_reductions      (mapping/merging policies)
- ci_replacements     (current-injection replacement policies)

These are *static templates* — no NEURON code, no combination logic here.
"""

# --- Simulation type presets ---
sim_type_params_all = {
    "sta": {  # in vivo-like simulation with recording
        "h_tstop": 20000,  # Reduced from 30s to speed up CA tuning experiments
        "merge_synapses": False,
        "record_ecp": False,
        "record_all_channels": True,
        "record_all_synapses": True,
        # "record_spike_trains": True,
        # "record_synapse_distributions": True,
    },
    "passive_syn_fi_ci": {  # ramp current injection WITHOUT synapses
        "h_tstop": 5000,
        "save_every_ms": 5000,
        "all_synapses_off": True,
        "CI_on": True,
        "h_i_duration": 4950,
        "h_i_delay": 50,
        "record_all_channels": True,
        "record_all_synapses": True,
    },
    "active_syn_fi_ci": {  # ramp current injection WITH synapses
        "h_tstop": 5000,
        "save_every_ms": 5000,
        "all_synapses_off": False,
        "CI_on": True,
        "h_i_duration": 4950,
        "h_i_delay": 50,
        "record_all_channels": True,
        "record_all_synapses": True,
    },
    "fi_exc": {  # ramps excitatory firing rates
        "h_tstop": 5000,
        "save_every_ms": 5000,
        "all_synapses_off": False,
        "exc_constant_fr": True,
        "h_i_duration": 4950,
        "h_i_delay": 50,
    },
    "check_synapses": {  # short run to record synapse distributions
        "h_tstop": 1000,
        "merge_synapses": False,
        "record_ecp": False,
        "record_all_channels": False,
        "record_all_synapses": False,
        "record_spike_trains": True,
        "record_synapse_distributions": True,
    },
    "tuning": {  # shorter in vivo-like
        "h_tstop": 5000,
        "merge_synapses": False,
        "record_ecp": False,
        "record_all_channels": True,
        "record_all_synapses": True,
        "record_spike_trains": True,
        "record_synapse_distributions": True,
    },
    "testing": {  # in vivo-like simulation with recording
        "h_tstop": 10000,
        "merge_synapses": False,
        "record_ecp": False,
        "record_all_channels": True,
        "record_all_synapses": True,
        # "record_spike_trains": True,
        # "record_synapse_distributions": True,
    },
}

# --- Synapse reduction policies ---
syn_reductions = {
    "None": {"sim_name_add_suffix": ""},
    "NoMapping": {"sim_name_add_suffix": "NoMapping", "synapse_mapping": False},
    "Merging": {"sim_name_add_suffix": "Merging", "merge_synapses": True},
    "MappingMerging": {
        "sim_name_add_suffix": "MappingMerging",
        "synapse_mapping": True,
        "merge_synapses": True,
    },
}

# --- Morphology profiles ---
morphologies = {
    "Complex": {"base_sim_name": "Complex"},
    "ReduceBranches": {
        "base_sim_name": "ReduceBranches",
        "reduce_obliques": 2,
        "reduce_tufts": 2,
        "reduce_basals": 4,
    },
    "ReduceTrees": {
        "base_sim_name": "ReduceTrees",
        "reduce_apic": 1,
        "reduce_basals": 1,
    },
}

# --- CI replacement policies ---
ci_replacements = {
    "None": {"sim_name_add_suffix": ""},
    "Basals": {"sim_name_add_suffix": "REPBasals", "num_basal_to_replace_with_CI": 8},
    "1Basal": {"sim_name_add_suffix": "REP1Basal", "num_basal_to_replace_with_CI": 1},
    "Tufts": {"sim_name_add_suffix": "REPTufts", "num_tuft_to_replace_with_CI": 2},
    "1Tuft": {"sim_name_add_suffix": "REP1Tuft", "num_tuft_to_replace_with_CI": 1},
    "Basals&Tufts": {
        "sim_name_add_suffix": "REPBasals&Tufts",
        "num_basal_to_replace_with_CI": 8,
        "num_tuft_to_replace_with_CI": 2,
    },
}
