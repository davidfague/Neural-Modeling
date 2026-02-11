#!/usr/bin/env python3
"""
configure_sim_params.py
-----------------------
User-facing configuration for a specific simulation set.

- Lets you initialize defaults from an optional parameters.pkl
- Builds HayParameters objects using reusable generation utilities
- Returns (parameter_sets, titles, bg rates, etc.) so your pre-sim driver
  can create folders, write spike-train mode notes, and run generators.

Usage (from another script):
    from configure_sim_params import configure_sim_params
    (all_parameter_sets, all_sim_titles, inh_bg_rate, exc_bg_rate,
     N_bg_synapses, SIM_SET_TITLE, inh_syn_props, exc_syn_props) = \
        configure_sim_params(parameters_pkl_path=None)
"""

import os
import copy
import pickle
from typing import Optional, Tuple, List

from Modules.parameters.constants import HayParameters
from Modules.clustering.clustering import (
    get_default_exc_clustering,
    get_default_inh_clustering,
)
from Modules.post_sim import analysis

# Reusable presets / templates
from Modules.pre_sim.simulation_templates import (
    sim_type_params_all,
    morphologies,
    syn_reductions,
    ci_replacements,
)

# Param generation logic
from Modules.parameters.generate_param_sets import generate_simulations


def configure_sim_params(parameters_pkl_path: Optional[str] = None) -> Tuple[
    List[HayParameters], List[str], float, float, int, str, dict, dict
]:
    """
    Prepare simulation parameters for this experiment.

    Args
    ----
    parameters_pkl_path : str or None
        If provided and exists, defaults (inh/exc syn props) are initialized from this pickle.
        Otherwise we fall back to HayParameters("dummy") for defaults.

    Returns
    -------
    all_parameter_sets : list[HayParameters]
    all_sim_titles     : list[str]
    inh_bg_rate        : float
    exc_bg_rate        : float
    N_bg_synapses      : int
    SIM_SET_TITLE      : str
    inh_syn_properties : dict
    exc_syn_properties : dict
    """

    # === USER CONFIGURABLE (experiment-specific) ===
    skeleton_cell_type = 'Hay'  # "Hay" or "Allen"
    SIM_SET_TITLE   = "lower_syn_conductances_increase_e_pas_to_-75_and_set_exc_fr_to_50mHz"  # descriptive name for this set of sims
    do_reduce_cell = False
    sim_type        = "sta"  # one of: 'sta', 'fi_ci', 'fi_exc', 'check_synapses', 'tuning'
    load_previous_sim_params = False #True
    
    # Clustering mode configuration
    # Options: 'terminal_branch_simple' (one FG per terminal branch, static PCs)
    #          'terminal_branch_fps' (FPS-based clustering, dynamic PCs - DEFAULT)
    exc_clustering_mode = 'terminal_branch_fps'
    inh_clustering_mode = 'global_inh'  # 'global_inh' or 'terminal_branch_fps'
    
    parameters_pkl_path = (
        "/home/drfrbc/Neural-Modeling/simulations/"
        "2025-10-16-08-28-IncreaseNexusMaxYTo950/"
        "allinh_rhythmic_depth_0.00_Np5000/"
        "parameters.pickle"
    )
    index_matched = False  # set False to do every params_to_vary combination, True to make each matching index a combination
    # analogous example (True, A:[1,2,3], B:[4,5,6]) = [1;4], [2;5], [3;6]
    
    # Clustering configuration
    cluster_exc = True
    assign_all_to_nearest_fg = True  # Assign all synapses to nearest FG (within input_source) even if outside radius
    
    params_to_vary = {  # set to {} for no parameter sweep
        # # Vary inhibitory spike train mode - auto-configures required fields for each mode
        # # Values can be: "delayed", "rhythmic", "poisson", or ["delayed", "rhythmic"] for combined modes
        # "all.spike_train_mode": {
        #     "apply_to": "inh_syn_properties",
        #     "values": ["delayed", "rhythmic", ["delayed", "rhythmic"]],  # Try all three variations 
        #     "sim_name_suffix": "Mode",
        # },
        
        # # Vary delay shift - ONLY for simulations with "delayed" in spike_train_mode
        # "all.delay_config.delay_shift": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [2, 4, 6, 8],
        #     "sim_name_suffix": "DelayShift",
        #     "requires_mode": {"all.spike_train_mode": "delayed"},  # Only vary when delayed mode is active
        # },
        
        # # Vary rhythmic depth - ONLY for simulations with "rhythmic" in spike_train_mode
        # "all.rhythmic_depth": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [0.1, 0.2, 0.3, 0.5],
        #     "sim_name_suffix": "RhyDepth",
        #     "requires_mode": {"all.spike_train_mode": "rhythmic"},  # Only vary when rhythmic mode is active
        # },
        
        # "inh_proximal_mean_fr": {
        #     "apply_to": "common_params",
        #     "values": [0.1, 0.5],
        #     "sim_name_suffix": "InhProxFR",
        # },
	    # "inh_proximal_std_fr": {
        #     "apply_to": "common_params",
        #     "values": [0.05, 0.25],
        #     "sim_name_suffix": "InhProxFR",
        # },
	    # "inh_distal_mean_fr": {
        #     "apply_to": "common_params",
        #     "values": [0.1, 0.5],
        #     "sim_name_suffix": "InhDistFR",
        # },
	    # "inh_distal_std_fr": {
        #     "apply_to": "common_params",
        #     "values": [0.05, 0.25],
        #     "sim_name_suffix": "InhDistFR",
        # },
        # "exc_syn_firing_rate_dist.target_mean": {
        #     "apply_to": "common_params",
        #     "values": [0.001, 0.005,0.01, 0.05],
        #     "sim_name_suffix": "ExcFR",
        # },
        # "do_reduce_cell": {
        #     "apply_to": "common_params",
        #     "values": [False, True],
        #     "sim_name_suffix": "Reduced",
        # },
        # "reduction.branch_disparity_elec_tolerance": {
        #     "apply_to": "common_params",
        #     "values": [0.03, 0.07],
        #     "sim_name_suffix": "BrDispTol",
        #     "requires": {"do_reduce_cell": True},  # Only vary when reduction is enabled
        # },
        # "reduction.branch_distance_elec_tolerance": {
        #     "apply_to": "common_params",
        #     "values": [0.03, 0.07],
        #     "sim_name_suffix": "BrDistTol",
        #     "requires": {"do_reduce_cell": True},  # Only vary when reduction is enabled
        # },
        # "reduction.series_constant_elec_tolerance": {
        #     "apply_to": "common_params",
        #     "values": [0.03, 0.07],
        #     "sim_name_suffix": "SerConstTol",
        #     "requires": {"do_reduce_cell": True},  # Only vary when reduction is enabled
        # },
        # "reduction.branch_mult_tolerance": {
        #     "apply_to": "common_params",
        #     "values": [1.5, 1.7],
        #     "sim_name_suffix": "BrMultTol",
        #     "requires": {"do_reduce_cell": True},  # Only vary when reduction is enabled
        # },
        # "reduction.seg_length_um": {
        #     "apply_to": "common_params",
        #     "values": [3, 10],
        #     "sim_name_suffix": "SegLen",
        #     "requires": {"do_reduce_cell": True},  # Only vary when reduction is enabled
        # },
        # "nexus.syn_density": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [0.22],#, 0.2],  # Min and max from original [0.05, 0.11, 0.15, 0.2]
        #     "sim_name_suffix": "NexInhDen",
        # },
        # "perisomatic.syn_density": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [0.6],#[0.15, 0.22],  # Min and max from original [0.15, 0.22, 0.275, 0.3]
        #     "sim_name_suffix": "PeriInhDen",
        # },
        # "distal_basal.syn_density": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [0.22],#[0.11, 0.22],  # Keep both (already only 2 values)
        #     "sim_name_suffix": "DistBasInhDen",
        # },
        # "tuft.syn_density": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [0.3],  # Min and max from original [0.1, 0.15, 0.2, 0.3, 0.4]
        #     "sim_name_suffix": "TuftInhDen",
        # },

        # "tuft_distant.syn_density": {
        #     "apply_to": "exc_syn_properties",
        #     "values": [0.90*s for s in [0.3]],  # Min and max scales from (2:6:1)
        #     "sim_name_suffix": "TuftExcScale",
        #     "group": "tuft_exc",  # All with same group covary
        # },
        # "tuft_local_L23.syn_density": {
        #     "apply_to": "exc_syn_properties",
        #     "values": [0.10*0.25*s for s in [0.3]],  # Matched
        #     "sim_name_suffix": "",
        #     "group": "tuft_exc",  # Same group - will vary together
        # },
        # "tuft_local_L5.syn_density": {
        #     "apply_to": "exc_syn_properties",
        #     "values": [0.10*0.75*s for s in [0.3]],  # Matched
        #     "sim_name_suffix": "",
        #     "group": "tuft_exc",  # Same group - will vary together
        # },

        # "distal_basal_distant.syn_density": {
        #     "apply_to": "exc_syn_properties",
        #     "values": [0.10*s for s in [0.6]],  # Min and max scales from 1.5, 2.2, 2.7, 3.1
        #     "sim_name_suffix": "",
        #     "group": "basal_exc",  # All with same group covary
        # },
        # "distal_basal_local_L23.syn_density": {
        #     "apply_to": "exc_syn_properties",
        #     "values": [0.9*0.1*s for s in [0.6]],  # Matched
        #     "sim_name_suffix": "",
        #     "group": "basal_exc",  # Same group - will vary together
        # },
        # "distal_basal_local_L5.syn_density": {
        #     "apply_to": "exc_syn_properties",
        #     "values": [0.9*0.9*s for s in [0.6]],  # Matched
        #     "sim_name_suffix": "DistBasExcScale",
        #     "group": "basal_exc",  # Same group - will vary together
        # },
        # # # Vary inhibitory weights
        # "tuft.initial_weight_distribution.params.mean": {
        #     "apply_to": "inh_syn_properties",
        #     "values": [0.003],  # MinS and max from [0.006, 0.012, 0.024]
        #     "sim_name_suffix": "TuftInhWtMean",
        # },
        # # "trunk.initial_weight_distribution.params.mean": {
        # #     "apply_to": "inh_syn_properties",
        # #     "values": [0.006, 0.012],  # Min and max from [0.006, 0.012, 0.024]
        # #     "sim_name_suffix": "TrunkInhWtMean",
        # # },
    }

    # Background spike-train knobs for post-generation update
    inh_bg_rate, exc_bg_rate = 0.85, 0.01
    N_bg_synapses = 0  # how many synapses to force to "background" (via replace_N_synapses)
    
    # Build clustering configurations based on selected modes
    exc_clustering_cfg = get_default_exc_clustering(mode=exc_clustering_mode, assign_all_to_nearest=assign_all_to_nearest_fg)
    inh_clustering_cfg = get_default_inh_clustering(mode=inh_clustering_mode, assign_all_to_nearest=assign_all_to_nearest_fg)

    # Seeds
    numpy_random_states  = [5000]
    neuron_random_states = [None]

    # Which “profiles” (templates) to use; keep these near defaults here
    morphologies_to_use      = ["Complex"]  # keys from simulation_templates.morphologies
    syn_reductions_to_use    = ["None"]     # keys from simulation_templates.syn_reductions
    ci_replacements_to_use   = ["None"]     # keys from simulation_templates.ci_replacements

    # Override clustering if cluster_exc is False
    if not cluster_exc:
        exc_clustering_cfg = {}  # disable excitatory clustering

    # Build parameter sets
    all_parameter_sets = []
    all_sim_titles     = []

    # Pull base sim-type params
    if sim_type not in sim_type_params_all:
        raise ValueError(f"Unknown sim_type '{sim_type}'. Valid: {list(sim_type_params_all.keys())} see Modules/cell_model/simulation_templates.py")
    base_params = sim_type_params_all[sim_type].copy()

    # Load defaults from parameters.pkl if provided, else HayParameters("dummy")
    if parameters_pkl_path and os.path.exists(parameters_pkl_path) and load_previous_sim_params:
        with open(parameters_pkl_path, "rb") as f:
            defaults = pickle.load(f)
        print(f"[configure_sim_params] Loaded defaults from {parameters_pkl_path}")
    else:
        defaults = HayParameters("dummy")

    # Deepcopy to avoid mutating shared defaults
    inh_syn_properties = copy.deepcopy(defaults.inh_syn_properties)
    exc_syn_properties = copy.deepcopy(defaults.exc_syn_properties)

    # Note: spike_train_mode configuration is now handled by params_to_vary with "all.spike_train_mode"
    # The mode-specific fields (delay_config, rhythmic_depth, etc.) are auto-configured
    # by the parameter generation system based on the spike_train_mode value

    # Compose common params passed into the generator
    common_params = base_params.copy()
    common_params.update({
        "inh_syn_properties": inh_syn_properties,
        "exc_syn_properties": exc_syn_properties,
        "exc_clustering":     exc_clustering_cfg,
        "inh_clustering":     inh_clustering_cfg,
        "h_i_amplitude":      0.0,
        "CI_on":              False,
        "skeleton_cell_type": skeleton_cell_type,
        "do_reduce_cell":     do_reduce_cell,
    })

    # Generate HayParameters objects (one per seed x profile x varied_param combo)
    all_parameter_sets = generate_simulations(
        neuron_random_states=neuron_random_states,
        numpy_random_states=numpy_random_states,
        params_to_vary=params_to_vary,
        common_params=common_params,
        sim_type=sim_type,
        morphologies=morphologies,
        syn_reductions=syn_reductions,
        ci_replacements=ci_replacements,
        morphologies_to_use=morphologies_to_use,
        syn_reductions_to_use=syn_reductions_to_use,
        ci_replacements_to_use=ci_replacements_to_use,
        index_matched=index_matched,
    )
    
    all_sim_titles = [p.sim_name for p in all_parameter_sets]

    # Return all the knobs your driver needs
    return (
        all_parameter_sets,
        all_sim_titles,
        inh_bg_rate,
        exc_bg_rate,
        N_bg_synapses,
        SIM_SET_TITLE,
        inh_syn_properties,
        exc_syn_properties,
    )
def log_params_updated_from_loaded_pickle(params: HayParameters, pickle_path: str) -> None:
    """Log which parameters were updated from the loaded pickle for transparency."""
    with open(pickle_path, "rb") as f:
        loaded_params = pickle.load(f)
    loaded_inh_props = loaded_params.inh_syn_properties
    loaded_exc_props = loaded_params.exc_syn_properties

    for input_source, props in params.inh_syn_properties.items():
        if input_source in loaded_inh_props:
            for key, value in props.items():
                if loaded_inh_props[input_source].get(key) != value:
                    print(f"[configure_sim_params] Inh param '{input_source}.{key}' updated from loaded pickle: {loaded_inh_props[input_source].get(key)} -> {value}")

    for input_source, props in params.exc_syn_properties.items():
        if input_source in loaded_exc_props:
            for key, value in props.items():
                if loaded_exc_props[input_source].get(key) != value:
                    print(f"[configure_sim_params] Exc param '{input_source}.{key}' updated from loaded pickle: {loaded_exc_props[input_source].get(key)} -> {value}")
    # check for clustering changes
    if params.exc_clustering != loaded_params.exc_clustering:
        print(f"[configure_sim_params] Exc clustering updated from loaded pickle.")
    if params.inh_clustering != loaded_params.inh_clustering:
        print(f"[configure_sim_params] Inh clustering updated from loaded pickle.")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Configure simulation parameter sets for a run.")
    ap.add_argument("--params", type=str, default=None, help="Path to parameters.pkl for defaults.")
    args = ap.parse_args()

    configure_sim_params(parameters_pkl_path=args.params)
