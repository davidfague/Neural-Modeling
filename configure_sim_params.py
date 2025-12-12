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
    from scripts.configure_sim_params import configure_sim_params
    (all_parameter_sets, all_sim_titles, inh_bg_rate, exc_bg_rate,
     N_bg_synapses, SIM_SET_TITLE, inh_syn_props, exc_syn_props) = \
        configure_sim_params(parameters_pkl_path=None)
"""

import os
import copy
import pickle
from typing import Optional, Tuple, List

from Modules.constants import HayParameters
from Modules import analysis

# Clustering module (unified)
from Modules.clustering import (
    get_default_exc_clustering,
    get_default_inh_clustering,
)

# Reusable presets / templates
from Modules.simulation_templates import (
    sim_type_params_all,
    morphologies,
    syn_reductions,
    ci_replacements,
)

# Param generation logic
from scripts.generate_param_sets import generate_simulations


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
    skeleton_cell_type = 'Hay'#"Hay" #"Allen"
    SIM_SET_TITLE   = "checking_synapses"
    do_reduce_cell = False
    sim_type        = "sta"         # one of: 'sta', 'fi_ci', 'fi_exc', 'check_synapses', 'tuning'
    load_previous_sim_params = True
    
    # Clustering mode configuration
    # Options: 'terminal_branch_simple' (one FG per terminal branch, static PCs)
    #          'terminal_branch_fps' (FPS-based clustering, dynamic PCs - DEFAULT)
    exc_clustering_mode = 'terminal_branch_fps'
    inh_clustering_mode = 'global_inh'  # 'global_inh' or 'terminal_branch_fps'
    
    # template_sim_dir = (
    #     "/home/drfrbc/Neural-Modeling/simulations/"
    #     "2025-10-16-08-28-IncreaseNexusMaxYTo950/"
    #     "allinh_rhythmic_depth_0.00_Np5000"
    # )
    parameters_pkl_path = (
        "/home/drfrbc/Neural-Modeling/simulations/"
        "2025-10-16-08-28-IncreaseNexusMaxYTo950/"
        "allinh_rhythmic_depth_0.00_Np5000/"
        "parameters.pickle"
    )
    index_matched = False # set False to do every params_to_vary combination, True to make each matching index a combination
    # analogous example (True, A:[1,2,3], B:[4,5,6]) = [1;4], [2;5], [3;6]
    params_to_vary = { # set to {} for no parameter sweep
        # # Inhibitory (nexus) density
        "nexus.syn_density": {
            "apply_to": "inh_syn_properties",
            "values": [0.22*3.3],  # [0.66, 0.715, 0.77]
            "sim_name_suffix": "NexInhDen",
        },
        "perisomatic.syn_density": {
            "apply_to": "inh_syn_properties",
            "values": [0.22*1.25],#,0.22*1.3, 0.22*1.35],#, 0.33, 0.44],
            "sim_name_suffix": "SomInhDen"
        },

        "distal_basal_local_L5.syn_density": {
            "apply_to": "exc_syn_properties",
            "values": [1.6621*1],#, 1.6621*1.1],#, 0.33, 0.44],
            "sim_name_suffix": "DBL5Den"
        },
    }

    # Background spike-train knobs for post-generation update
    inh_bg_rate, exc_bg_rate = 0.85, 0.01
    N_bg_synapses = 0  # how many synapses to force to "background" (via replace_N_synapses)

    # Clustering & rhythmicity
    cluster_exc   = True
    inh_mode = "delayed"
    depth_values  = [0]          # sweep over inhibitory rhythmic depth(s)
    
    # Build clustering configurations based on selected modes
    exc_clustering_cfg = get_default_exc_clustering(mode=exc_clustering_mode)
    inh_clustering_cfg = get_default_inh_clustering(mode=inh_clustering_mode)

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
        raise ValueError(f"Unknown sim_type '{sim_type}'. Valid: {list(sim_type_params_all.keys())}")
    base_params = sim_type_params_all[sim_type].copy()

    # Loop over your inhibitory rhythmic depths (small sweep)
    for rhythmic_depth in depth_values:
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

        # Modify all inhibitory section inputs per experiment design
        for input_source, props in inh_syn_properties.items():
            if inh_mode == "rhythmic":
                props["spike_train_mode"] = "rhythmic"
                props["rhythmic_depth"]  = rhythmic_depth
                props.pop("delay_config", None)
            elif inh_mode == "delayed":
                props["spike_train_mode"] = "delay"
                props["delay_config"] = {
                    "delay_shift": 4,          # tweak if you want a lag/lead (ms)
                    "ref_synapse_type": "exc", # use excitatory trains
                    "ref_sec_type":   "all",   # across ALL exc input_sources
                    "ref_fg_id":      "all",   # across ALL exc FGs
                    "ref_pc_id":      "all",   # across ALL exc PCs
                }
                # Optional: if these were set in defaults, remove rhythmic keys
                props.pop("rhythmic_frequency", None)
                props.pop("rhythmic_depth", None)
            else:
                # ensure rhythmic is removed if present
                mode = props.get("spike_train_mode")
                if isinstance(mode, list):
                    props["spike_train_mode"] = [m for m in mode if m != "rhythmic"]
                elif mode == "rhythmic":
                    props["spike_train_mode"] = "poisson"

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

        # Generate HayParameters objects (one per seed x profile combo)
        param_objs = generate_simulations(
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

        # # Name them to reflect your rhythmic depth and seed
        # for p in param_objs:
        #     if inh_mode == "delayed":
        #         p.sim_name = f"allinh_delay_shift_{int(inh_syn_properties[next(iter(inh_syn_properties))]['delay_config']['delay_shift'])}ms_Np{p.numpy_random_state}"
        #     else:
        #         p.sim_name = f"allinh_rhythmic_depth_{rhythmic_depth:.2f}_Np{p.numpy_random_state}"

        all_parameter_sets.extend(param_objs)
        all_sim_titles.extend([p.sim_name for p in param_objs])

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Configure simulation parameter sets for a run.")
    ap.add_argument("--params", type=str, default=None, help="Path to parameters.pkl for defaults.")
    args = ap.parse_args()

    configure_sim_params(parameters_pkl_path=args.params)
