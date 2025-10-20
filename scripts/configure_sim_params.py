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
from Modules.clusters_global_l5_fg import exc_clustering, inh_clustering

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
    SIM_SET_TITLE   = "SetNexusExc0.3from0.25_DecreaseNexusInh_resetTrunkOblique"
    sim_type        = "sta"         # one of: 'sta', 'fi_ci', 'fi_exc', 'check_synapses', 'tuning'

    # Background spike-train knobs for post-generation update
    inh_bg_rate, exc_bg_rate = 0.85, 0.01
    N_bg_synapses = 0  # how many synapses to force to "background" (via replace_N_synapses)

    # Clustering & rhythmicity
    cluster_exc   = True
    rhythmic_inh  = True
    depth_values  = [0.15]          # sweep over inhibitory rhythmic depth(s)

    # Seeds
    numpy_random_states  = [5000]
    neuron_random_states = [None]

    # Which “profiles” (templates) to use; keep these near defaults here
    morphologies_to_use      = ["Complex"]  # keys from simulation_templates.morphologies
    syn_reductions_to_use    = ["None"]     # keys from simulation_templates.syn_reductions
    ci_replacements_to_use   = ["None"]     # keys from simulation_templates.ci_replacements

    if not cluster_exc:
        # empty out clustering if requested
        try:
            exc_clustering.clear()
        except Exception:
            # if it's not a mutable mapping, just shadow it with an empty one
            exc_clustering = {}

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
        if parameters_pkl_path and os.path.exists(parameters_pkl_path):
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
            if rhythmic_inh:
                props["spike_train_mode"] = "rhythmic"
                props["rhythmic_depth"]  = rhythmic_depth
                props.pop("delay_config", None)
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
            "exc_clustering":     exc_clustering,
            "inh_clustering":     inh_clustering,
            "h_i_amplitude":      0.0,
            "CI_on":              False,
        })

        # No extra sweep here; keep it explicit/empty so future you can add easily
        select_parameters_to_vary = {}

        # Generate HayParameters objects (one per seed x profile combo)
        param_objs = generate_simulations(
            neuron_random_states=neuron_random_states,
            numpy_random_states=numpy_random_states,
            select_params=select_parameters_to_vary,
            common_params=common_params,
            sim_type=sim_type,
            morphologies=morphologies,
            syn_reductions=syn_reductions,
            ci_replacements=ci_replacements,
            morphologies_to_use=morphologies_to_use,
            syn_reductions_to_use=syn_reductions_to_use,
            ci_replacements_to_use=ci_replacements_to_use,
        )

        # Name them to reflect your rhythmic depth and seed
        for p in param_objs:
            p.sim_name = f"allinh_rhythmic_depth_{rhythmic_depth:.2f}_Np{p.numpy_random_state}"

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
