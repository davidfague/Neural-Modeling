"""
generate_param_sets.py
----------------------
Reusable parameter-combination & construction utilities.

Key exports:
- get_parameter_combinations()
- get_index_matched_parameter_combinations()
- create_parameters()
- generate_simulations()

These functions are agnostic to your specific experiment. They take:
- seeds,
- templates (morphologies, syn reductions, CI replacements),
- a sim_type string,
- a select_params dict for sweeps (can be {}),
- and a common_params dict (already composed by your configuration script).

They return a list of HayParameters objects ready for folder creation and simulation.
"""

from typing import Dict, Any, List, Tuple, Iterable
from Modules.constants import HayParameters


# ---------- helpers for formatting suffixes ----------
def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, tuple):
        return "-".join(str(x) for x in value)
    if isinstance(value, dict):
        # Order keys for consistency
        return "_".join(f"{k}-{value[k]}" for k in sorted(value))
    return str(value)


def get_parameter_combinations(param_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cartesian-product combinations for select_params.
    Each key maps to a dict with:
        - "values": list of values
        - "sim_name_suffix": str for naming
        - optional "nested_keys": list[str] for dict-valued params
        - optional "always_include_suffix": bool
    """
    import itertools

    keys = list(param_dict.keys())
    value_lists = [param_dict[k]["values"] for k in keys]
    combos: List[Dict[str, Any]] = []

    for values in itertools.product(*value_lists):
        combo: Dict[str, Any] = {}
        suffix_parts: List[str] = []

        for key, value in zip(keys, values):
            combo[key] = value

            show_suffix = (
                len(param_dict[key]["values"]) > 1
                or param_dict[key].get("always_include_suffix", False)
            )
            if show_suffix:
                if isinstance(value, dict) and "nested_keys" in param_dict[key]:
                    suffix_values = []
                    for nk in param_dict[key]["nested_keys"]:
                        path = nk.split(".")
                        v = value
                        for step in path:
                            v = v[step]
                        suffix_values.append(_format_value(v))
                    formatted = "_".join(suffix_values)
                else:
                    formatted = _format_value(value)
                suffix_parts.append(f"{param_dict[key]['sim_name_suffix']}{formatted}")

        combo["sim_name_suffix"] = "_".join(suffix_parts) if suffix_parts else ""
        combos.append(combo)

    return combos


def get_index_matched_parameter_combinations(param_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Index-matched combinations (all varying lists must have equal length).
    This matches your previous behavior for tightly coupled sweeps.
    """
    if not param_dict:
        return [{}]

    keys = list(param_dict.keys())
    lengths = [len(param_dict[k]["values"]) for k in keys]
    max_len = max(lengths)

    # Validate lengths
    for k, L in zip(keys, lengths):
        if L not in (1, max_len):
            raise ValueError(
                f"Parameter '{k}' has {L} values, but others vary with {max_len}. "
                "All lists with len>1 must match."
            )

    combos: List[Dict[str, Any]] = []
    for i in range(max_len):
        combo: Dict[str, Any] = {}
        suffix_parts: List[str] = []

        for k in keys:
            values = param_dict[k]["values"]
            value = values[i] if len(values) > 1 else values[0]
            combo[k] = value

            show_suffix = len(values) > 1 or param_dict[k].get("always_include_suffix", False)
            if show_suffix:
                if isinstance(value, dict) and "nested_keys" in param_dict[k]:
                    suffix_values = []
                    for nk in param_dict[k]["nested_keys"]:
                        path = nk.split(".")
                        v = value
                        for step in path:
                            v = v[step]
                        suffix_values.append(_format_value(v))
                    formatted = "_".join(suffix_values)
                else:
                    formatted = _format_value(value)
                suffix_parts.append(f"{param_dict[k]['sim_name_suffix']}{formatted}")

        combo["sim_name_suffix"] = "_".join(suffix_parts) if suffix_parts else ""
        combos.append(combo)

    return combos


# ---------- construction ----------
def create_parameters(
    *,
    numpy_seed: int,
    neuron_seed: int | None,
    sim_type: str,
    common_params: Dict[str, Any],
    morphology_params: Dict[str, Any],
    syn_reduction_params: Dict[str, Any],
    ci_replacement_params: Dict[str, Any],
    varied_params: Dict[str, Any],
    amp: float | None = None,
    excFR_increase: float | None = None,
) -> HayParameters:
    """
    Merge parameter dicts into a HayParameters object and build an informative sim_name.
    """
    name_parts: List[str] = []

    # Sim type first, then morphology base
    name_parts.append(sim_type)
    name_parts.append(morphology_params.get("base_sim_name", ""))

    # Reduction / CI suffixes
    if syn_reduction_params.get("sim_name_add_suffix"):
        name_parts.append(syn_reduction_params["sim_name_add_suffix"])
    if ci_replacement_params.get("sim_name_add_suffix"):
        name_parts.append(ci_replacement_params["sim_name_add_suffix"])

    # Varied param suffix (if present)
    if varied_params.get("sim_name_suffix"):
        name_parts.append(varied_params["sim_name_suffix"])

    # Seed suffixes
    name_parts.append(f"Np{numpy_seed}")
    if neuron_seed is not None:
        name_parts.append(f"Neu{neuron_seed}")

    # Amplitude / FR increases if used
    if amp is not None:
        name_parts.append(f"amp{round(amp, 1)}")
    if excFR_increase is not None:
        name_parts.append(f"EXCinc{round(excFR_increase, 1)}")

    sim_name = "_".join([s for s in name_parts if s])

    # Merge params (later entries override earlier)
    params: Dict[str, Any] = {}
    params.update(common_params)
    params.update(morphology_params)
    params.update(syn_reduction_params)
    params.update(ci_replacement_params)
    params.update(varied_params)

    # Extra metadata
    params["sim_name"] = sim_name
    params["numpy_random_state"] = numpy_seed
    params["morphology_name"] = morphology_params.get("base_sim_name", "no_morphology_name_provided")
    params["sim_type"] = sim_type
    if neuron_seed is not None:
        params["neuron_random_state"] = neuron_seed
    if amp is not None:
        params["h_i_amplitude"] = round(amp, 1)
    if excFR_increase is not None:
        params["excFR_increase"] = round(excFR_increase, 1)

    # Only pass keys HayParameters accepts (defensive)
    valid_keys = HayParameters.__init__.__code__.co_varnames
    filtered = {k: v for k, v in params.items() if k in valid_keys}

    print(f"[generate_param_sets] creating HayParameters with keys: {sorted(filtered.keys())}")
    return HayParameters(**filtered)


def generate_simulations(
    *,
    neuron_random_states: Iterable[int | None],
    numpy_random_states: Iterable[int],
    select_params: Dict[str, Dict[str, Any]],
    common_params: Dict[str, Any],
    sim_type: str,
    morphologies: Dict[str, Dict[str, Any]],
    syn_reductions: Dict[str, Dict[str, Any]],
    ci_replacements: Dict[str, Dict[str, Any]],
    morphologies_to_use: List[str],
    syn_reductions_to_use: List[str],
    ci_replacements_to_use: List[str],
    index_matched: bool = True,
) -> List[HayParameters]:
    """
    Core combinator:
      seeds × morphologies × syn_reductions × ci_replacements × varied_params
      (and optional inner loops for CI or exc-constant-FR sims)

    select_params: {}
      Structure like:
        {
          "param_name": {
             "values": [ ... ],
             "sim_name_suffix": "MyParam",
             # optional:
             "always_include_suffix": True,
             "nested_keys": ["tuft.syn_density", ...]  # for dict-valued params
          },
          ...
        }
    """
    # Build varied-parameter combinations
    if index_matched:
        varied_list = get_index_matched_parameter_combinations(select_params)
    else:
        varied_list = get_parameter_combinations(select_params)

    results: List[HayParameters] = []

    for np_seed in numpy_random_states:
        for nrn_seed in neuron_random_states:
            for m_name in morphologies_to_use:
                for sr_name in syn_reductions_to_use:
                    for ci_name in ci_replacements_to_use:
                        morph = morphologies[m_name]
                        sred  = syn_reductions[sr_name]
                        cirep = ci_replacements[ci_name]

                        for varied in varied_list:
                            # If the simulation uses current injection, iterate amplitudes
                            if common_params.get("CI_on", False):
                                for amp in _frange(-0.5, 2.1, 0.5):
                                    hp = create_parameters(
                                        numpy_seed=np_seed,
                                        neuron_seed=nrn_seed,
                                        sim_type=sim_type,
                                        common_params=common_params,
                                        morphology_params=morph,
                                        syn_reduction_params=sred,
                                        ci_replacement_params=cirep,
                                        varied_params=varied,
                                        amp=amp,
                                    )
                                    results.append(hp)

                            # If the simulation uses a constant excitatory FR, iterate increases
                            elif common_params.get("exc_constant_fr", False):
                                for inc in _frange(0.0, 8.1, 2.0):
                                    hp = create_parameters(
                                        numpy_seed=np_seed,
                                        neuron_seed=nrn_seed,
                                        sim_type=sim_type,
                                        common_params=common_params,
                                        morphology_params=morph,
                                        syn_reduction_params=sred,
                                        ci_replacement_params=cirep,
                                        varied_params=varied,
                                        excFR_increase=inc,
                                    )
                                    results.append(hp)

                            else:
                                hp = create_parameters(
                                    numpy_seed=np_seed,
                                    neuron_seed=nrn_seed,
                                    sim_type=sim_type,
                                    common_params=common_params,
                                    morphology_params=morph,
                                    syn_reduction_params=sred,
                                    ci_replacement_params=cirep,
                                    varied_params=varied,
                                )
                                results.append(hp)

    return results


# Small float-range utility
def _frange(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    x = start
    # tolerate floating rounding at the end
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals
