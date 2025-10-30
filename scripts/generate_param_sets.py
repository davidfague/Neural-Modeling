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
- a params_to_vary dict for sweeps (can be {}),
- and a common_params dict (already composed by your configuration script).

They return a list of HayParameters objects ready for folder creation and simulation.
"""
from Modules.constants import HayParameters
from typing import Dict, Any, List, Tuple, Iterable, Optional
from copy import deepcopy

# ---------- minimal-decimal suffix formatting ----------

def _min_decimals_for_uniqueness(values: List[float], max_decimals: int = 6) -> int:
    """
    Return the smallest number of decimal places in [0..max_decimals]
    such that rounding each value to that precision yields unique strings.
    Falls back to max_decimals if collisions remain.
    """
    # Coerce to float (ignore Nones / non-floats)
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if len(nums) <= 1:
        return 0

    for d in range(max_decimals + 1):
        seen = set(f"{x:.{d}f}" for x in nums)
        if len(seen) == len(nums):
            return d
    return max_decimals


def compute_suffix_decimals(
    params_to_vary: Dict[str, Dict[str, Any]],
    max_decimals: int = 6
) -> Dict[str, int]:
    """
    Inspect params_to_vary["..."]["values"] and choose the fewest decimals
    per *key* needed to keep suffix entries distinct.

    Only applies to scalar numeric values. (Tuples/dicts keep default formatting.)
    """
    plan: Dict[str, int] = {}
    for key, spec in params_to_vary.items():
        vals = spec.get("values", [])
        if vals and all(isinstance(v, (int, float)) for v in vals):
            plan[key] = _min_decimals_for_uniqueness(list(vals), max_decimals)
    return plan


def _format_value_for_key(
    key: str,
    value: Any,
    decimals_plan: Optional[Dict[str, int]] = None
) -> str:
    """
    Like _format_value, but if value is a float and we have a decimals plan
    for this key, use that precision for suffix formatting.
    """
    if isinstance(value, float) and decimals_plan and key in decimals_plan:
        d = decimals_plan[key]
        return f"{value:.{d}f}"
    # fallback to legacy formatting
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, tuple):
        return "-".join(str(x) for x in value)
    if isinstance(value, dict):
        return "_".join(f"{k}-{value[k]}" for k in sorted(value))
    return str(value)

# ---------- small utilities ----------

def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, tuple):
        return "-".join(str(x) for x in value)
    if isinstance(value, dict):
        # Order keys for consistency
        return "_".join(f"{k}-{value[k]}" for k in sorted(value))
    return str(value)

def _frange(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    x = start
    while x <= stop + 1e-9:   # tolerate float rounding
        vals.append(round(x, 10))
        x += step
    return vals

def _copy_props(props: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Deep-copy a dict-of-dicts without sharing references."""
    return {k: deepcopy(v) for k, v in props.items()}

def _set_by_path(d: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set d[path] where path is dot-separated, e.g. 'nexus.syn_density'
    or 'perisomatic.delay_config.delay_shift'.
    """
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value

def _materialize_varied_params(
    *,
    params_to_vary: Dict[str, Dict[str, Any]],
    common_params: Dict[str, Any],
    varied_atomic: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert atomic overrides like:
        {'nexus.syn_density': 0.77, 'perisomatic.delay_config.delay_shift': 6}
    into full dict-valued updates under exc/inh syn_properties in varied_params:
        {'inh_syn_properties': {... 'nexus': {'syn_density': 0.77}, ...,
                                 'perisomatic': {'delay_config': {'delay_shift': 6}}}}
    Rules:
      - Each atomic entry in params_to_vary may have:
            'apply_to' ∈ {'inh_syn_properties','exc_syn_properties'} or omitted.
        If omitted/None, we treat the key as a top-level parameter (pass-through).
      - 'sim_name_suffix' is preserved if present on varied_atomic.
    """
    result: Dict[str, Any] = {}
    rebuilt: Dict[str, Dict[str, Dict[str, Any]]] = {}  # apply_to -> (full props dict)

    for key, value in varied_atomic.items():
        if key == "sim_name_suffix":
            continue

        spec = params_to_vary.get(key, {})
        apply_to = spec.get("apply_to")

        if apply_to in ("inh_syn_properties", "exc_syn_properties"):
            if apply_to not in rebuilt:
                base = common_params.get(apply_to)
                if base is None:
                    raise KeyError(f"{apply_to} missing in common_params.")
                rebuilt[apply_to] = _copy_props(base)
            _set_by_path(rebuilt[apply_to], key, value)
        else:
            # Treat as plain top-level override
            result[key] = value

    # Attach any rebuilt properties
    for apply_to, props in rebuilt.items():
        result[apply_to] = props

    # Keep composed suffix (set by get_*_parameter_combinations)
    if "sim_name_suffix" in varied_atomic:
        result["sim_name_suffix"] = varied_atomic["sim_name_suffix"]

    return result

# Small float-range utility
def _frange(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    x = start
    # tolerate floating rounding at the end
    while x <= stop + 1e-9:
        vals.append(round(x, 10))
        x += step
    return vals

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


def get_parameter_combinations(param_dict: Dict[str, Dict[str, Any]],
                               decimals_plan: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """
    Cartesian-product combinations for params_to_vary.
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
                        suffix_values.append(_format_value_for_key(key, v, decimals_plan))
                    formatted = "_".join(suffix_values)
                else:
                    formatted = _format_value_for_key(key, value, decimals_plan)
                suffix_parts.append(f"{param_dict[key]['sim_name_suffix']}{formatted}")

        combo["sim_name_suffix"] = "_".join(suffix_parts) if suffix_parts else ""
        combos.append(combo)

    return combos


def get_index_matched_parameter_combinations(
    param_dict: Dict[str, Dict[str, Any]],
    decimals_plan: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Index-matched combinations (all varying lists must have equal length).
    Uses decimals_plan to format float suffixes with the minimum precision
    needed for uniqueness per key.
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
                    # Build suffix from nested keys, using decimals_plan per *top-level* key k
                    suffix_values = []
                    for nk in param_dict[k]["nested_keys"]:
                        v = value
                        for step in nk.split("."):
                            v = v[step]
                        suffix_values.append(_format_value_for_key(k, v, decimals_plan))
                    formatted = "_".join(suffix_values)
                else:
                    formatted = _format_value_for_key(k, value, decimals_plan)

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
    if type(morphology_params) is dict: # use morphology_params["base_sim_name"], but if morphology_params is a string then just use it.
        name_parts.append(morphology_params.get("base_sim_name", ""))
    elif type(morphology_params) is str:
        name_parts.append(morphology_params)
    else:
        raise(ValueError(f"morphology_params {morphology_params} is neither dict nor str, instead {type(morphology_params)}"))

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
    neuron_random_states: Iterable[Optional[int]],
    numpy_random_states: Iterable[int],
    params_to_vary: Dict[str, Dict[str, Any]],
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
    """
    # Build varied-parameter combinations from the concise params_to_vary
    decimals_plan = compute_suffix_decimals(params_to_vary)

    varied_list = (
        get_index_matched_parameter_combinations(params_to_vary, decimals_plan=decimals_plan)
        if index_matched else
        get_parameter_combinations(params_to_vary, decimals_plan=decimals_plan)
    )
    # Decide the "modifier" dimension once (CI amps / EXC FR increases / or nothing)
    if common_params.get("CI_on", False):
        modifier_kind = "amp"
        modifier_values = _frange(-0.5, 2.1, 0.5)
    elif common_params.get("exc_constant_fr", False):
        modifier_kind = "exc_inc"
        modifier_values = _frange(0.0, 8.1, 2.0)
    else:
        modifier_kind = None
        modifier_values = [None]

    parameters_list: List[HayParameters] = []

    for np_seed in numpy_random_states:
        for nrn_seed in neuron_random_states:
            for m_name in morphologies_to_use:
                for sr_name in syn_reductions_to_use:
                    for ci_name in ci_replacements_to_use:
                        morphology_params  = morphologies[m_name] if m_name in morphologies.keys() else m_name
                        syn_reduct_params  = syn_reductions[sr_name]
                        ci_replace_params  = ci_replacements[ci_name]

                        for varied_atomic in varied_list:
                            # Expand concise atomics into full dict updates (exc/inh props)
                            varied_params = _materialize_varied_params(
                                params_to_vary=params_to_vary,
                                common_params=common_params,
                                varied_atomic=varied_atomic,
                            )

                            for mod in modifier_values:
                                amp_arg: Optional[float] = None
                                exc_inc_arg: Optional[float] = None
                                if modifier_kind == "amp":
                                    amp_arg = mod
                                elif modifier_kind == "exc_inc":
                                    exc_inc_arg = mod

                                parameters = create_parameters(
                                    numpy_seed=np_seed,
                                    neuron_seed=nrn_seed,
                                    sim_type=sim_type,
                                    common_params=common_params,
                                    morphology_params=morphology_params,
                                    syn_reduction_params=syn_reduct_params,
                                    ci_replacement_params=ci_replace_params,
                                    varied_params=varied_params,
                                    amp=amp_arg,
                                    excFR_increase=exc_inc_arg,
                                )
                                parameters_list.append(parameters)

    return parameters_list
