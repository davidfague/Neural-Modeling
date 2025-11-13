import sys
sys.path.append('../terminal_branching_coords_for_clusters')
from terminal_branch_statistics import branch_stats
import numpy as np
from functools import partial

# --- picklable divergence spec for EXC (uniform 2–8) ---
EXC_DIVERGENCE_SPEC = {"kind": "uniform_int", "low": 2, "high": 8}

def divergence_spec_for_input_source(input_source: str, syn_type: str):
    """
    Return a PICKLABLE dict describing how to sample #synapses per PC.
    """
    name = (input_source or "").lower()

    if syn_type == "exc":
        # Uniform 2–8
        return {'kind': 'uniform_int', 'low': 2, 'high': 8}

    # inhibitory by location
    if ("perisomatic" in name) or ("peri" in name and "perisomatic" not in name):
        return {'kind': 'truncnorm_int', 'mean': 2.8, 'sd': 1.9, 'low': 1,  'high': 5}

    if "basal" in name:
        return {'kind': 'truncnorm_int', 'mean': 2.7, 'sd': 1.6, 'low': 1,  'high': 5}

    if any(k in name for k in ("apic", "tuft", "nexus", "oblique", "trunk")):
        return {'kind': 'truncnorm_int', 'mean': 12.0, 'sd': 3.0, 'low': 6, 'high': 18}

    raise ValueError(f"[divergence] No divergence rule for input_source='{input_source}' (syn_type={syn_type}).")

def build_exc_clustering(branch_stats, mode="one_fg_per_input_source", rng_dist=None):
    """
    mode:
      - "one_fg_per_input_source": 1 FG that encloses all branches of a sec_type
      - "one_fg_per_branch":       1 FG per branch (what you remember using before)
    rng_dist: optional callable for dynamic max_synapses_per_pc (e.g., partial(np.random.uniform, low=3, high=9))
    """
    exc_clustering = {}
    # if rng_dist is None:
    #     rng_dist = partial(np.random.uniform, low=3, high=9)

    exc_clustering = {}

    for sec_type, branches in branch_stats.items():
        input_source = f"{sec_type}_local_L5"

        if not branches:
            continue

        # choose divergence sampler per input_source (uniform 2–8), unless user overrides
        dist_sampler = rng_dist or divergence_spec_for_input_source(input_source, syn_type="exc")

        if mode == "one_fg_per_input_source":
            # Center at mean, radius encloses all branch centers
            all_centers = np.array([stats['center_coords'] for stats in branches.values()])
            fg_center = np.mean(all_centers, axis=0)
            fg_radius = max(np.linalg.norm(c - fg_center) for c in all_centers) + 1.0

            fg = {
                'center': fg_center.tolist(),
                'radius': float(fg_radius),
                'modulation_mode': 'pink_noise',
                'input_source': input_source,
                'presynaptic_cells': {
                    'mode': 'dynamic',
                    'max_synapses_per_pc': {'dist': dist_sampler},
                },
            }
            exc_clustering[input_source] = {'functional_groups': [fg]}

        elif mode == "one_fg_per_branch":
            if input_source not in ['tuft_local_L5', 'distal_basal_local_L5']:
                continue
            # Many FGs, each one centered on an individual branch
            fgs = []
            for seg_id, stats in branches.items():
                center = np.array(stats['center_coords'])
                # choose your own radius heuristic here:
                radius = float(stats.get('total_length', 1.0)) * 7.5

                # (i) Dynamic PCs inside each FG:
                pcs = {
                    'mode': 'dynamic',
                    'max_synapses_per_pc': {'dist': dist_sampler},
                }

                # (ii) If you wanted static PCs instead, swap pcs to a list:
                # pcs = [{
                #     'center': center.tolist(),
                #     'radius': radius * 0.5,
                #     'name': f'PC_{seg_id}',
                #     'max_synapses': 10,
                # }]

                fgs.append({
                    'center': center.tolist(),
                    'radius': radius,
                    'modulation_mode': 'pink_noise',
                    'input_source': input_source,
                    'presynaptic_cells': pcs,
                })

            exc_clustering[input_source] = {'functional_groups': fgs}

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return exc_clustering

def _branch_centers(branches_dict):
    """Return (seg_ids, centers[N,3]) for a {seg_id: {'center_coords':[...]}, ...} mapping."""
    if not branches_dict:
        return [], np.zeros((0,3), dtype=float)
    seg_ids = list(branches_dict.keys())
    centers = np.array([branches_dict[s]["center_coords"] for s in seg_ids], dtype=float)
    return seg_ids, centers

def _radius_from_branch(stat, scale=7.5):
    """Your per-branch radius heuristic."""
    return float(stat.get("total_length", 1.0)) * float(scale)

def _fps_reduce(centers, k):
    """
    Farthest-Point Sampling (greedy) to pick k centers from points (N,3).
    Returns indices of chosen 'centers' and assignment of each point to nearest chosen center.
    """
    N = centers.shape[0]
    if N == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    k = max(1, min(k, N))

    chosen = np.zeros(k, dtype=int)
    # seed: farthest from mean
    centroid = centers.mean(axis=0, keepdims=True)
    d2c = np.linalg.norm(centers - centroid, axis=1)
    chosen[0] = int(np.argmax(d2c))

    min_dist = np.linalg.norm(centers - centers[chosen[0]], axis=1)

    for i in range(1, k):
        next_idx = int(np.argmax(min_dist))
        chosen[i] = next_idx
        # update distances
        new_d = np.linalg.norm(centers - centers[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_d)

    # final assignment to nearest chosen center
    # compute distances to chosen centers (k x N) cheaply
    dists = np.stack([np.linalg.norm(centers - centers[c], axis=1) for c in chosen], axis=0)  # (k,N)
    assign = np.argmin(dists, axis=0)  # which chosen center each point goes to (0..k-1)
    return chosen, assign

def build_exc_clustering_v2(
    branch_stats,
    per_branch_suffixes=("_local_L5", "_local_L23", "_distant"),
    branch_radius_scale=7.5,
    default_modulation_mode="pink_noise",
    divergence_spec=EXC_DIVERGENCE_SPEC,
    # OPTIONAL CAPPING:
    target_fgs_per_input_source=None,   # e.g., {"tuft_local_L23": 10, "distal_basal_distant": 8}
    global_target_fgs=None,              # e.g., 50 total across *all* sources; overrides per-input if set
    pc_locality: str = "nearest"
):
    """
    Build EXC clustering with per-branch functional groups for the given suffixes.
    Optionally cap the number of FGs per input_source or globally.

    branch_stats: {sec_type: {seg_id: {'center_coords': [x,y,z], 'total_length': float, ...}, ...}, ...}
    """
    exc_clustering = {}

    # Collect (for optional global capping) all points grouped by input_source
    # input_source = f"{sec_type}{suffix}"
    per_source_points = {}  # input_source -> dict(seg_ids=..., centers=..., stats_per_seg=...)

    for sec_type, branches in branch_stats.items():
        if not branches:
            continue

        seg_ids, centers = _branch_centers(branches)
        if centers.shape[0] == 0:
            continue

        # Pre-store per sec_type so we can reuse the same branch set for each suffix
        for suffix in per_branch_suffixes:
            input_source = f"{sec_type}{suffix}"
            per_source_points[input_source] = {
                "seg_ids": seg_ids,
                "centers": centers,
                "stats": branches,  # the raw dict so we can get total_length later
            }

    # If global cap is requested, we need to compute per-source K to hit approximately that total.
    per_source_caps = {}
    if global_target_fgs is not None and len(per_source_points):
        # naive split: proportional to #branches in that source
        total_branches = sum(len(v["seg_ids"]) for v in per_source_points.values())
        if total_branches == 0:
            total_branches = 1
        # guarantee at least 1 FG per source
        remaining = global_target_fgs
        keys = list(per_source_points.keys())
        for i, key in enumerate(keys):
            n_br = len(per_source_points[key]["seg_ids"])
            # proportional share (round)
            k_i = max(1, int(round(global_target_fgs * (n_br / total_branches))))
            # last source fixes rounding drift
            if i == len(keys) - 1:
                k_i = max(1, remaining)
            remaining -= k_i
            per_source_caps[key] = k_i
        # sanity: if rounding went weird, fix to at least 1
        for k in keys:
            per_source_caps[k] = max(1, per_source_caps.get(k, 1))

    for input_source, blob in per_source_points.items():
        seg_ids = blob["seg_ids"]
        centers = blob["centers"]
        stats = blob["stats"]

        # Decide how many FGs we want for this input_source
        if global_target_fgs is not None:
            K = per_source_caps[input_source]
        elif isinstance(target_fgs_per_input_source, dict) and input_source in target_fgs_per_input_source:
            K = int(target_fgs_per_input_source[input_source])
        else:
            # default: pure per-branch FGs (one FG per branch)
            K = len(seg_ids)

        # If K >= #branches: old behavior (one FG per branch)
        if K >= len(seg_ids):
            fgs = []
            for seg_id in seg_ids:
                c = np.array(stats[seg_id]["center_coords"], dtype=float)
                r = _radius_from_branch(stats[seg_id], branch_radius_scale)
                fgs.append({
                    "center": c.tolist(),
                    "radius": float(r),
                    "modulation_mode": default_modulation_mode,
                    "input_source": input_source,
                    "presynaptic_cells": {
                        "mode": "dynamic",
                        "max_synapses_per_pc": {"dist": divergence_spec},
                        "locality": pc_locality,
                    },
                })
            exc_clustering[input_source] = {"functional_groups": fgs}
            continue

        # Else: reduce to K groups with farthest-point sampling, then define each FG
        chosen, assign = _fps_reduce(centers, k=K)

        fgs = []
        for k_idx in range(K):
            member_mask = (assign == k_idx)
            member_idx  = np.where(member_mask)[0]
            member_centers = centers[member_mask]
            if member_centers.shape[0] == 0:
                continue
            # FG center = mean of member branch centers
            fg_center = member_centers.mean(axis=0)
            # FG radius = max distance from fg_center to member centers, padded by 1.0
            per_branch_radii = [
                _radius_from_branch(stats[seg_ids[j]], branch_radius_scale)  # e.g. total_length * 7.5
                for j in member_idx
            ]
            dists = np.linalg.norm(member_centers - fg_center, axis=1)
            fg_radius = float(np.max(dists + np.asarray(per_branch_radii)) * 1.05)  # 5% pad
            fgs.append({
                "center": fg_center.tolist(),
                "radius": fg_radius,
                "modulation_mode": default_modulation_mode,
                "input_source": input_source,
                "presynaptic_cells": {
                    "mode": "dynamic",
                    "max_synapses_per_pc": {"dist": divergence_spec},
                    "locality": pc_locality,
                },
            })

        exc_clustering[input_source] = {"functional_groups": fgs}

    return exc_clustering

def build_inh_clustering(
    branch_stats,
    mode: str = "one_fg_per_input_source",
    rng_dist=None,
    allowed_input_sources_for_per_branch=None,
    radius_pad: float = 1.0,
    branch_radius_scale: float = 7.5,
    default_modulation_mode: str = "pink_noise",
    global_center=None,
    global_radius=None,
):
    """
    Build *inhibitory* clustering dict in one of three modes:

    Modes
    -----
    1) "one_fg_per_input_source"
        • For each sec_type (your inh input_source label), create ONE FG that
          encloses all branch centers for that sec_type (mean center + padded radius).

    2) "one_fg_per_branch"
        • For each (sec_type, branch) pair, create ONE FG centered on that branch.
        • Optionally restrict to a subset of sec_types with `allowed_input_sources_for_per_branch`.

    3) "global_from_exc"
        • For each sec_type, create ONE *global* FG (very large radius or user-provided)
          so *every inhibitory synapse* falls into that FG.
        • Pair this with params.inh_syn_properties[...]['spike_train_mode'] = 'delay'
          and delay_config using 'all' to reference ALL excitatory spike trains.

    Parameters
    ----------
    branch_stats : dict
        {sec_type: {seg_id: {'center_coords': [x,y,z], 'total_length': float, ...}, ...}, ...}
    mode : str
        One of {"one_fg_per_input_source", "one_fg_per_branch", "global_from_exc"}.
    rng_dist : callable | None
        Optional distribution for dynamic PC chunking. If None: Uniform[3,9).
        Example: partial(np.random.uniform, low=9, high=20)
    allowed_input_sources_for_per_branch : set|list|None
        If provided and mode == "one_fg_per_branch", only generate FGs for those sec_types.
    radius_pad : float
        Added padding to enclosing radius in "one_fg_per_input_source".
    branch_radius_scale : float
        FG radius = total_length * branch_radius_scale in "one_fg_per_branch".
    default_modulation_mode : str
        FG-level modulation flag; actual spike-train behavior still comes from params.
    global_center : list|tuple|np.ndarray|None
        If provided in "global_from_exc", use this FG center instead of [0,0,0].
    global_radius : float|None
        If provided in "global_from_exc", use this radius; else a very large default (1e12).

    Returns
    -------
    dict
        { sec_type: { 'functional_groups': [ FG, FG, ... ] }, ... }
        where each FG is:
        {
          'center': [x,y,z],
          'radius': float,
          'modulation_mode': <str>,
          'input_source': <sec_type>,
          'presynaptic_cells': {
              'mode': 'dynamic',
              'max_synapses_per_pc': {'dist': rng_dist},  # sampler called per chunk
          }
        }
    """

    if isinstance(allowed_input_sources_for_per_branch, list):
        allowed_input_sources_for_per_branch = set(allowed_input_sources_for_per_branch)

    inh_clustering = {}

    for sec_type, branches in branch_stats.items():
        input_source = f"{sec_type}"
        dist_sampler = rng_dist or divergence_spec_for_input_source(input_source, syn_type="inh")


        if mode == "one_fg_per_input_source":
            if not branches:
                # No geometry for this sec_type; skip
                continue
            all_centers = np.array([stats['center_coords'] for stats in branches.values()])
            fg_center = np.mean(all_centers, axis=0)
            fg_radius = max(np.linalg.norm(c - fg_center) for c in all_centers) + float(radius_pad)

            fg = {
                'center': fg_center.tolist(),
                'radius': float(fg_radius),
                'modulation_mode': default_modulation_mode,
                'input_source': input_source,
                'presynaptic_cells': {
                    'mode': 'dynamic',
                    'max_synapses_per_pc': {'dist': dist_sampler},
                },
            }
            inh_clustering[input_source] = {'functional_groups': [fg]}

        elif mode == "one_fg_per_branch":
            if not branches:
                continue
            if (allowed_input_sources_for_per_branch is not None
                and input_source not in allowed_input_sources_for_per_branch):
                # Skip sec_types you don't want per-branch FGs on
                continue

            fgs = []
            for seg_id, stats in branches.items():
                center = np.array(stats['center_coords'])
                total_length = float(stats.get('total_length', 1.0))
                radius = total_length * float(branch_radius_scale)

                fgs.append({
                    'center': center.tolist(),
                    'radius': float(radius),
                    'modulation_mode': default_modulation_mode,
                    'input_source': input_source,
                    'presynaptic_cells': {
                        'mode': 'dynamic',
                        'max_synapses_per_pc': {'dist': dist_sampler},
                    },
                })

            if fgs:
                inh_clustering[input_source] = {'functional_groups': fgs}

        elif mode == "global_from_exc":
            # Build exactly one global FG per inhibitory input_source.
            # This is *spatially* global; to make it *activity*-global-from-exc,
            # set spike_train_mode='delay' with 'all' in delay_config (see examples below).
            c = np.array(global_center if global_center is not None else [0.0, 0.0, 0.0], dtype=float)
            r = float(global_radius if global_radius is not None else 1e12)

            fg = {
                'center': c.tolist(),
                'radius': r,
                'modulation_mode': default_modulation_mode,
                'input_source': input_source,
                'presynaptic_cells': {
                    'mode': 'dynamic',
                    'max_synapses_per_pc': {'dist': dist_sampler},
                },
            }
            inh_clustering[input_source] = {'functional_groups': [fg]}

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return inh_clustering

def _compute_global_center_radius(branch_stats, pad=1.0):
    """
    Compute one center/radius that encloses *all* branch centers across *all* sec_types.
    """
    all_centers = []
    for branches in branch_stats.values():
        if not branches:
            continue
        all_centers.extend([stats['center_coords'] for stats in branches.values()])
    if not all_centers:
        # Fallback if branch_stats is empty (use origin, huge radius)
        return np.array([0.0, 0.0, 0.0]), 1e12

    centers = np.array(all_centers, dtype=float)
    center = centers.mean(axis=0)
    radius = float(np.max(np.linalg.norm(centers - center, axis=1)) + pad)
    return center, radius


def build_inh_clustering_one_global_for_all(
    branch_stats,
    rng_dist=None,
    default_modulation_mode="delay",
    global_center=None,
    global_radius=None,
    pad=2.0,
    pc_locality="nearest",
):
    """
    Build *one global FG* that covers ALL inhibitory synapses, then replicate
    that *same* FG under every inhibitory input_source (sec_type) so your
    current input_source-matching logic assigns everything into it.

    Returns:
      {
        '<sec_type_A>': {'functional_groups': [ FG_global ]},
        '<sec_type_B>': {'functional_groups': [ FG_global ]},
        ...
      }
    """
    inh_clustering = {}
    # One enclosing sphere for *all* inhibitory geometry
    if global_center is None or global_radius is None:
        c, r = _compute_global_center_radius(branch_stats, pad=pad)
    else:
        c, r = np.array(global_center, dtype=float), float(global_radius)

    # Define the single global FG (to be replicated per input_source)
    for sec_type in branch_stats.keys():
        input_source = f"{sec_type}"
        dist_sampler = rng_dist or divergence_spec_for_input_source(input_source, syn_type="inh")
        fg = {
            "center": c.tolist(),
            "radius": r,
            "modulation_mode": default_modulation_mode,
            "input_source": input_source,
            "presynaptic_cells": {
                "mode": "dynamic",
                "max_synapses_per_pc": {"dist": dist_sampler},
                "locality": pc_locality,
            },
        }
        inh_clustering[input_source] = {"functional_groups": [fg]}

    return inh_clustering

# exc_clustering = build_exc_clustering(branch_stats, mode="one_fg_per_branch")
# exc_clustering = build_exc_clustering_v2(
#     branch_stats,
#     global_target_fgs=50  # adjust as you like
# )
exc_clustering = build_exc_clustering_v2(
    branch_stats,
    pc_locality="nearest",
    target_fgs_per_input_source = {
    # TUFT  (L2/3 dominates tuft; L5 modest; distant largest but capped)
    "tuft_local_L23": 4,
    "tuft_local_L5":  3,
    "tuft_distant":   12,

    # DISTAL BASAL (rich L5; smaller L23; modest distant)
    "distal_basal_local_L5": 10,
    "distal_basal_local_L23": 2,#3, # 2
    "distal_basal_distant":   3,

    # OBLIQUE (moderate L5; small L23/distant)
    "oblique_local_L5":  4,
    "oblique_local_L23": 2,
    "oblique_distant":   3,

    # TRUNK (sparse overall; L5 > L23/distant)
    "trunk_local_L5":  3,#4, #3
    "trunk_local_L23": 2,#3, #2
    "trunk_distant":   2,#3, #2

    # NEXUS (very light overall by your densities)
    "nexus_local_L5":  1,#2, # 1
    "nexus_local_L23": 1,#2, # 1
    "nexus_distant":   2,#3, # 2
    }
)

# inh_clustering = build_inh_clustering(
#     branch_stats,
#     mode="one_fg_per_input_source",
#     rng_dist=partial(np.random.uniform, low=9, high=20)
# )
inh_clustering = build_inh_clustering_one_global_for_all(branch_stats)
__all__ = ['exc_clustering', 'inh_clustering']

# # Example usage: print a preview for each section type
# if __name__ == '__main__':
#     for sec_type, fg_data in exc_clustering.items():
#         print(f"\nSection type: {sec_type}, #groups={len(fg_data['functional_groups'])}")
#         for fg in fg_data['functional_groups'][:2]:  # Show only first 2 per type
#             print(f"  FG center: {fg['center']}, radius: {fg['radius']}, PC: {fg['presynaptic_cells'][0]['name']}")