"""
Clustering module for functional groups and presynaptic cells.

This module provides different clustering strategies for organizing synapses
into functional groups (FGs) and presynaptic cells (PCs):

1. "terminal_branch_simple": One FG per terminal branch with static PC assignment
2. "terminal_branch_fps": Advanced FPS-based clustering with dynamic PC assignment
3. Custom: User can provide their own clustering configuration

Both strategies operate on terminal branch statistics computed from segment_data.csv.
"""

import sys
import numpy as np
from functools import partial

# Terminal branch statistics should be imported from the simulation directory
# The default import here is a fallback; actual usage will pass branch_stats as parameter
try:
    sys.path.append('../terminal_branching_coords_for_clusters')
    from terminal_branch_statistics import branch_stats as DEFAULT_BRANCH_STATS
except ImportError:
    DEFAULT_BRANCH_STATS = {}
    print("Warning: Could not import default terminal_branch_statistics. Use compute_terminal_branch_stats() first.")


# ============================================================================
# Divergence specifications (for dynamic PC assignment)
# ============================================================================

EXC_DIVERGENCE_SPEC = {"kind": "uniform_int", "low": 2, "high": 8}

def divergence_spec_for_input_source(input_source: str, syn_type: str):
    """
    Return a PICKLABLE dict describing how to sample #synapses per PC.
    """
    name = (input_source or "").lower()

    if syn_type == "exc":
        return {'kind': 'uniform_int', 'low': 2, 'high': 8}

    # inhibitory by location
    if ("perisomatic" in name) or ("peri" in name and "perisomatic" not in name):
        return {'kind': 'truncnorm_int', 'mean': 2.8, 'sd': 1.9, 'low': 1, 'high': 5}

    if "basal" in name:
        return {'kind': 'truncnorm_int', 'mean': 2.7, 'sd': 1.6, 'low': 1, 'high': 5}

    if any(k in name for k in ("apic", "tuft", "nexus", "oblique", "trunk")):
        return {'kind': 'truncnorm_int', 'mean': 12.0, 'sd': 3.0, 'low': 6, 'high': 18}

    raise ValueError(f"[divergence] No divergence rule for input_source='{input_source}' (syn_type={syn_type}).")


# ============================================================================
# MODE 1: Simple terminal branch clustering (one FG per terminal branch)
# ============================================================================

def build_simple_terminal_branch_clustering(
    branch_stats,
    synapse_type='exc',
    input_source_suffixes=('_local_L5',),
    radius_scale=5.0,
    max_synapses_per_pc=10,
    modulation_mode='pink_noise'
):
    """
    Simple clustering: One functional group per terminal branch.
    Each FG has one PC centered at the branch center.
    
    Args:
        branch_stats: Dict {sec_type: {seg_id: {'center_coords': [x,y,z], 'total_length': float}}}
        synapse_type: 'exc' or 'inh'
        input_source_suffixes: Tuple of suffixes to append to sec_type for input_source naming
        radius_scale: Multiplier for branch length to determine FG radius
        max_synapses_per_pc: Maximum synapses per presynaptic cell
        modulation_mode: Spike train modulation mode
        
    Returns:
        Dict with clustering configuration
    """
    clustering = {}
    
    for sec_type, branches in branch_stats.items():
        clustering[sec_type] = {'functional_groups': []}
        
        for seg_id, stats in branches.items():
            center = stats['center_coords']
            radius = stats['total_length'] * radius_scale
            
            for suffix in input_source_suffixes:
                input_source = f"{sec_type}{suffix}"
                
                functional_group = {
                    'center': center,
                    'radius': radius,
                    'modulation_mode': modulation_mode,
                    'input_source': input_source,
                    'presynaptic_cells': [
                        {
                            'center': center,
                            'radius': radius,
                            'name': f'PC_{seg_id}',
                            'max_synapses': max_synapses_per_pc
                        }
                    ]
                }
                clustering[sec_type]['functional_groups'].append(functional_group)
    
    return clustering


# ============================================================================
# MODE 2: FPS-based terminal branch clustering (advanced)
# ============================================================================

def _fps_reduce(centers, k):
    """
    Farthest Point Sampling to select k representative centers from a set of points.
    """
    if k <= 0 or len(centers) == 0:
        return np.empty((0, 3))
    if k >= len(centers):
        return centers.copy()

    chosen_idx = [0]
    chosen = [centers[0]]
    remain = list(range(1, len(centers)))

    for _ in range(k - 1):
        if not remain:
            break
        dists_to_chosen = []
        for r_idx in remain:
            min_dist = min(np.linalg.norm(centers[r_idx] - c) for c in chosen)
            dists_to_chosen.append(min_dist)
        farthest = remain[int(np.argmax(dists_to_chosen))]
        chosen_idx.append(farthest)
        chosen.append(centers[farthest])
        remain.remove(farthest)

    return np.array(chosen)


def _branch_centers(branches_dict):
    """Return (seg_ids, centers[N,3]) for a {seg_id: {'center_coords':[...]}} mapping."""
    if not branches_dict:
        return [], np.empty((0, 3))
    seg_ids = list(branches_dict.keys())
    centers = np.array([branches_dict[sid]['center_coords'] for sid in seg_ids])
    return seg_ids, centers


def _radius_from_branch(stat, scale=7.5):
    return stat['total_length'] * scale


def build_fps_terminal_branch_clustering(
    branch_stats,
    per_branch_suffixes=("_local_L5", "_local_L23", "_distant"),
    branch_radius_scale=7.5,
    default_modulation_mode="pink_noise",
    divergence_spec=EXC_DIVERGENCE_SPEC,
    target_fgs_per_input_source=None,
    global_target_fgs=None,
    pc_locality: str = "nearest"
):
    """
    FPS-based clustering with dynamic PC assignment.
    Uses Farthest Point Sampling to distribute functional groups across terminal branches.
    
    Args:
        branch_stats: Dict of terminal branch statistics
        per_branch_suffixes: Suffixes to create multiple input sources per sec_type
        branch_radius_scale: Scale factor for FG radius based on branch length
        default_modulation_mode: Default spike train modulation
        divergence_spec: Specification for dynamic PC divergence
        target_fgs_per_input_source: Dict mapping input_source to desired # of FGs
        global_target_fgs: Total target FGs across all input sources (alternative to above)
        pc_locality: 'nearest' for spatial locality, None for sequential assignment
        
    Returns:
        Dict with clustering configuration
    """
    clustering = {}

    for sec_type, branches in branch_stats.items():
        if not branches:
            continue

        seg_ids, centers = _branch_centers(branches)
        n_branches = len(seg_ids)

        for suffix in per_branch_suffixes:
            input_source = f"{sec_type}{suffix}"

            # Determine how many FGs for this input_source
            if target_fgs_per_input_source and input_source in target_fgs_per_input_source:
                n_fg = target_fgs_per_input_source[input_source]
            elif global_target_fgs:
                n_total_input_sources = len(branch_stats) * len(per_branch_suffixes)
                n_fg = max(1, global_target_fgs // n_total_input_sources)
            else:
                n_fg = n_branches  # one FG per branch (default)

            n_fg = min(n_fg, n_branches)

            # Use FPS to select representative centers
            if n_fg < n_branches:
                selected_centers = _fps_reduce(centers, n_fg)
            else:
                selected_centers = centers

            # Create FGs - key by input_source to match legacy behavior
            if input_source not in clustering:
                clustering[input_source] = {'functional_groups': []}

            for i, fg_center in enumerate(selected_centers):
                # Find nearest branch to this FG center for radius estimation
                dists = np.linalg.norm(centers - fg_center, axis=1)
                nearest_branch_idx = int(np.argmin(dists))
                nearest_seg_id = seg_ids[nearest_branch_idx]
                fg_radius = _radius_from_branch(branches[nearest_seg_id], scale=branch_radius_scale)

                fg = {
                    'center': fg_center.tolist(),
                    'radius': float(fg_radius),
                    'modulation_mode': default_modulation_mode,
                    'input_source': input_source,
                    'presynaptic_cells': {
                        'mode': 'dynamic',
                        'max_synapses_per_pc': {'dist': divergence_spec},
                        'locality': pc_locality
                    }
                }
                clustering[input_source]['functional_groups'].append(fg)

    return clustering


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
    Create a single global inhibitory FG that covers all branches.
    
    Args:
        branch_stats: Dict of terminal branch statistics
        default_modulation_mode: Spike train modulation mode
        global_center: Override center position
        global_radius: Override radius
        pad: Padding factor for radius calculation
        pc_locality: 'nearest' for spatial PC assignment
        
    Returns:
        Dict with clustering configuration
    """
    if global_center is None or global_radius is None:
        c, r = _compute_global_center_radius(branch_stats, pad=pad)
        if global_center is None:
            global_center = c
        if global_radius is None:
            global_radius = r

    clustering = {}

    for sec_type in branch_stats.keys():
        divergence_cfg = rng_dist if rng_dist else {'kind': 'truncnorm_int', 'mean': 12.0, 'sd': 3.0, 'low': 6, 'high': 18}

        input_source = f"{sec_type}_perisomatic"
        fg = {
            'center': list(global_center),
            'radius': float(global_radius),
            'modulation_mode': default_modulation_mode,
            'input_source': input_source,
            'presynaptic_cells': {
                'mode': 'dynamic',
                'max_synapses_per_pc': {'dist': divergence_cfg},
                'locality': pc_locality
            }
        }

        if sec_type not in clustering:
            clustering[sec_type] = {'functional_groups': []}
        clustering[sec_type]['functional_groups'].append(fg)

    return clustering


def _compute_global_center_radius(branch_stats, pad=1.0):
    """Compute bounding sphere for all terminal branches."""
    all_coords = []
    for branches in branch_stats.values():
        for stats in branches.values():
            all_coords.append(stats['center_coords'])

    if not all_coords:
        return np.array([0., 0., 0.]), 1000.0

    pts = np.array(all_coords)
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=0)
    radius = dists.max() + pad
    return center, radius


# ============================================================================
# Main clustering factory function
# ============================================================================

def build_clustering(
    mode='terminal_branch_fps',
    synapse_type='exc',
    branch_stats=None,
    **kwargs
):
    """
    Factory function to build clustering configuration based on mode.
    
    Args:
        mode: Clustering strategy
            - 'terminal_branch_simple': One FG per terminal branch
            - 'terminal_branch_fps': FPS-based clustering with dynamic PCs
            - 'global_inh': Single global inhibitory FG
        synapse_type: 'exc' or 'inh'
        branch_stats: Terminal branch statistics dict (if None, uses DEFAULT_BRANCH_STATS)
        **kwargs: Additional arguments passed to specific clustering function
        
    Returns:
        Clustering configuration dict
    """
    if branch_stats is None:
        branch_stats = DEFAULT_BRANCH_STATS
        
    if not branch_stats:
        raise ValueError("branch_stats is empty. Run compute_terminal_branch_stats() first.")
    
    if mode == 'terminal_branch_simple':
        return build_simple_terminal_branch_clustering(
            branch_stats=branch_stats,
            synapse_type=synapse_type,
            **kwargs
        )
    elif mode == 'terminal_branch_fps':
        if synapse_type == 'inh' and kwargs.get('use_global_inh', False):
            return build_inh_clustering_one_global_for_all(
                branch_stats=branch_stats,
                **kwargs
            )
        else:
            return build_fps_terminal_branch_clustering(
                branch_stats=branch_stats,
                **kwargs
            )
    elif mode == 'global_inh':
        return build_inh_clustering_one_global_for_all(
            branch_stats=branch_stats,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")


# ============================================================================
# Backward compatibility: Generate default clusterings
# ============================================================================

def get_default_exc_clustering(mode='terminal_branch_fps', branch_stats=None):
    """Get default excitatory clustering configuration."""
    if branch_stats is None:
        branch_stats = DEFAULT_BRANCH_STATS
        
    if mode == 'terminal_branch_fps':
        return build_fps_terminal_branch_clustering(
            branch_stats,
            pc_locality="nearest",
            target_fgs_per_input_source={
                "tuft_local_L23": 4,
                "tuft_local_L5": 3,
                "tuft_distant": 12,
                "distal_basal_local_L5": 10,
                "distal_basal_local_L23": 2,
                "distal_basal_distant": 3,
                "oblique_local_L5": 4,
                "oblique_local_L23": 2,
                "oblique_distant": 3,
                "trunk_local_L5": 3,
                "trunk_local_L23": 2,
                "trunk_distant": 2,
                "nexus_local_L5": 1,
                "nexus_local_L23": 1,
                "nexus_distant": 2,
            }
        )
    else:
        return build_simple_terminal_branch_clustering(
            branch_stats,
            synapse_type='exc',
            input_source_suffixes=('_local_L5',),
            radius_scale=5.0
        )


def get_default_inh_clustering(mode='global_inh', branch_stats=None):
    """Get default inhibitory clustering configuration."""
    if branch_stats is None:
        branch_stats = DEFAULT_BRANCH_STATS
        
    return build_inh_clustering_one_global_for_all(branch_stats)


# ============================================================================
# Export default configurations for backward compatibility
# ============================================================================

try:
    exc_clustering = get_default_exc_clustering(mode='terminal_branch_fps')
    inh_clustering = get_default_inh_clustering(mode='global_inh')
except Exception as e:
    print(f"Warning: Could not generate default clusterings: {e}")
    exc_clustering = {}
    inh_clustering = {}

__all__ = [
    'build_clustering',
    'build_simple_terminal_branch_clustering',
    'build_fps_terminal_branch_clustering',
    'build_inh_clustering_one_global_for_all',
    'get_default_exc_clustering',
    'get_default_inh_clustering',
    'exc_clustering',
    'inh_clustering',
    'EXC_DIVERGENCE_SPEC',
    'divergence_spec_for_input_source',
]
