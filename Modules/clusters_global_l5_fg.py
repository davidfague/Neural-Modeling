import sys
sys.path.append('/home/drfrbc/Neural-Modeling/terminal_branching_coords_for_clusters')
from terminal_branch_statistics import branch_stats
import numpy as np
from functools import partial
exc_clustering = {}


def build_exc_clustering(branch_stats, mode="one_fg_per_input_source", rng_dist=None):
    """
    mode:
      - "one_fg_per_input_source": 1 FG that encloses all branches of a sec_type
      - "one_fg_per_branch":       1 FG per branch (what you remember using before)
    rng_dist: optional callable for dynamic max_synapses_per_pc (e.g., partial(np.random.uniform, low=3, high=9))
    """
    if rng_dist is None:
        rng_dist = partial(np.random.uniform, low=3, high=9)

    exc_clustering = {}

    for sec_type, branches in branch_stats.items():
        input_source = f"{sec_type}_local_L5"

        if not branches:
            continue

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
                    'max_synapses_per_pc': {'dist': rng_dist},
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
                    'max_synapses_per_pc': {'dist': rng_dist},
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

from functools import partial
import numpy as np

def build_inh_clustering(branch_stats,
                         mode="one_fg_per_input_source",
                         rng_dist=None,
                         allowed_input_sources_for_per_branch=None,
                         radius_pad=1.0,
                         branch_radius_scale=7.5,
                         default_modulation_mode="pink_noise"):
    """
    Build inhibitory clustering exactly like excitatory.

    Parameters
    ----------
    branch_stats : dict
        {sec_type: {seg_id: {'center_coords': [x,y,z], 'total_length': float, ...}, ...}, ...}
    mode : str
        "one_fg_per_input_source" or "one_fg_per_branch"
    rng_dist : callable or None
        If None, defaults to Uniform[3,9) used to sample dynamic max_synapses_per_pc.
        Example: partial(np.random.uniform, low=3, high=9)
    allowed_input_sources_for_per_branch : set/list or None
        If provided and mode == "one_fg_per_branch", only build FGs for these input_sources.
        (Useful if you only want per-branch FGs on, say, distal_basal/perisomatic, etc.)
    radius_pad : float
        Extra margin for the single-FG radius (enclosing all branches).
    branch_radius_scale : float
        Radius heuristic for per-branch FGs: radius = total_length * branch_radius_scale
    default_modulation_mode : str
        FG-level modulation override; kept same as excitatory default ("pink_noise").
    """
    if rng_dist is None:
        rng_dist = partial(np.random.uniform, low=3, high=9)

    inh_clustering = {}

    for sec_type, branches in branch_stats.items():
        # For inhibition we typically use the raw sec_type as the input_source label.
        # (If you prefer a suffix like "_local_L5", change this line accordingly.)
        input_source = f"{sec_type}"

        if not branches:
            continue

        if mode == "one_fg_per_input_source":
            # Center: mean of branch centers; Radius: encloses all branch centers (+pad)
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
                    'max_synapses_per_pc': {'dist': rng_dist},
                },
            }
            inh_clustering[input_source] = {'functional_groups': [fg]}

        elif mode == "one_fg_per_branch":
            if allowed_input_sources_for_per_branch is not None:
                allowed = set(allowed_input_sources_for_per_branch)
                if input_source not in allowed:
                    # Skip sec_types you don't want per-branch clustering on
                    continue

            fgs = []
            for seg_id, stats in branches.items():
                center = np.array(stats['center_coords'])
                # Same radius heuristic you used for exc:
                total_length = float(stats.get('total_length', 1.0))
                radius = total_length * float(branch_radius_scale)

                pcs = {
                    'mode': 'dynamic',
                    'max_synapses_per_pc': {'dist': rng_dist},
                }

                fgs.append({
                    'center': center.tolist(),
                    'radius': float(radius),
                    'modulation_mode': default_modulation_mode,
                    'input_source': input_source,
                    'presynaptic_cells': pcs,
                })

            if fgs:
                inh_clustering[input_source] = {'functional_groups': fgs}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return inh_clustering


exc_clustering = build_exc_clustering(branch_stats, mode="one_fg_per_branch", rng_dist=partial(np.random.uniform, low=3, high=9))
inh_clustering = build_inh_clustering(
    branch_stats,
    mode="one_fg_per_input_source",
    rng_dist=partial(np.random.uniform, low=9, high=20)
)
__all__ = ['exc_clustering', 'inh_clustering']

# # Example usage: print a preview for each section type
# if __name__ == '__main__':
#     for sec_type, fg_data in exc_clustering.items():
#         print(f"\nSection type: {sec_type}, #groups={len(fg_data['functional_groups'])}")
#         for fg in fg_data['functional_groups'][:2]:  # Show only first 2 per type
#             print(f"  FG center: {fg['center']}, radius: {fg['radius']}, PC: {fg['presynaptic_cells'][0]['name']}")