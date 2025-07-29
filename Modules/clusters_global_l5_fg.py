from terminal_branch_statistics import branch_stats
import numpy as np
from functools import partial
exc_clustering = {}

for sec_type, branches in branch_stats.items():
    input_source = f"{sec_type}_local_L5"
    # --- One FG per input_source ---
    all_centers = [stats['center_coords'] for stats in branches.values()]
    # Use a big enough radius to cover all, or a spatial envelope, or just choose one (see NOTE)
    fg_center = np.mean(np.array(all_centers), axis=0)  # Or another center, as you wish
    fg_radius = max(
        np.linalg.norm(np.array(center) - fg_center)
        for center in all_centers
    ) + 1  # Just ensure all are included

    # === Choose dynamic assignment for local_L5 ===
    presynaptic_cells = {
        'mode': 'dynamic',
        # 'max_synapses_per_pc': 6,
        'max_synapses_per_pc': {'dist':partial(np.random.uniform, low=3, high=8+1)},  # Use a dict to specify mean and std
        # optionally: 'assignment_strategy': 'sequential'
    }
    # # Example: If you wanted static, one pc per branch uncomment:
    # presynaptic_cells = []
    # for seg_id, stats in branches.items():
    #     presynaptic_cells.append({
    #         'center': stats['center_coords'],
    #         'radius': stats['total_length'] * 5,
    #         'name': f'PC_{seg_id}',
    #         'max_synapses': 10
    #     })

    # Store under input_source (not sec_type)
    exc_clustering[input_source] = {
        'functional_groups': [{
            'center': fg_center.tolist(),
            'radius': fg_radius,
            'modulation_mode': 'pink_noise',
            'input_source': input_source,
            'presynaptic_cells': presynaptic_cells
        }]
    }
# Example usage: print a preview for each section type
if __name__ == '__main__':
    for sec_type, fg_data in exc_clustering.items():
        print(f"\nSection type: {sec_type}, #groups={len(fg_data['functional_groups'])}")
        for fg in fg_data['functional_groups'][:2]:  # Show only first 2 per type
            print(f"  FG center: {fg['center']}, radius: {fg['radius']}, PC: {fg['presynaptic_cells'][0]['name']}")

# Optionally, if you want to export inh_clustering, just follow similar logic as below (commented out):
#
# inh_clustering = {}
# for sec_type, branches in branch_stats.items():
#     inh_clustering[sec_type] = {
#         'functional_groups': []
#     }
#     for seg_id, stats in branches.items():
#         center = stats['center_coords']
#         radius = stats['total_length'] / 2  # (for inhibitory, as in your example)
#         functional_group = {
#             'center': center,
#             'radius': radius,
#             'presynaptic_cells': [
#                 {
#                     'center': center,
#                     'radius': radius / 2,
#                     'name': f'PC_{seg_id}',
#                     'max_synapses': 10
#                 }
#             ]
#         }
#         inh_clustering[sec_type]['functional_groups'].append(functional_group)
#
# __all__ = ['exc_clustering', 'inh_clustering']

__all__ = ['exc_clustering']