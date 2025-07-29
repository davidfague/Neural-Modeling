from terminal_branch_statistics import branch_stats

exc_clustering = {}

# for input_source_type in ['local_L5', 'distant']:
for sec_type, branches in branch_stats.items():
    exc_clustering[sec_type] = {
        'functional_groups': []
    }
    for seg_id, stats in branches.items():
        center = stats['center_coords']
        radius = stats['total_length'] * 5  # Functional group radius relative to branch length (* 25 because branch length got messed up and is only 1 segment long it seems.)

        functional_group = {
            'center': center,
            'radius': radius,
            'modulation_mode': 'pink_noise',
            'input_source': f"{sec_type}_local_L5",
            'presynaptic_cells': [
                {
                    'center': center,
                    'radius': radius,
                    'name': f'PC_{seg_id}',
                    'max_synapses': 10
                }
            ]
        }
        exc_clustering[sec_type]['functional_groups'].append(functional_group)

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
