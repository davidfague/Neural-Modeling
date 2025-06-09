from dataclasses import dataclass, field
from branch_statistics_combined import combined_branch_statistics

# Create excitatory clustering based on branch statistics
exc_clustering = { # TODO: will not be used on only tuft. Need to update combined_branch_statistics.py to include all branches with keys by section type..
    'tuft': {
        'functional_groups': []
    }
}

# Create functional groups from branch statistics
for branch_id, stats in combined_branch_statistics.items():
    center = stats['center_coords']
    radius = stats['total_length'] / 4  # Use half the branch length as radius
    
    # Create a functional group for each branch
    functional_group = {
        'center': center,
        'radius': radius,
        'presynaptic_cells': [
            {
                'center': center,  # Use the same center as the functional group
                'radius': radius / 1,  # Half the functional group radius
                'name': f'PC_{branch_id}',
                'max_synapses': 10
            }
        ]
    }
    
    exc_clustering['tuft']['functional_groups'].append(functional_group)

# # Create inhibitory clustering with same structure
# inh_clustering = {
#     'perisomatic': {
#         'functional_groups': []
#     }
# }

# # Copy the same structure for inhibitory clustering
# for branch_id, stats in combined_branch_statistics.items():
#     center = stats['center_coords']
#     radius = stats['total_length'] / 2
    
#     functional_group = {
#         'center': center,
#         'radius': radius,
#         'presynaptic_cells': [
#             {
#                 'center': center,  # Use the same center as the functional group
#                 'radius': radius / 2,
#                 'name': f'PC_{branch_id}',
#                 'max_synapses': 10
#             }
#         ]
#     }
    
#     inh_clustering['perisomatic']['functional_groups'].append(functional_group)

# Export the configurations
__all__ = ['exc_clustering', 'inh_clustering'] 