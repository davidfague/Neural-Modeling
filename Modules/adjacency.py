'''
expecting a nsegxnseg 2d matrix which has a 1 at (i,j) where i is parent of j

if row i has no 1's then seg i has no children.
if column j has no 1's then seg j has no parents.
'''

import numpy as np
import pandas as pd

def find_branching_seg_with_most_branching_descendants_in_subset_y(adjacency_matrix, segment_indices, segment_y_coordinates, min_y=500): # min_y = 500 for Hay et al. 2011 model
    branching_segments = [i for i in segment_indices if sum(adjacency_matrix[i]) > 1 and segment_y_coordinates[i] >= min_y]
    
    max_branching_descendants = -1
    segment_with_most = None
    
    for segment in branching_segments:
        count = get_total_branching_descendants_count(adjacency_matrix, segment)
        if count > max_branching_descendants:
            max_branching_descendants = count
            segment_with_most = segment
            
    return segment_with_most, max_branching_descendants


def get_total_branching_descendants_count(adjacency_matrix, start_segment):
    # Use a helper function to get all descendants
    all_descendants = get_all_descendants(adjacency_matrix, start_segment)
    
    # Count how many of these descendants are branching
    branching_descendants_count = sum(1 for d in all_descendants if sum(adjacency_matrix[d]) > 1)
    
    return branching_descendants_count

def get_all_descendants(adjacency_matrix, start_segment, descendants=None):
    if descendants is None:
        descendants = set()
    for child_index, is_child in enumerate(adjacency_matrix[start_segment]):
        if is_child and child_index not in descendants:
            descendants.add(child_index)
            get_all_descendants(adjacency_matrix, child_index, descendants)
    return list(descendants)

def get_branching_seg_indices(adjacency_matrix):
    # Initialize an empty list to store indices of branching segments
    branching_seg_indices = []
    
    # Iterate through each row and its index in the adjacency matrix
    for i, row in enumerate(adjacency_matrix):
        # Check if the sum of the row is greater than 1 (indicating a branching segment)
        if sum(row) > 1:
            branching_seg_indices.append(i)
            
    return branching_seg_indices

def get_branching_descendants_count(adjacency_matrix, start_segment):
    descendants = get_all_descendants(adjacency_matrix, start_segment)
    # Count how many of these descendants are branching
    return sum(1 for d in descendants if sum(adjacency_matrix[d]) > 1)

def get_all_descendants(adjacency_matrix, start_segment, descendants=None):
    if descendants is None:
        descendants = []

    # Ensure the adjacency matrix is treated as a NumPy array
    adjacency_matrix = np.asarray(adjacency_matrix)

    for i in range(adjacency_matrix.shape[1]):
        # Cast to bool explicitly
        isConnected = bool(adjacency_matrix[start_segment, i])
        if isConnected:
            if i not in descendants:
                descendants.append(i)
                get_all_descendants(adjacency_matrix, i, descendants)

    return descendants

 
def get_terminal_seg_indices(adjacency_matrix):
    # Initialize an empty list to store indices of terminal segments
    terminal_seg_indices = []
    
    # Iterate through each row and its index in the adjacency matrix
    for i, row in enumerate(adjacency_matrix):
        # Check if the sum of the row is 0 (indicating no children)
        if sum(row) == 0:
            # If so, add the index to our list of terminal segment indices
            terminal_seg_indices.append(i)
            
    # Return the list of terminal segment indices
    return terminal_seg_indices    

def find_terminal_descendants(adjacency_matrix, start_index, visited=None, terminal_indices=None):
    if visited is None:
        visited = [False] * adjacency_matrix.shape[0]
    if terminal_indices is None:
        terminal_indices = []

    # Mark the start index as visited to prevent revisiting
    visited[start_index] = True

    # Check if the current segment is a terminal segment
    if all(value == 0 for value in adjacency_matrix[start_index]):
        terminal_indices.append(start_index)
    else:
        # Explore all children of the current segment
        for i, isConnected in enumerate(adjacency_matrix[start_index]):
            if isConnected and not visited[i]:
                # Recursively search for terminal descendants
                find_terminal_descendants(adjacency_matrix, i, visited, terminal_indices)

    return terminal_indices

def find_path_segments(adjacency_matrix, start, end, path=None, visited=None, recursing=False):
    #print(f"FINDING PATH from start: {start} to end: {end}")
    if visited is None:
        visited = [False] * adjacency_matrix.shape[0]  # Assuming adjacency_matrix is a list of lists or similar structure
    if path is None:
        path = []

    # Mark the current node as visited
    visited[start] = True
    path.append(start)

    # If the start is the end, we've found a path
    if start == end:
        return path

    # Recurse on adjacent segments that haven't been visited
    for i, is_connected in enumerate(adjacency_matrix[start]):
        if is_connected == 1 and not visited[i]:  # Ensure that 'is_connected' check is properly done for adjacency matrix
            result = find_path_segments(adjacency_matrix, i, end, path.copy(), visited.copy(), recursing=True)
            if result is not None:
                return result
                
    # If no path is found to the end, indicate with a print statement and return None
    if not recursing:
      print(f"WARNING: No path found from start: {start} to end: {end}.")
    return None

def is_path_exist(adjacency_matrix, start, end, visited=None):
    if visited is None:
        visited = [False] * len(adjacency_matrix)
    
    # Check if we've reached the end segment
    if start == end:
        return True

    # Mark this node as visited to avoid cycles
    visited[start] = True

    # Recursively visit all children of the current segment
    for i, isConnected in enumerate(adjacency_matrix[start]):
        if isConnected and not visited[i]:
            if is_path_exist(adjacency_matrix, i, end, visited):
                return True

    # If no path is found to the end, return False
    return False

def get_divergent_children_of_branching_segments(adjacency_matrix, start, end):
    # Find the path from start to end using the original find_path_segments function
    path = find_path_segments(adjacency_matrix, start, end)
    
    if path is None:
        raise ValueError(f"No path exists between segment {start} and {end}")
    
    divergent_children = []
    
    # We need to make sure the starting segment (path[0]) is never considered, so begin from path[1]
    if len(path) > 1:
        # Start the loop from the second segment (index 1) in the path to exclude the starting segment
        for i in range(1, len(path) - 1):  # Exclude the last segment because it doesn't branch to the end
            current_segment = path[i]
            next_segment_on_path = path[i + 1]
            
            if sum(adjacency_matrix[current_segment]) > 1:  # It's a branching segment
                children = np.where(adjacency_matrix[current_segment] == 1)[0]
                for child in children:
                    if child != next_segment_on_path:
                        divergent_children.append(child)
                    
    return divergent_children

def get_children_indices(parent_idx, adjacency_matrix):
    """
    Get the column indices representing the children of a given parent segment.

    Args:
        parent_idx (int): Row index of the parent segment in the adjacency matrix.
        adjacency_matrix (numpy.ndarray): Directed adjacency matrix.

    Returns:
        List[int]: Column indices representing the children of the parent segment.
    """
    if parent_idx < 0 or parent_idx >= adjacency_matrix.shape[0]:
        raise ValueError("Invalid parent index")

    # Find all column indices where value == 1 (indicating children)
    children_indices = list(np.where(adjacency_matrix[parent_idx, :] == 1)[0])

    return children_indices

def get_all_ascendant_seg_indices(adjacency_matrix, start_segments, ascendants=None): # used for tracing the path back from a terminal segment to the soma
    """
    Get all ascendant (parent, grandparent, etc.) segment indices for the given segment indices.

    Args:
        adjacency_matrix (numpy.ndarray): Directed adjacency matrix
        start_segments (list): List of segment indices to find all ascendants for
        ascendants (dict, optional): Dictionary to store found ascendants. Defaults to None.

    Returns:
        dict: Dictionary mapping each input segment index to its set of all ascendant indices
    """
    if ascendants is None:
        ascendants = {seg: set() for seg in start_segments}
    
    # Ensure the adjacency matrix is treated as a NumPy array
    adjacency_matrix = np.asarray(adjacency_matrix)
    
    for seg_idx in start_segments:
        # Look through each row (parent) in the adjacency matrix
        for parent_idx in range(adjacency_matrix.shape[0]):
            # If this parent connects to our segment, it's an ascendant
            if adjacency_matrix[parent_idx, seg_idx] == 1:
                # Initialize the set for parent_idx if it doesn't exist
                if parent_idx not in ascendants:
                    ascendants[parent_idx] = set()
                
                if parent_idx not in ascendants[seg_idx]:
                    ascendants[seg_idx].add(parent_idx)
                    # Recursively find ascendants of this parent
                    get_all_ascendant_seg_indices(adjacency_matrix, [parent_idx], ascendants)
                    # Add all ascendants of the parent to the current segment's ascendants
                    ascendants[seg_idx].update(ascendants[parent_idx])
    
    return ascendants

def get_all_ascendant_seg_indices_of_type(adjacency_matrix, start_segments, seg_data, sec_type_to_get): # used for tracing the path back from a terminal segment to the start of this section type (identifying a branch or path to root)
    """
    Get all ascendant segment indices filtered by section type.

    Args:
        adjacency_matrix (numpy.ndarray): Directed adjacency matrix
        start_segments (list): List of segment indices to find all ascendants for
        seg_data (pandas.DataFrame): DataFrame containing segment data including 'sec_type_precise'
        sec_type_to_get (str): Section type to filter ascendants by

    Returns:
        dict: Dictionary mapping each input segment index to its set of ascendant indices of the specified type
    """
    # First get all ascendants
    all_ascendants = get_all_ascendant_seg_indices(adjacency_matrix, start_segments)
    
    # Filter ascendants by section type
    filtered_ascendants = {}
    for seg_idx in start_segments:
        ascendants = all_ascendants[seg_idx]
        ascendant_types = seg_data.loc[list(ascendants), 'sec_type_precise']
        # Print warnings for non-matching types
        # for asc in ascendants:
            # if ascendant_types[asc] != sec_type_to_get:
            #     print(f"WARNING: {ascendant_types[asc]} is not {sec_type_to_get}")
        # Filter to only keep ascendants of the specified type
        filtered_ascendants[seg_idx] = {
            asc for asc in ascendants 
            if ascendant_types[asc] == sec_type_to_get
        }
        
    return filtered_ascendants