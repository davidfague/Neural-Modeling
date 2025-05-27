import os
import pandas as pd
import numpy as np
from Modules.segments_file import generate_segments_csv
from Modules.cell_builder import CellBuilder, SkeletonCell
from Modules.constants import HayParameters
from neuron import h

def get_branch_statistics_by_section_type(sim_dir, parameters):
    """
    Generate branch statistics organized by section type.
    
    Args:
        sim_dir (str): Directory containing simulation data
        parameters (HayParameters): Simulation parameters
        
    Returns:
        dict: Dictionary containing branch statistics organized by section type
    """
    # Build the cell
    cell_builder = CellBuilder(getattr(SkeletonCell, parameters.skeleton_cell_type), parameters)
    cell, _ = cell_builder.build_cell()
    
    # Get all segments
    segments, seg_data = cell.get_segments(["all"])
    
    # Get section types
    sec_types_to_get = np.unique([sec_type for syn_properties in [parameters.exc_syn_properties, parameters.inh_syn_properties] 
                                 for sec_type in syn_properties.keys()])
    
    # Initialize statistics dictionary
    branch_statistics = {sec_type: {} for sec_type in sec_types_to_get}
    
    # Process each section type
    for sec_type in sec_types_to_get:
        try:
            # Get segments of this type
            type_segments = cell.get_segments_of_type(sec_type)
            
            # Group segments by branch
            branch_segments = {}
            for seg in type_segments:
                sec_name = seg.sec.name()
                if sec_name not in branch_segments:
                    branch_segments[sec_name] = []
                branch_segments[sec_name].append(seg)
            
            # Calculate statistics for each branch
            for branch_name, segs in branch_segments.items():
                # Calculate total length
                total_length = sum(seg.L for seg in segs)
                
                # Calculate center coordinates
                coords = np.array([seg.coords for seg in segs])
                center_coords = tuple(np.mean(coords, axis=0))
                
                # Store statistics
                branch_id = int(branch_name.split('[')[1].split(']')[0])
                branch_statistics[sec_type][branch_id] = {
                    'total_length': round(total_length, 2),
                    'center_coords': tuple(round(x, 2) for x in center_coords)
                }
                
        except ValueError:
            # Skip if section type is not implemented
            continue
    
    return branch_statistics

# Example usage:
if __name__ == "__main__":
    # Create parameters
    parameters = HayParameters(
        "test",
        all_synapses_off=True,
        exc_clustering={},
        skeleton_cell_type="L5PCtemplate"
    )
    
    # Get statistics
    sim_dir = "simulations/test"
    os.makedirs(sim_dir, exist_ok=True)
    stats = get_branch_statistics_by_section_type(sim_dir, parameters)
    
    # Print statistics
    for sec_type, branches in stats.items():
        print(f"\n{sec_type}:")
        for branch_id, data in branches.items():
            print(f"  Branch {branch_id}:")
            print(f"    Total length: {data['total_length']}")
            print(f"    Center coordinates: {data['center_coords']}") 