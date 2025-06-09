import os
from Modules.cell_builder import CellBuilder, SkeletonCell
import numpy as np
import pandas as pd
from neuron import h
import pickle
from Modules.logger import Logger

def generate_segments_csv(sim_dir, parameters=None, logger=None): #@TODO: add this to some class... Maybe not CellBuilder because the CellBuilder instance could be temporary instead? discuss with @davidfague
    """
    Generate and save segment CSV for the simulation.

    Args:
        sim_dir (str): Directory of the simulation.
        parameters (ParametersClass, optional): Simulation parameters.
            If None, will load from sim_dir/parameters.pickle.
        logger (Logger, optional): Logger instance. If None, will create one.
    """
    if parameters is None:
        with open(os.path.join(sim_dir, "parameters.pickle"), "rb") as f:
            parameters = pickle.load(f)
    if logger is None:
        logger = Logger(sim_dir) # create per‑sim logger (write info into "sims_dir/sim_dir/log.txt")
    initial_all_synapses_off_parameters = parameters.all_synapses_off
    parameters.all_synapses_off = True
    # build the cell
    logger.log(f"Building cell to generate segments.csv")
    cell_builder = CellBuilder(getattr(SkeletonCell, parameters.skeleton_cell_type), parameters, logger)
    cell, _ = cell_builder.build_cell()
    logger.log(f"Cell built successfully.")

    logger.log(f"Changing cell morphology, segmentation, etc")
    # manipulate morphology: reduction, segmentation
    #@TODO add code from cellbuilder.py: CellBuilder.build_cell  -lines around reductor code block
    ####
    ####
    logger.log(f"Finished changing cell morphology, segmentation, etc")

    logger.log("Saving adjacency matrix")
    if parameters.save_adj_matrix:
        adj_matrix = cell.compute_directed_adjacency_matrix()
        np.savetxt(os.path.join(sim_dir, "adj_matrix.txt"), adj_matrix.astype(int))
    logger.log("Finished saving adjacency matrix")

    logger.log("Getting segments data")

    # save segments csv in simulation folder - the rest of this cell
    #@TODO: Make modularized code for this and clean. standardize between here and cell_model (this is from simulation.py)
    #@TODO: clean up cell.get_segments alongside cell.get_segments_of_type
    #@TODO: add sec_type (or another name for the variable) for denoting the segment type at the 'distal_basal' level instead of 'dend' for example.
    # Classify segments by morphology, save coordinates
    segments, seg_data = cell.get_segments(["all"]) # (segments is returned here to preserve NEURON references)
    seg_sections = []
    seg_idx = []
    seg_coords = []
    seg_half_seg_RAs = []
    seg = []
    seg_Ls = []
    sec_Ls = []
    sec_Ds = []
    seg_distance = []
    psegs=[]
    
    for i,entry in enumerate(seg_data):
        # if parameters.build_stylized: #@DEPRACATED
        #     sec_name = entry.section.split(".")[-1]
        # else:
        sec_name = entry.section.split(".")[-1] # name[idx]
        #print(f"sec_name: {sec_name}")
        seg_sections.append(sec_name.split("[")[0])
        seg_idx.append(sec_name.split("[")[1].split("]")[0])
        seg_coords.append(entry.coords)
        seg_half_seg_RAs.append(entry.seg_half_seg_RA)
        seg.append(entry.seg)
        seg_Ls.append(entry.L)
        psegs.append(entry.pseg)
        sec_Ls.append(segments[i].sec.L)
        sec_Ds.append(segments[i].sec.diam)
        seg_distance.append(h.distance(segments[0], segments[i]))
        
        
    seg_sections = pd.DataFrame({ #@TODO: rename seg_sections to seg_sec_data or something
        "section": seg_sections, 
        "idx_in_section_type": seg_idx,
        "seg_half_seg_RA": seg_half_seg_RAs,
        "L": seg_Ls,
        "length": seg_Ls,
        "seg":seg,
        "pseg":psegs,
        "Section_L":sec_Ls,
        "Section_diam":sec_Ds,
        "Distance":seg_distance,
        })

    seg_coords = pd.concat(seg_coords)

    seg_data = pd.concat((seg_sections.reset_index(drop = True), seg_coords.reset_index(drop = True)), axis = 1) #@TODO: compute these together instead or make seg_sections computation more concise?
    seg_data = seg_data.reset_index(drop=True) #@TODO: check if this is needed
    seg_data['seg_id'] = seg_data.index # add a seg_id so that row i → seg_id i
    seg_data.to_csv(os.path.join(sim_dir, "segment_data1.csv"))
    logger.log("Saved segments data to segment_data.csv")

    ### new version @TODO: finish this implementation for adding the new sec_types to segments.csv
    sec_types_to_get = np.unique([sec_type for syn_properties in [parameters.exc_syn_properties, parameters.inh_syn_properties] for sec_type in syn_properties.keys()])
    print(f"getting segments of types: {sec_types_to_get} for synapses")
    # These are the types the method handles currently:
    # sec_types_to_get = [
    #     'soma',
    #     'perisomatic',
    #     'trunk',
    #     'distal_basal',
    #     'distal_apic',
    #     'nexus',
    #     'tuft',
    #     'oblique'
    # ]
    rows = []
    for stype in sec_types_to_get:
        try:
            segs = cell.get_segments_of_type(stype)
        except ValueError:
            # in case a type is empty / not implemented
            continue
        for seg in segs:
            rows.append({
                'sec_name': seg.sec.name(),  # e.g. "/cell/apic[12]"
                'seg_x':    seg.x,           # normalized position along the section
                'sec_type': stype,
                'seg_id': segments.index(seg), # index of the segment in the list returned from cell.get_segments(['all'])
            })

    df = pd.DataFrame(rows, # @TODO: check this dataframe. remove duplicate segments. include segment id from the index of the list returned from cell.get_segments(['all'])
            columns=['sec_name','seg_x','sec_type', 'seg_id'])
    # df.to_csv(os.path.join(sim_dir, "segment_data2.csv"), index=False)
    ###

    ### combining main seg_data with this precise sec_type
    # print out any seg_ids that have more than one precise type #@TODO: debugging overlapping precise sec_types that may need clearer definitions or stricter logic
    # (e.g. "perisomatic" and "trunk")
    # group to collect all sec_type per seg_id
    grouped = df.groupby('seg_id')['sec_type'].unique()
    overlaps = grouped[grouped.apply(lambda arr: len(arr) > 1)]
    if not overlaps.empty:
        print("Segments with multiple precise sec_types:")
        for sid, types in overlaps.items():
            print(f"  seg_id {sid}: {types.tolist()}")

    # 5) build a map: if there's exactly one type, keep it; otherwise None
    precise_map = grouped.apply(lambda arr: arr[0] if len(arr) == 1 else None)

    # 6) assign into your main DataFrame
    seg_data['sec_type_precise'] = seg_data['seg_id'].map(precise_map)
    seg_data.loc[seg_data['section'] == 'axon', 'sec_type_precise'] = 'axon'
    seg_data.loc[seg_data['section'] == 'soma', 'sec_type_precise'] = 'soma'

    # 7) write the integrated CSV back out (overwriting the old one)
    seg_data.to_csv(os.path.join(sim_dir, "segment_data.csv"), index=False)
    logger.log("Saved integrated segment_data.csv with sec_type_precise")

    parameters.all_synapses_off = initial_all_synapses_off_parameters # not sure if this matters. depends on if alterations to parameters in here would affect parameters outside this function.
