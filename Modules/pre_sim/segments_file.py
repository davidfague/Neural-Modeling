'''
Generates sim_dir/segment_data.csv
labels each segment with a precise section type (distal_basal, perisomatic, trunk, oblique, tuft, nexus)
'''
import os
from Modules.cell_model.cell_builder import CellBuilder, SkeletonCell
import numpy as np
import pandas as pd
from neuron import h
import pickle
from Modules.logger import Logger
import warnings

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
        sec_name = entry.section.split(".")[-1] # name[idx]
        seg_sections.append(sec_name.split("[")[0])
        seg_idx.append(int(sec_name.split("[")[1].split("]")[0].split(",")[0].strip()))
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

    sec_types_to_get = np.unique([props['sec_type'] for syn_properties in [parameters.exc_syn_properties, parameters.inh_syn_properties] for input_source, props in syn_properties.items()])

    rows = []
    for stype in sec_types_to_get:
        segs = cell.get_segments_of_type(stype)
        for seg in segs:
            rows.append({
                'sec_name': seg.sec.name(),  # e.g. "/cell/apic[12]"
                'seg_x':    seg.x,           # normalized position along the section
                'sec_type': stype,
                'seg_id': segments.index(seg), # index of the segment in the list returned from cell.get_segments(['all'])
            })

    df = pd.DataFrame(rows, columns=['sec_name','seg_x','sec_type', 'seg_id']) # @TODO: check this dataframe. remove duplicate segments. include segment id from the index of the list returned from cell.get_segments(['all'])
    
    grouped = df.groupby('seg_id')['sec_type'].unique() # group to collect all sec_type per seg_id
    check_overlapping_labels(grouped, logger)

    precise_map = grouped.apply(lambda arr: arr[0] if len(arr) == 1 else None) # if there's exactly one type, keep it; otherwise None
    check_not_labeled(seg_data, precise_map, logger)
    check_nans_labels(seg_data, precise_map, logger)

    # assign into main DataFrame
    seg_data['sec_type_precise'] = seg_data['seg_id'].map(precise_map) #seg_data['sec_type_precise_depracating'] = seg_data['seg_id'].map(precise_map) # deprecating
    seg_data['sec_type_precise'] = seg_data['sec_type_precise'].fillna('unlabeled') # replace None (nan) with 'unlabeled'

    # force soma/axon override (paranoid, but harmless)
    seg_data.loc[seg_data['section'].str.contains('soma'), 'sec_type_precise'] = 'soma'
    seg_data.loc[seg_data['section'].str.contains('axon'), 'sec_type_precise'] = 'axon'

    # save
    seg_data.to_csv(os.path.join(sim_dir, "segment_data.csv"), index=False)
    logger.log(f'Saved segments with with sec_type_precise: {os.path.join(sim_dir, "segment_data.csv")}')

    parameters.all_synapses_off = initial_all_synapses_off_parameters # not sure if this matters. depends on if alterations to parameters in here would affect parameters outside this function.

def check_not_labeled(seg_data, precise_map, logger):
    '''segs not covered'''
    all_seg_ids = set(seg_data['seg_id'])
    typed_seg_ids = set(precise_map.index)
    missing = sorted(all_seg_ids - typed_seg_ids)
    if len(missing) > 0:
        logger.log(f"[segments_file] Seg IDs missing from df (no precise type): {len(missing)} -> {missing[:20]} ...")

# overlaps check
def check_overlapping_labels(grouped, logger):
    '''checks if any segments have more than one precise section type label'''
    overlaps = grouped[grouped.apply(lambda arr: len(arr) > 1)]
    if len(overlaps) > 0:
        logger.log(f"[segments_file] Overlapping seg_ids: {len(overlaps)}")
        unique_labels = np.unique(np.concatenate(overlaps.values))
        logger.log(f"[segments_file] All sec_types found in overlaps:", unique_labels)

# final NaNs after assignment
def check_nans_labels(seg_data, precise_map, logger):
    #@DEPRECATING they later recieve the label 'unlabeled' instead of getting None, which turns into nan.
    # pass
    tmp = seg_data.copy()
    tmp['sec_type_precise'] = tmp['seg_id'].map(precise_map)
    NaN_count = tmp['sec_type_precise'].isna().sum()
    if NaN_count > 0:
        logger.log(f"[segments_file] NaN count: {NaN_count}")
        logger.log(f"[segments_file] tmp.loc[tmp['sec_type_precise'].isna(), ['seg_id','section','sec_type_precise']].head(20): \n{tmp.loc[tmp['sec_type_precise'].isna(), ['seg_id','section','sec_type_precise']].head(20)}")