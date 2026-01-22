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
import hashlib
import json
import shutil
from pathlib import Path

def compute_morphology_hash(parameters):
    """
    Compute a hash from parameters that uniquely identify the morphology.
    Simulations with the same hash will have identical segment_data.csv.
    
    Args:
        parameters: Simulation parameters object
        
    Returns:
        str: Hash string identifying the morphology
    """
    # Extract morphology-determining fields
    morph_keys = {
        'skeleton_cell_type': parameters.skeleton_cell_type,
        'morphology_name': getattr(parameters, 'morphology_name', None),
        'use_allen_cell': getattr(parameters, 'use_allen_cell', None),
        'sec_type_rules': getattr(parameters, 'sec_type_rules', None),
        'do_reduce_cell': getattr(parameters, 'do_reduce_cell', False),
        'reduction': getattr(parameters, 'reduction', None),
        # Add other morphology-affecting parameters if needed
        'reduce_apic': getattr(parameters, 'reduce_apic', False),
        'reduce_soma_gpas': getattr(parameters, 'reduce_soma_gpas', False),
    }
    
    # Serialize to JSON (sorted keys for consistency)
    morph_json = json.dumps(morph_keys, sort_keys=True, default=str)
    
    # Compute hash
    return hashlib.md5(morph_json.encode()).hexdigest()[:16]

def find_existing_segment_data(sims_dir, morph_hash, current_sim_dir):
    """
    Search for an existing segment_data.csv with the same morphology hash.
    
    Args:
        sims_dir: Parent directory containing all simulation folders
        morph_hash: Morphology hash to search for
        current_sim_dir: Current simulation directory (to skip itself)
        
    Returns:
        str or None: Path to existing segment_data.csv if found
    """
    # Check for cached mapping file
    cache_file = Path(sims_dir) / '.morphology_cache.json'
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                if morph_hash in cache:
                    source_path = cache[morph_hash]
                    if os.path.exists(source_path) and source_path != os.path.join(current_sim_dir, 'segment_data.csv'):
                        return source_path
        except:
            pass
    
    return None

def register_segment_data(sims_dir, morph_hash, segment_data_path):
    """
    Register a newly generated segment_data.csv in the cache.
    
    Args:
        sims_dir: Parent directory containing all simulation folders
        morph_hash: Morphology hash
        segment_data_path: Path to the segment_data.csv file
    """
    cache_file = Path(sims_dir) / '.morphology_cache.json'
    
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except:
            pass
    
    # Only register if not already present (first one wins)
    if morph_hash not in cache:
        cache[morph_hash] = segment_data_path
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except:
            pass  # Non-critical if caching fails

def generate_segments_csv(sim_dir, parameters=None, logger=None, force_regenerate=False, sims_dir=None): #@TODO: add this to some class... Maybe not CellBuilder because the CellBuilder instance could be temporary instead? discuss with @davidfague
    """
    Generate and save segment CSV for the simulation.

    Args:
        sim_dir (str): Directory of the simulation.
        parameters (ParametersClass, optional): Simulation parameters.
            If None, will load from sim_dir/parameters.pickle.
        logger (Logger, optional): Logger instance. If None, will create one.
        force_regenerate (bool): If True, regenerate even if file exists.
    """
    # Skip if already exists (for reruns)
    segments_csv_path = os.path.join(sim_dir, "segment_data.csv")
    if not force_regenerate and os.path.exists(segments_csv_path):
        if logger is None:
            logger = Logger(sim_dir)
        logger.log(f"segment_data.csv already exists, skipping generation")
        return
    
    if parameters is None:
        with open(os.path.join(sim_dir, "parameters.pickle"), "rb") as f:
            parameters = pickle.load(f)
    if logger is None:
        logger = Logger(sim_dir)
    
    # Try to reuse segment_data.csv from a simulation with identical morphology
    if sims_dir is not None and not force_regenerate:
        morph_hash = compute_morphology_hash(parameters)
        existing_path = find_existing_segment_data(sims_dir, morph_hash, sim_dir)
        
        if existing_path:
            try:
                shutil.copy2(existing_path, segments_csv_path)
                logger.log(f"Copied segment_data.csv from {existing_path} (morphology hash: {morph_hash})")
                return
            except Exception as e:
                logger.log(f"Failed to copy segment_data.csv: {e}. Generating from scratch.")
    
    initial_all_synapses_off_parameters = parameters.all_synapses_off
    parameters.all_synapses_off = True
    # build the cell
    logger.log(f"Building cell to generate segments.csv (morphology hash: {compute_morphology_hash(parameters) if sims_dir else 'N/A'})")
    cell_builder = CellBuilder(getattr(SkeletonCell, parameters.skeleton_cell_type), parameters, logger)
    cell, _ = cell_builder.build_cell()
    logger.log(f"Cell built successfully.")

    logger.log(f"Changing cell morphology, segmentation, etc")
    # manipulate morphology: reduction, segmentation
    #@TODO add code from cellbuilder.py: CellBuilder.build_cell  -lines around reductor code block
    logger.log(f"Finished changing cell morphology, segmentation, etc")

    # Skip adjacency matrix for speed (re-enable if needed)
    # logger.log("Saving adjacency matrix")
    # if parameters.save_adj_matrix:
    #     adj_matrix = cell.compute_directed_adjacency_matrix()
    #     np.savetxt(os.path.join(sim_dir, "adj_matrix.txt"), adj_matrix.astype(int))
    # logger.log("Finished saving adjacency matrix")

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
    
    # Optimize with list comprehensions and avoid repeated string operations
    for i, entry in enumerate(seg_data):
        sec_name = entry.section.split(".")[-1]  # name[idx]
        bracket_split = sec_name.split("[")
        seg_sections.append(bracket_split[0])
        seg_idx.append(int(bracket_split[1].split("]")[0].split(",")[0].strip()))
        seg_coords.append(entry.coords)
        seg_half_seg_RAs.append(entry.seg_half_seg_RA)
        seg.append(entry.seg)
        seg_Ls.append(entry.L)
        psegs.append(entry.pseg)
        sec = segments[i].sec
        sec_Ls.append(sec.L)
        sec_Ds.append(sec.diam)
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
    
    # Register in cache for future simulations to reuse
    if sims_dir is not None:
        morph_hash = compute_morphology_hash(parameters)
        register_segment_data(sims_dir, morph_hash, segments_csv_path)

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