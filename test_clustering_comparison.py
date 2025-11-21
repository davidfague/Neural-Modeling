#!/usr/bin/env python3
"""
Test script to compare legacy clustering with refactored clustering.
Run this to ensure refactoring maintains functionality.

Usage:
    python test_clustering_comparison.py
"""

import sys
import numpy as np
import json
from pprint import pprint

sys.path.append('Modules')
sys.path.append('terminal_branching_coords_for_clusters')

print("="*80)
print("CLUSTERING COMPARISON TEST")
print("="*80)

# ============================================================================
# PART 1: Test legacy clustering modules
# ============================================================================
print("\n[1] Testing LEGACY clustering modules...")
print("-"*80)

try:
    from terminal_branch_statistics import branch_stats
    print(f"✓ Loaded terminal_branch_statistics")
    print(f"  Section types: {list(branch_stats.keys())}")
    total_branches = sum(len(branches) for branches in branch_stats.values())
    print(f"  Total terminal branches: {total_branches}")
except ImportError as e:
    print(f"✗ Could not load terminal_branch_statistics: {e}")
    print("  Please run this from a simulation directory or update the path.")
    sys.exit(1)

# Test legacy clusters.py (simple)
print("\n--- Testing clusters.py (simple) ---")
try:
    from clusters import exc_clustering as legacy_simple_exc
    print(f"✓ Loaded clusters.py")
    
    legacy_simple_stats = {}
    for sec_type, data in legacy_simple_exc.items():
        n_fgs = len(data['functional_groups'])
        legacy_simple_stats[sec_type] = {
            'n_fgs': n_fgs,
            'sample_fg': data['functional_groups'][0] if n_fgs > 0 else None
        }
        print(f"  {sec_type}: {n_fgs} FGs")
    
except ImportError as e:
    print(f"✗ Could not load clusters.py: {e}")
    legacy_simple_stats = None

# Test legacy clusters_global_l5_fg.py (FPS)
print("\n--- Testing clusters_global_l5_fg.py (FPS) ---")
try:
    from clusters_global_l5_fg import (
        exc_clustering as legacy_fps_exc,
        inh_clustering as legacy_fps_inh
    )
    print(f"✓ Loaded clusters_global_l5_fg.py")
    
    legacy_fps_exc_stats = {}
    for sec_type, data in legacy_fps_exc.items():
        n_fgs = len(data['functional_groups'])
        legacy_fps_exc_stats[sec_type] = {
            'n_fgs': n_fgs,
            'input_sources': list(set(fg['input_source'] for fg in data['functional_groups'])),
            'sample_fg': data['functional_groups'][0] if n_fgs > 0 else None
        }
        print(f"  {sec_type}: {n_fgs} FGs")
        print(f"    Input sources: {legacy_fps_exc_stats[sec_type]['input_sources'][:3]}...")
    
    legacy_fps_inh_stats = {}
    for sec_type, data in legacy_fps_inh.items():
        n_fgs = len(data['functional_groups'])
        legacy_fps_inh_stats[sec_type] = {
            'n_fgs': n_fgs,
            'sample_fg': data['functional_groups'][0] if n_fgs > 0 else None
        }
        print(f"  {sec_type} (inh): {n_fgs} FGs")
    
except ImportError as e:
    print(f"✗ Could not load clusters_global_l5_fg.py: {e}")
    legacy_fps_exc_stats = None
    legacy_fps_inh_stats = None

# ============================================================================
# PART 2: Test refactored clustering module
# ============================================================================
print("\n[2] Testing REFACTORED clustering module...")
print("-"*80)

try:
    from clustering import (
        build_clustering,
        get_default_exc_clustering,
        get_default_inh_clustering
    )
    print(f"✓ Loaded clustering.py")
except ImportError as e:
    print(f"✗ Could not load clustering.py: {e}")
    sys.exit(1)

# Test simple mode (equivalent to clusters.py)
print("\n--- Testing terminal_branch_simple mode ---")
try:
    new_simple_exc = build_clustering(
        mode='terminal_branch_simple',
        synapse_type='exc',
        branch_stats=branch_stats,
        input_source_suffixes=('_local_L5',),
        radius_scale=5.0
    )
    print(f"✓ Built simple clustering")
    
    new_simple_stats = {}
    for sec_type, data in new_simple_exc.items():
        n_fgs = len(data['functional_groups'])
        new_simple_stats[sec_type] = {
            'n_fgs': n_fgs,
            'sample_fg': data['functional_groups'][0] if n_fgs > 0 else None
        }
        print(f"  {sec_type}: {n_fgs} FGs")
    
except Exception as e:
    print(f"✗ Error building simple clustering: {e}")
    import traceback
    traceback.print_exc()
    new_simple_stats = None

# Test FPS mode (equivalent to clusters_global_l5_fg.py)
print("\n--- Testing terminal_branch_fps mode ---")
try:
    new_fps_exc = get_default_exc_clustering(mode='terminal_branch_fps', branch_stats=branch_stats)
    print(f"✓ Built FPS excitatory clustering")
    
    new_fps_exc_stats = {}
    for sec_type, data in new_fps_exc.items():
        n_fgs = len(data['functional_groups'])
        new_fps_exc_stats[sec_type] = {
            'n_fgs': n_fgs,
            'input_sources': list(set(fg['input_source'] for fg in data['functional_groups'])),
            'sample_fg': data['functional_groups'][0] if n_fgs > 0 else None
        }
        print(f"  {sec_type}: {n_fgs} FGs")
        print(f"    Input sources: {new_fps_exc_stats[sec_type]['input_sources'][:3]}...")
    
    new_fps_inh = get_default_inh_clustering(mode='global_inh', branch_stats=branch_stats)
    print(f"✓ Built FPS inhibitory clustering")
    
    new_fps_inh_stats = {}
    for sec_type, data in new_fps_inh.items():
        n_fgs = len(data['functional_groups'])
        new_fps_inh_stats[sec_type] = {
            'n_fgs': n_fgs,
            'sample_fg': data['functional_groups'][0] if n_fgs > 0 else None
        }
        print(f"  {sec_type} (inh): {n_fgs} FGs")
    
except Exception as e:
    print(f"✗ Error building FPS clustering: {e}")
    import traceback
    traceback.print_exc()
    new_fps_exc_stats = None
    new_fps_inh_stats = None

# ============================================================================
# PART 3: Compare results
# ============================================================================
print("\n[3] COMPARISON RESULTS")
print("="*80)

def compare_clusterings(legacy, new, name):
    """Compare two clustering configurations."""
    print(f"\n--- Comparing {name} ---")
    
    if legacy is None:
        print("  ⚠ Legacy clustering not available")
        return
    if new is None:
        print("  ⚠ New clustering not available")
        return
    
    all_sec_types = set(legacy.keys()) | set(new.keys())
    
    matches = []
    mismatches = []
    
    for sec_type in sorted(all_sec_types):
        legacy_n = legacy.get(sec_type, {}).get('n_fgs', 0)
        new_n = new.get(sec_type, {}).get('n_fgs', 0)
        
        if legacy_n == new_n:
            matches.append((sec_type, legacy_n, new_n))
        else:
            mismatches.append((sec_type, legacy_n, new_n))
    
    if matches:
        print(f"  ✓ Matching section types ({len(matches)}):")
        for sec_type, legacy_n, new_n in matches:
            print(f"    {sec_type}: {legacy_n} FGs")
    
    if mismatches:
        print(f"  ✗ Mismatching section types ({len(mismatches)}):")
        for sec_type, legacy_n, new_n in mismatches:
            print(f"    {sec_type}: legacy={legacy_n}, new={new_n}")
    
    if not mismatches:
        print(f"  ✅ PERFECT MATCH - All section types have same FG counts!")
        return True
    else:
        print(f"  ⚠ DIFFERENCES FOUND - Review mismatches above")
        return False

# Compare simple mode
simple_match = compare_clusterings(legacy_simple_stats, new_simple_stats, "Simple Mode")

# Compare FPS excitatory mode
fps_exc_match = compare_clusterings(legacy_fps_exc_stats, new_fps_exc_stats, "FPS Excitatory Mode")

# Compare FPS inhibitory mode
fps_inh_match = compare_clusterings(legacy_fps_inh_stats, new_fps_inh_stats, "FPS Inhibitory Mode")

# ============================================================================
# PART 4: Detailed comparison of structure
# ============================================================================
print("\n[4] DETAILED STRUCTURE COMPARISON")
print("="*80)

def compare_fg_structure(legacy_fg, new_fg, fg_id):
    """Compare individual FG structure."""
    diffs = []
    
    # Compare centers
    legacy_center = legacy_fg.get('center')
    new_center = new_fg.get('center')
    if legacy_center and new_center:
        center_diff = np.linalg.norm(np.array(legacy_center) - np.array(new_center))
        if center_diff > 0.01:  # tolerance for floating point
            diffs.append(f"center differs by {center_diff:.4f}")
    
    # Compare radius
    legacy_radius = legacy_fg.get('radius')
    new_radius = new_fg.get('radius')
    if legacy_radius and new_radius:
        radius_diff = abs(legacy_radius - new_radius)
        if radius_diff > 0.01:
            diffs.append(f"radius differs by {radius_diff:.4f}")
    
    # Compare presynaptic cells structure
    legacy_pcs = legacy_fg.get('presynaptic_cells')
    new_pcs = new_fg.get('presynaptic_cells')
    
    if isinstance(legacy_pcs, list) and isinstance(new_pcs, dict):
        diffs.append("PC structure changed: list → dict (dynamic mode)")
    elif isinstance(legacy_pcs, list) and isinstance(new_pcs, list):
        if len(legacy_pcs) != len(new_pcs):
            diffs.append(f"PC count: {len(legacy_pcs)} → {len(new_pcs)}")
    
    return diffs

# Sample comparison for first FG in each section type
print("\nSampling first FG from each section type for detailed comparison:")

if legacy_fps_exc_stats and new_fps_exc_stats:
    for sec_type in list(legacy_fps_exc_stats.keys())[:2]:  # Just first 2 for brevity
        legacy_fg = legacy_fps_exc_stats[sec_type]['sample_fg']
        new_fg = new_fps_exc_stats[sec_type]['sample_fg']
        
        if legacy_fg and new_fg:
            diffs = compare_fg_structure(legacy_fg, new_fg, 0)
            print(f"\n  {sec_type} (first FG):")
            if diffs:
                for diff in diffs:
                    print(f"    - {diff}")
            else:
                print(f"    ✓ Structures match")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

results = []
if legacy_simple_stats:
    results.append(("Simple mode", simple_match))
if legacy_fps_exc_stats:
    results.append(("FPS excitatory", fps_exc_match))
if legacy_fps_inh_stats:
    results.append(("FPS inhibitory", fps_inh_match))

if all(match for _, match in results):
    print("✅ ALL TESTS PASSED - Refactoring maintains functionality!")
elif any(match for _, match in results):
    print("⚠ PARTIAL SUCCESS - Some modes match, review differences above")
else:
    print("✗ TESTS FAILED - Significant differences found, review above")

print("\nResults by mode:")
for mode, match in results:
    status = "✅ PASS" if match else "✗ FAIL"
    print(f"  {status} - {mode}")

print("\n" + "="*80)
