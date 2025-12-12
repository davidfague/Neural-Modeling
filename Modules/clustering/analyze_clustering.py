#!/usr/bin/env python3
"""Analyze synapse clustering structure and generate summary report.

Creates clustering_summary.txt showing unique functional_group and 
functional_group × presynaptic_cell combinations for each input_source.
"""

import argparse
import os
import sys
import pandas as pd


def analyze_clustering(sim_dir: str) -> None:
    """Generate clustering summary report for a simulation directory."""
    
    # Load synapses.csv
    syn_path = os.path.join(sim_dir, "synapses.csv")
    if not os.path.isfile(syn_path):
        raise FileNotFoundError(f"synapses.csv not found in {sim_dir}")
    
    print(f"[analyze_clustering] Loading synapses from {syn_path}")
    syn_df = pd.read_csv(syn_path)
    
    # Create output directory
    outdir = os.path.join(sim_dir, "clustering")
    os.makedirs(outdir, exist_ok=True)
    
    # Generate summary report
    out_path = os.path.join(outdir, "clustering_summary.txt")
    
    with open(out_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SYNAPSE CLUSTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Simulation: {os.path.basename(sim_dir)}\n")
        f.write(f"Total synapses: {len(syn_df)}\n\n")
        
        # Check if clustering columns exist
        has_fg = "functional_group" in syn_df.columns
        has_pc = "presynaptic_cell" in syn_df.columns
        
        if not has_fg:
            f.write("WARNING: 'functional_group' column not found in synapses.csv\n")
            f.write("Clustering analysis requires functional_group assignments.\n\n")
        
        if not has_pc:
            f.write("WARNING: 'presynaptic_cell' column not found in synapses.csv\n")
            f.write("Detailed clustering analysis requires presynaptic_cell assignments.\n\n")
        
        if not has_fg and not has_pc:
            f.write("No clustering data available.\n")
            print(f"[analyze_clustering] Warning: No clustering columns found")
            print(f"[analyze_clustering] Summary written to {out_path}")
            return
        
        # Get unique input sources
        input_sources = sorted(syn_df["input_source"].unique())
        
        f.write("-" * 80 + "\n")
        f.write("CLUSTERING BY INPUT SOURCE\n")
        f.write("-" * 80 + "\n\n")
        
        for src in input_sources:
            src_df = syn_df[syn_df["input_source"] == src]
            
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Input Source: {src}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Total synapses: {len(src_df)}\n\n")
            
            if has_fg:
                # Filter out negative functional groups if present
                fg_df = src_df[src_df["functional_group"] >= 0] if (src_df["functional_group"] < 0).any() else src_df
                
                unique_fgs = sorted(fg_df["functional_group"].unique())
                f.write(f"Unique functional groups: {len(unique_fgs)}\n")
                
                if len(unique_fgs) > 0:
                    f.write(f"  FG IDs: {unique_fgs}\n")
                    
                    # Show synapse counts per FG
                    fg_counts = fg_df["functional_group"].value_counts().sort_index()
                    f.write(f"\n  Synapses per functional group:\n")
                    for fg_id, count in fg_counts.items():
                        f.write(f"    FG {fg_id}: {count} synapses\n")
                else:
                    f.write(f"  (No functional groups assigned)\n")
                
                f.write("\n")
                
                if has_pc:
                    # Analyze FG × PC combinations
                    fg_pc_combos = fg_df.groupby(["functional_group", "presynaptic_cell"]).size()
                    
                    f.write(f"Functional Group × Presynaptic Cell combinations:\n")
                    f.write(f"  Total unique FG×PC pairs: {len(fg_pc_combos)}\n\n")
                    
                    if len(fg_pc_combos) > 0:
                        f.write(f"  Details by functional group:\n")
                        for fg_id in unique_fgs:
                            fg_only = fg_df[fg_df["functional_group"] == fg_id]
                            unique_pcs = sorted(fg_only["presynaptic_cell"].unique())
                            
                            f.write(f"\n    FG {fg_id}:\n")
                            f.write(f"      Unique presynaptic cells: {len(unique_pcs)}\n")
                            if len(unique_pcs) <= 20:  # Only show IDs if not too many
                                f.write(f"      PC IDs: {unique_pcs}\n")
                            
                            # Show synapse counts per PC within this FG
                            pc_counts = fg_only["presynaptic_cell"].value_counts().sort_index()
                            f.write(f"      Synapses per PC:\n")
                            for pc_id, count in pc_counts.items():
                                f.write(f"        PC {pc_id}: {count} synapses\n")
            
            elif has_pc:
                # Only PC info available
                unique_pcs = sorted(src_df["presynaptic_cell"].unique())
                f.write(f"Unique presynaptic cells: {len(unique_pcs)}\n")
                
                if len(unique_pcs) > 0 and len(unique_pcs) <= 50:
                    f.write(f"  PC IDs: {unique_pcs}\n")
                
                pc_counts = src_df["presynaptic_cell"].value_counts().sort_index()
                f.write(f"\n  Synapses per presynaptic cell:\n")
                for pc_id, count in pc_counts.items():
                    f.write(f"    PC {pc_id}: {count} synapses\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF CLUSTERING SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"[analyze_clustering] Analysis complete!")
    print(f"[analyze_clustering] Summary written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze synapse clustering structure and generate summary report"
    )
    parser.add_argument(
        "-d", "--dir",
        required=True,
        help="Path to simulation directory (contains synapses.csv)"
    )
    args = parser.parse_args()
    
    sim_dir = args.dir
    if not os.path.isdir(sim_dir):
        raise SystemExit(f"Simulation directory not found: {sim_dir}")
    
    analyze_clustering(sim_dir)


if __name__ == "__main__":
    main()
