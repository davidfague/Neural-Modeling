"""
Modules/post_sim/post_sim_funcs.py

Post-simulation processing functions extracted from scripts/AA_post_sim_analysis.py.
These functions perform analysis and visualization after simulation completion.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Modules.analysis as analysis


def plot_soma_voltage(sim_dir: str):
    """
    Generate and save soma voltage plot.
    
    Extracted from AA_post_sim_analysis.py (lines 23-36).
    
    Creates a plot of soma voltage over time and saves it to 
    sim_dir/figs/soma_voltage.png.
    
    Args:
        sim_dir: Path to simulation directory
        
    Raises:
        Exception: If voltage data cannot be loaded or plot cannot be created
    """
    parameters = analysis.DataReader.load_parameters(sim_dir)
    v = analysis.DataReader.read_data(sim_dir, 'v')  # shape [nseg, nt]
    t = np.arange(0, parameters.h_tstop + parameters.h_dt, parameters.h_dt)
    
    os.makedirs(os.path.join(sim_dir, "figs"), exist_ok=True)
    
    plt.figure()
    plt.plot(t[:v.shape[1]], v[0, :len(t)])
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Soma voltage (seg 0)')
    plt.tight_layout()
    plt.savefig(os.path.join(sim_dir, "figs", "soma_voltage.png"), dpi=150)
    plt.close()
