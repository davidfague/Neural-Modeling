import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
from Modules.segment import SegmentManager
from Modules.plotting_utils import plot_sta, plot_edges
from Modules.logger import Logger

# Output folder should store folders 'saved_at_step_xxxx'
output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/2023-10-10_23-06-37_seeds_130_90PTcell[0]_174nseg_102nbranch_14134NCs_14134nsyn"

import importlib
def load_constants_from_folder(output_folder):
    current_script_path = "/home/drfrbc/Neural-Modeling/scripts/"
    absolute_path = current_script_path + output_folder
    sys.path.append(absolute_path)
    
    constants_module = importlib.import_module('constants_image')
    sys.path.remove(absolute_path)
    return constants_module
constants = load_constants_from_folder(output_folder)

if 'BenModel' in output_folder:
  constants.save_every_ms = 3000
  constants.h_tstop = 3000
  transpose =True
else:
  transpose=False

what_to_plot = {
    "Na": False,
    "Ca": True,
    "NMDA": True,
    "Ca_NMDA": True
}

# Na
threshold = 0.003 / 1000 # hay model: 0.003 / 1000
ms_within_somatic_spike = 2
na_apic_clip = (-5, 5)
na_basal_clip = (-1, 1)

# Ca
lowery, uppery = -1300, -100 # cell reports cell is below y axis #Hay bounds #500, 1500
ca_apic_clip = (-0.25, 1.5)

# NMDA
nmda_apic_clip = (-5, 5)
nmda_basal_clip = (-1.5, 1.5)

# NMDA Ca
nmda_ca_apic_clip = (-2, 2)

save_path = os.path.join(output_folder, "Analysis Dendritic Events")
if os.path.exists(save_path):
  logger = Logger(output_dir = save_path, active = True)
  logger.log(f'Directory already exists: {save_path}')
else:
  os.mkdir(save_path)
  logger = Logger(output_dir = save_path, active = True)
  logger.log(f'Creating Directory: {save_path}')

def compute_mean_and_plot_sta(spikes: np.ndarray, edges: np.ndarray, title: str, path: str, 
                              clipping_values: tuple = (None, None), clip: str = "img"):
    if clipping_values[0] is None:
        clipping_values[0] = spikes.min()
    if clipping_values[1] is None:
        clipping_values[1] = spikes.max()

    spk_mean = np.mean(spikes, axis = 1, keepdims = True)
    to_plot = (spikes - spk_mean) / (np.abs(spk_mean) + 1e-10)

    if clip == "data":
        to_plot = np.clip(to_plot, clipping_values[0], clipping_values[1])
    to_plot = to_plot * 100

    x_ticks = np.arange(0, 50, 5)
    x_tick_labels = ['{}'.format(i) for i in np.arange(-50, 50, 10)]
    
    if clip == "img":
        plot_sta(to_plot, edges, title, x_ticks, x_tick_labels, None, (clipping_values[0] * 100, clipping_values[1] * 100), save_to = os.path.join(path, f"{title.replace(' ', '_')}.png"))
    elif clip == "data":
        plot_sta(to_plot, edges, title, x_ticks, x_tick_labels, None, (None, None), save_to = os.path.join(path, f"{title.replace(' ', '_')}.png"))

def main(random_state):

    step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
    steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps

    random_state = np.random.RandomState(random_state)
    sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, transpose=transpose)
    

    if what_to_plot["Na"]:
        # Get lower bounds for Na
        sm.get_na_lower_bounds_for_seg(sm.segments[0], threshold, ms_within_somatic_spike)
        na_lower_bounds, _, flattened_peak_values, _ = sm.get_na_lower_bounds_and_peaks(threshold, ms_within_somatic_spike)

        # Get edges for Na
        edges_dend = sm.get_edges(na_lower_bounds, "dend")
        edges_apic = sm.get_edges(na_lower_bounds, "apic")

        # Get STA for Na
        na_dend = sm.get_sta(sm.soma_spiketimes, na_lower_bounds, edges_dend, "dend", current_type = 'gna', elec_dist_var = 'soma_passive')
        na_apic = sm.get_sta(sm.soma_spiketimes, na_lower_bounds, edges_apic, "apic", current_type = 'gna', elec_dist_var = 'soma_passive')

        # Save Na plots
        na_path = os.path.join(save_path, "Na")
        if not os.path.exists(na_path):
          os.mkdir(na_path)

        # Hist of Na peaks
        fig, ax = plt.subplots()
        title = "Flattened Na Peaks"
        ax.hist(np.array(flattened_peak_values), bins = 100)
        ax.set_title(title)
        fig.savefig(os.path.join(na_path, title + ".png"))

        # Check for na_lower_bounds
        fig, ax = plt.subplots()
        #ax.plot(np.arange(0, len(sm.segments[0].v)*0.1, 0.1), sm.segments[0].gNaTa)
        ax.plot(np.arange(0, len(sm.segments[0].v)*0.1, 0.1), sm.segments[0].gna)
        for bound in na_lower_bounds[0]:
            ax.vlines(bound * 0.1, ymin = 0, ymax = 0.1, color = 'black', label = "Na lower bounds")
        #for i, val in enumerate(np.diff(sm.segments[0].gNaTa > threshold)): # threshold crossings
        for i, val in enumerate(np.diff(sm.segments[0].gna > threshold)): # threshold crossings
            if val == True:
                ax.vlines(i * 0.1, ymin = 0, ymax = 0.05, color = 'red', label = "Threshold crossings")
        ax.legend()
        title = "Na Lower Bounds vs Thr Crossings"
        ax.set_title(title)
        fig.savefig(os.path.join(na_path, title + ".png"))

        # Edges
        plot_edges(edges_dend, sm.segments, na_path, elec_dist_var = 'soma_passive', filename = "na_edges_dend.png", seg_type = 'dend')
        plot_edges(edges_apic, sm.segments, na_path, elec_dist_var = 'soma_passive', filename = "na_edges_apic.png", seg_type = 'apic')

        # STA
        compute_mean_and_plot_sta(na_apic, edges_apic, "Na Spikes - Apical", na_path, na_apic_clip, "img")
        compute_mean_and_plot_sta(na_dend, edges_dend, "Na Spikes - Basal", na_path, na_basal_clip, "img")

    if what_to_plot["Ca"]:
        # Get bounds for Ca
        ca_lower_bounds, _, _, _, _, _ = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = lowery, 
                                                                                        uppery = uppery, 
                                                                                        random_state = random_state)
        
        # Get edges
        edges_ca = sm.get_edges(ca_lower_bounds)
        ca_apic = sm.get_sta(sm.soma_spiketimes, ca_lower_bounds, edges_ca, "apic", current_type = 'ica', elec_dist_var = 'soma_passive')

        # Save Ca plots
        ca_path = os.path.join(save_path, "Ca")
        if not os.path.exists(ca_path):
          os.mkdir(ca_path)

        compute_mean_and_plot_sta(ca_apic, edges_ca, 'Ca2+ Spikes - Nexus', ca_path, ca_apic_clip, "img")


    if what_to_plot["NMDA"]:
        # Get bounds for NMDA
        nmda_lower_bounds, _, nmda_mag, _, _, _ = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = None, 
                                                                                        uppery = None, 
                                                                                        random_state = random_state)
        
        # Get edges
        edges_nmda_apic = sm.get_edges(nmda_lower_bounds, "apic")
        nmda_apic = sm.get_sta(sm.soma_spiketimes, nmda_lower_bounds, edges_nmda_apic, "apic", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
        
        edges_nmda_dend = sm.get_edges(nmda_lower_bounds, "dend")
        nmda_dend = sm.get_sta(sm.soma_spiketimes, nmda_lower_bounds, edges_nmda_dend, "dend", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
        
        # Save NMDA plots
        nmda_path = os.path.join(save_path, "NMDA")
        if not os.path.exists(nmda_path):
          os.mkdir(nmda_path)

        compute_mean_and_plot_sta(nmda_apic, edges_nmda_apic, 'NMDA Spikes - Apical', nmda_path, nmda_apic_clip, "img")
        compute_mean_and_plot_sta(nmda_dend, edges_nmda_dend, 'NMDA Spikes - Basal', nmda_path, nmda_basal_clip, "img")

    if (what_to_plot["Ca"]) & (what_to_plot["NMDA"]) & (what_to_plot["Ca_NMDA"]):
        # Set Ca-NMDA
        ca_spiketimes = []
        for ind, i in enumerate(ca_lower_bounds):
            if (len(i) > 0) & ('apic[50]' in sm.segments[ind].sec):
                ca_spiketimes.extend(i.tolist())

        ca_spiketimes = np.sort(ca_spiketimes) * constants.h_dt
        ca_spiketimes = ca_spiketimes[1:][np.diff(ca_spiketimes) > 100] # This condition is from Ben's code. It's supposed to remove duplicates.
        ca_nmda_apic = sm.get_sta(ca_spiketimes, nmda_lower_bounds, edges_nmda_apic, "apic", current_type='ica', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.1)
        
        # Save Ca-NMDA plots
        ca_nmda_path = os.path.join(save_path, "Ca_NMDA")
        if not os.path.exists(ca_nmda_path):
          os.mkdir(ca_nmda_path)

        compute_mean_and_plot_sta(ca_nmda_apic, edges_nmda_apic, 'Ca - NMDA Spikes - Apical', ca_nmda_path, nmda_ca_apic_clip, "img")

if __name__ == "__main__":
    main(random_state = 123)
