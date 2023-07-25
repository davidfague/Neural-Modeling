import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
from Modules.segment import SegmentManager
from Modules.plotting_utils import plot_sta, plot_edges

# Output folder should store folders 'saved_at_step_xxxx'
output_folder = "output/2023-07-24_17-02-55_seeds_123_1L5PCtemplate[0]_642nseg_108nbranch_28918NCs_28918nsyn"
step_size = 2000
steps = range(2000, 10001, step_size)
dt = 0.1
what_to_plot = {
    "Na": True,
    "Ca": True,
    "NMDA": True,
    "Ca_NMDA": True
}

# Na
threshold = 0.003 / 1000
ms_within_somatic_spike = 2

# Ca
lowery, uppery = 500, 1500

def main(random_state):

    random_state = np.random.RandomState(random_state)
    sm = SegmentManager(output_folder, steps = steps, dt = dt)
    

    if what_to_plot["Na"]:
        # Get lower bounds for Na
        sm.get_na_lower_bounds_for_seg(sm.segments[0], threshold, ms_within_somatic_spike)
        na_lower_bounds, _, flattened_peak_values, _ = sm.get_na_lower_bounds_and_peaks(threshold, ms_within_somatic_spike)

        # Get edges for Na
        edges_dend = sm.get_edges(na_lower_bounds, "dend")
        edges_apic = sm.get_edges(na_lower_bounds, "apic")

        # Get STA for Na
        na_dend = sm.get_sta(sm.soma_spiketimes, na_lower_bounds, edges_dend, "dend", current_type = 'ina', elec_dist_var = 'soma_passive')
        na_apic = sm.get_sta(sm.soma_spiketimes, na_lower_bounds, edges_apic, "apic", current_type = 'ina', elec_dist_var = 'soma_passive')

        rand_spktimes = np.sort(np.random.choice(np.arange(0, len(sm.segments[0].v-1)), sm.soma_spiketimes.shape[0]))
        na_dend_rand = sm.get_sta(rand_spktimes, na_lower_bounds, edges_dend, "dend", current_type = 'ina', elec_dist_var = 'soma_passive')
        na_apic_rand = sm.get_sta(rand_spktimes, na_lower_bounds, edges_apic, "apic", current_type = 'ina', elec_dist_var = 'soma_passive')

        # Save Na plots
        na_path = os.path.join(output_folder, "Na")
        os.mkdir(na_path)

        # Hist of Na peaks
        fig, ax = plt.subplots()
        title = "Flattened Na Peaks"
        ax.hist(np.array(flattened_peak_values), bins = 100)
        ax.set_title(title)
        fig.savefig(os.path.join(na_path, title + ".png"))

        # Check for na_lower_bounds
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(sm.segments[0].v)*0.1, 0.1), sm.segments[0].gNaTa)
        for bound in na_lower_bounds[0]:
            ax.vlines(bound * 0.1, ymin = 0, ymax = 0.1, color = 'black', label = "Na lower bounds")
        for i, val in enumerate(np.diff(sm.segments[0].gNaTa > threshold)): # threshold crossings
            if val == True:
                ax.vlines(i * 0.1, ymin = 0, ymax = 0.05, color = 'red', label = "Threshold crossings")
        ax.legend()
        title = "Na Lower Bounds vs Thr Crossings"
        ax.set_title(title)
        fig.savefig(os.path.join(na_path, title + ".png"))

        # Edges
        plot_edges(edges_dend, sm.segments, na_path, elec_dist_var = 'soma_passive', filename = "na_edges_dend.png")
        plot_edges(edges_apic, sm.segments, na_path, elec_dist_var = 'soma_passive', filename = "na_edges_apic.png")

        # STA
        to_plot = np.clip((na_apic - na_apic_rand) / (np.abs(na_apic_rand) + 1e-10), -5, 5) * 100
        print(na_apic)
        print("----")
        print(na_dend_rand)
        title = 'Na Spikes - Apical'
        x_ticks = np.arange(0, 40, 5)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-20, 20, 5)]
        xlim = (5, 35)
        plot_sta(to_plot, edges_apic, title, x_ticks, x_tick_labels, xlim, save_to = os.path.join(na_path, "na_spikes_apical.png"))

        to_plot = np.clip((na_dend - na_dend_rand) / (np.abs(na_dend_rand) + 1e-15), -5, 5) * 100
        title = 'Na Spikes - Basal'
        x_ticks = np.arange(0, 40, 5)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-20, 20, 5)]
        xlim = (5, 35)
        plot_sta(to_plot, edges_apic, title, x_ticks, x_tick_labels, xlim, save_to = os.path.join(na_path, "na_spikes_basal.png"))

    if what_to_plot["Ca"]:
        # Get bounds for Ca
        ca_lower_bounds, _, _, _, _, _ = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = lowery, 
                                                                                        uppery = uppery, 
                                                                                        random_state = random_state)
        
        # Get edges
        edges_ca = sm.get_edges(ca_lower_bounds)
        ca_apic = sm.get_sta(sm.soma_spiketimes, ca_lower_bounds, edges_ca, "apic", current_type = 'ica', elec_dist_var = 'soma_passive')
        rand_spktimes = np.sort(np.random.choice(np.arange(0, len(sm.segments[0].v-1)), sm.soma_spiketimes.shape[0]))
        ca_apic_rand = sm.get_sta(rand_spktimes, ca_lower_bounds, edges_ca, "apic", current_type = 'ica', elec_dist_var = 'soma_passive')

        # Save Ca plots
        ca_path = os.path.join(output_folder, "Ca")
        os.mkdir(ca_path)

        to_plot = np.clip((ca_apic - ca_apic_rand) / (np.abs(ca_apic_rand) + 1e-15), -5, 5) * 100
        title = 'Ca2+ Spikes - Nexus'
        x_ticks = np.arange(0, 26, 4)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-100, 40, 20)]
        plot_sta(to_plot, edges_ca, title, x_ticks, x_tick_labels, [], save_to = os.path.join(ca_path, "ca_spikes_apical.png"))

    if what_to_plot["NMDA"]:
        # Get bounds for NMDA
        nmda_lower_bounds, _, nmda_mag, _, _, _ = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = None, 
                                                                                        uppery = None, 
                                                                                        random_state = random_state)
        
        # Get edges
        edges_nmda_apic = sm.get_edges(nmda_lower_bounds, "apic")
        nmda_apic = sm.get_sta(sm.soma_spiketimes, nmda_lower_bounds, edges_nmda_apic, "apic", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
        rand_spktimes_apic = np.sort(np.random.choice(np.arange(0, len(sm.segments[0].v-1)), sm.soma_spiketimes.shape[0]))
        nmda_rand_apic = sm.get_sta(rand_spktimes_apic, nmda_lower_bounds, edges_nmda_apic, "apic", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)        
        
        edges_nmda_dend = sm.get_edges(nmda_lower_bounds, "dend")
        nmda_dend = sm.get_sta(sm.soma_spiketimes, nmda_lower_bounds, edges_nmda_dend, "dend", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
        rand_spktimes_dend = np.sort(np.random.choice(np.arange(0, len(sm.segments[0].v-1)), sm.soma_spiketimes.shape[0]))
        nmda_rand_dend = sm.get_sta(rand_spktimes_dend, nmda_lower_bounds, edges_nmda_dend, "dend", current_type='inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
        
        # Save NMDA plots
        nmda_path = os.path.join(output_folder, "NMDA")
        os.mkdir(nmda_path)

        to_plot = np.clip((nmda_apic - nmda_rand_apic) / (np.abs(nmda_rand_apic) + 1e-15), -5, 5) * 100
        title = 'NMDA Spikes - Apical'
        x_ticks = np.arange(0, 26, 4)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-100, 40, 20)]
        plot_sta(to_plot, edges_nmda_apic, title, x_ticks, x_tick_labels, [], save_to = os.path.join(nmda_path, "nmda_spikes_apical.png"))
        
        to_plot = np.clip((nmda_dend - nmda_rand_dend) / (np.abs(nmda_rand_dend) + 1e-15), -5, 5) * 100
        title = 'NMDA Spikes - Basal'
        x_ticks = np.arange(0, 26, 4)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-100, 40, 20)]
        plot_sta(to_plot, edges_nmda_apic, title, x_ticks, x_tick_labels, [], save_to = os.path.join(nmda_path, "nmda_spikes_basal.png"))

    if (what_to_plot["Ca"]) & (what_to_plot["NMDA"]) & (what_to_plot["Ca_NMDA"]):
        # Set Ca-NMDA
        ca_spiketimes = []
        for ind, i in enumerate(ca_lower_bounds):
            if (len(i) > 0) & ('apic[50]' in sm.segments[ind].sec):
                ca_spiketimes.extend(i.tolist())

        ca_spiketimes = np.sort(ca_spiketimes) * dt
        ca_spiketimes = ca_spiketimes[1:][np.diff(ca_spiketimes) > 100] # This condition is from Ben's code. It's supposed to remove duplicates.
        ca_nmda_apic = sm.get_sta(ca_spiketimes, nmda_lower_bounds, edges_nmda_apic, "apic", current_type='ica', elec_dist_var = 'nexus_passive', mag = nmda_mag, mag_th=-0.1)
        ca_rand_spktimes_apic = np.sort(np.random.choice(np.arange(0, len(sm.segments[0].v-1)), sm.soma_spiketimes.shape[0]))
        ca_nmda_rand_apic = sm.get_sta(ca_rand_spktimes_apic, nmda_lower_bounds, edges_nmda_apic, "apic", current_type='ica', elec_dist_var = 'nexus_passive', mag = nmda_mag, mag_th=-0.1)

        ca_nmda_dend = sm.get_sta(ca_spiketimes, nmda_lower_bounds, edges_nmda_dend, "dend", current_type='ica', elec_dist_var = 'nexus_passive', mag = nmda_mag, mag_th=-0.1)
        ca_rand_spktimes_dend = np.sort(np.random.choice(np.arange(0, len(sm.segments[0].v-1)), sm.soma_spiketimes.shape[0]))
        ca_nmda_rand_dend = sm.get_sta(ca_rand_spktimes_dend, nmda_lower_bounds, edges_nmda_dend, "dend", current_type='ica', elec_dist_var = 'nexus_passive',mag = nmda_mag, mag_th=-0.1)        
        
        # Save Ca-NMDA plots
        ca_nmda_path = os.path.join(output_folder, "Ca_NMDA")
        os.mkdir(ca_nmda_path)

        to_plot = np.clip((ca_nmda_apic - ca_nmda_rand_apic) / (np.abs(ca_nmda_rand_apic) + 1e-15), -5, 5) * 100
        title = 'Ca - NMDA Spikes - Apical'
        x_ticks = np.arange(0, 26, 4)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-100, 40, 20)]
        plot_sta(to_plot, edges_nmda_apic, title, x_ticks, x_tick_labels, [], save_to = os.path.join(ca_nmda_path, "ca_nmda_spikes_apical.png"))

        to_plot = np.clip((ca_nmda_dend - ca_nmda_rand_dend) / (np.abs(ca_nmda_rand_dend) + 1e-15), -5, 5) * 100
        title = 'Ca - NMDA Spikes - Basal'
        x_ticks = np.arange(0, 26, 4)
        x_tick_labels = ['{}'.format(i) for i in np.arange(-100, 40, 20)]
        plot_sta(to_plot, edges_nmda_apic, title, x_ticks, x_tick_labels, [], save_to = os.path.join(ca_nmda_path, "ca_nmda_spikes_basal.png"))

if __name__ == "__main__":
    main(random_state = 123)
