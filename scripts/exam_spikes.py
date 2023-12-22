import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
from Modules.segment import SegmentManager
from Modules.plotting_utils import plot_sta, plot_edges
from Modules.constants import SimulationParameters
from Modules.logger import Logger

import pickle

from multiprocessing import Pool, cpu_count

what_to_plot = {
		"Na": False,
		"Ca": True,
		"NMDA": True,
		"Ca_NMDA": True,
    "Na_sta": False,
    "Ca_sta": False,
    "NMDA_sta": False
}

# Some cells require data to be transposed
transpose = False

# Na
threshold = 0.001 / 1000 # hay model: 0.003 / 1000
ms_within_somatic_spike = 2
na_apic_clip = (-5, 5)
na_basal_clip = (-1, 1)

# Ca
lowery, uppery = 500, 1500 # -1300, -100
ca_apic_clip = (-0.25, 1.5)

# NMDA
nmda_apic_clip = (-5, 5)
nmda_basal_clip = (-1.5, 1.5)

# NMDA Ca
nmda_ca_apic_clip = (-2, 2)

def plot_voltage_for_spike_segments(segment_manager, segment_indices, spike_start_times, spike_stop_times, spike_type, output_folder, color, max_indices=2000):
    plt.figure(figsize=(15, 10))
    
    plt.plot(segment_manager.segments[0].v[:max_indices], label=f'{segment_manager.segments[0].seg}', color='black', alpha=0.6)
    soma_spike_times = [time for time in segment_manager.soma_spiketimes if time < max_indices]
            # Mark each spike start time with a green star
    for soma_spike_time in soma_spike_times:
        plt.plot(soma_spike_time, segment_manager.segments[0].v[soma_spike_time], '*', markersize=10, color='Black')
      

    for seg_index in np.unique(segment_indices):
        seg = segment_manager.segments[seg_index]

        # Plot the entire voltage trace up to max_indices for each segment
        plt.plot(seg.v[:max_indices], label=f'{seg.seg}', color=color, alpha=0.6)

        # Select spike start and stop times for the current segment, ensuring they are within max_indices
        seg_spike_start_times = [time for time in spike_start_times[seg_index] if time < max_indices]
        seg_spike_stop_times = [time for time in spike_stop_times[seg_index] if time < max_indices]

        # Mark each spike start time with a green star
        for spike_time in seg_spike_start_times:
            plt.plot(spike_time, seg.v[spike_time], '*', markersize=10, color='green')

        # Mark each spike stop time with a red rectangle
        for spike_time in seg_spike_stop_times:
            plt.plot(spike_time, seg.v[spike_time], 's', markersize=10, color='red')

    plt.xlabel('Index')
    plt.ylabel('Voltage')
    plt.title(f'{spike_type} Spikes')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'{spike_type}_spikes.png'))
    plt.close()



def compute_mean_and_plot_sta(
			spikes: np.ndarray, 
			edges: np.ndarray, 
			title: str, 
			path: str, 
			clipping_values: tuple = (None, None), 
			clip: str = "img"):
		
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
			plot_sta(
				to_plot, 
				edges, 
				title, 
				x_ticks, 
				x_tick_labels, 
				None, 
				(clipping_values[0] * 100, clipping_values[1] * 100), 
				save_to = os.path.join(path, f"{title.replace(' ', '_')}.png"))
	elif clip == "data":
			plot_sta(
				to_plot, 
				edges, 
				title, 
				x_ticks, 
				x_tick_labels, 
				None, 
				(None, None), 
				save_to = os.path.join(path, f"{title.replace(' ', '_')}.png"))

def analyse_spikes(parameters: SimulationParameters):

	step_size = int(parameters.save_every_ms / parameters.h_dt) # Timestamps
	steps = range(step_size, int(parameters.h_tstop / parameters.h_dt) + 1, step_size) # Timestamps

	random_state = np.random.RandomState(parameters.numpy_random_state)
	sm = SegmentManager(parameters.path, steps = steps, dt = parameters.h_dt, transpose = transpose)

	plt.plot(sm.segments[0].v)
	plt.savefig(os.path.join(parameters.path, "v.png"))

	plt.plot(sm.segments[0].inmda)
	plt.savefig(os.path.join(parameters.path, "inmda.png"))

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
			na_path = os.path.join(parameters.path, "Na")
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
			try:ax.plot(np.arange(0, len(sm.segments[0].gna)*0.1, 0.1), sm.segments[0].gna)
			except:ax.plot(np.arange(0, len(sm.segments[0].gna)*0.1-0.1, 0.1), sm.segments[0].gna)
			for bound in na_lower_bounds[0]:
					ax.vlines(bound * 0.1, ymin = 0, ymax = 0.1, color = 'black', label = "Na lower bounds")
			for i, val in enumerate(np.diff(sm.segments[0].gna > threshold)): # Threshold crossings
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
			if what_to_plot["Na_sta"]:  # Check if STA plotting is enabled
						compute_mean_and_plot_sta(na_apic, edges_apic, "Na Spikes - Apical", na_path, na_apic_clip, "img")
						compute_mean_and_plot_sta(na_dend, edges_dend, "Na Spikes - Basal", na_path, na_basal_clip, "img")

	if what_to_plot["Ca"]:
			# Get bounds for Ca
			ca_start_times, ca_end_times, _, _, _, _, segments_with_ca_spikes = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = lowery, uppery = uppery, random_state = random_state)
			
			# Get edges
			edges_ca = sm.get_edges(ca_start_times)
			ca_apic = sm.get_sta(sm.soma_spiketimes, ca_start_times, edges_ca, "apic", current_type = 'ica', elec_dist_var = 'soma_passive')

			# Save Ca plots
			ca_path = os.path.join(parameters.path, "Ca")
			if not os.path.exists(ca_path):
				os.mkdir(ca_path)

			if what_to_plot["Ca_sta"]:  # Check if STA plotting is enabled
						compute_mean_and_plot_sta(ca_apic, edges_ca, 'Ca2+ Spikes - Nexus', ca_path, ca_apic_clip, "img")


	if what_to_plot["NMDA"]:
			# Get bounds for NMDA
			nmda_start_times, nmda_end_times, nmda_mag, _, _, _, segments_with_nmda_spikes = sm.get_ca_nmda_lower_bounds_durations_and_peaks(lowery = None, uppery = None, random_state = random_state)
			
			# Get edges
			edges_nmda_apic = sm.get_edges(nmda_start_times, "apic")
			nmda_apic = sm.get_sta(sm.soma_spiketimes, nmda_start_times, edges_nmda_apic, "apic", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
			
			edges_nmda_dend = sm.get_edges(nmda_start_times, "dend")
			nmda_dend = sm.get_sta(sm.soma_spiketimes, nmda_start_times, edges_nmda_dend, "dend", current_type = 'inmda', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.0001)
			
			# Save NMDA plots
			nmda_path = os.path.join(parameters.path, "NMDA")
			if not os.path.exists(nmda_path):
				os.mkdir(nmda_path)
			if what_to_plot["NMDA_sta"]:  # Check if STA plotting is enabled
						compute_mean_and_plot_sta(nmda_apic, edges_nmda_apic, 'NMDA Spikes - Apical', nmda_path, nmda_apic_clip, "img")
						compute_mean_and_plot_sta(nmda_dend, edges_nmda_dend, 'NMDA Spikes - Basal', nmda_path, nmda_basal_clip, "img")

	if (what_to_plot["Ca"]) & (what_to_plot["NMDA"]) & (what_to_plot["Ca_NMDA"]):
			# Set Ca-NMDA
			ca_spiketimes = []
			for ind, i in enumerate(ca_start_times):
					if (len(i) > 0) & ('apic[50]' in sm.segments[ind].sec):
							ca_spiketimes.extend(i.tolist())

			ca_spiketimes = np.sort(ca_spiketimes) * parameters.h_dt
			ca_spiketimes = ca_spiketimes[1:][np.diff(ca_spiketimes) > 100] # This condition is from Ben's code. It's supposed to remove duplicates.
			ca_nmda_apic = sm.get_sta(ca_spiketimes, nmda_start_times, edges_nmda_apic, "apic", current_type='ica', elec_dist_var = 'soma_passive', mag = nmda_mag, mag_th=-0.1)
			
			# Save Ca-NMDA plots
			ca_nmda_path = os.path.join(parameters.path, "Ca_NMDA")
			if not os.path.exists(ca_nmda_path):
				os.mkdir(ca_nmda_path)
			if what_to_plot["NMDA_sta"]:  # Check if STA plotting is enabled
						compute_mean_and_plot_sta(ca_nmda_apic, edges_nmda_apic, 'Ca - NMDA Spikes - Apical', ca_nmda_path, nmda_ca_apic_clip, "img")
      
	color_map = {'Na': 'red', 'Ca': 'blue', 'NMDA': 'green'}
  # Example of plotting for Na spikes
#  if what_to_plot["Na"]:
#      na_indices = [i for i, seg in enumerate(sm.segments) if seg_has_na_spike(seg)]  # Define seg_has_na_spike
#      plot_voltage_of_segments(sm, na_indices, 'Na', color_map)
    
	if what_to_plot["Ca"]:
		plot_voltage_for_spike_segments(sm, segments_with_ca_spikes, ca_start_times, ca_end_times, 'Ca', color='blue', output_folder=ca_path)

	if what_to_plot["NMDA"]:
		plot_voltage_for_spike_segments(sm, segments_with_nmda_spikes, nmda_start_times, nmda_end_times, 'NMDA', color = 'grey', output_folder=nmda_path)

  # Plotting the voltage of the 0th segment
	plt.figure()
	plt.plot(sm.segments[0].v[:2000], label='Segment 0', color='black')
	plt.xlabel('Index')
	plt.ylabel('Voltage')
	plt.title('Voltage of Segment 0')
	plt.legend()
	plt.savefig('voltage_segment_0.png')

if __name__ == "__main__":
		
	# Parse cl args
	if len(sys.argv) != 3:
		raise RuntimeError("usage: python file -f folder")
	else:
		sim_folder = sys.argv[sys.argv.index("-f") + 1]

	logger = Logger(None)
	
	# Retrieve all jobs
	jobs = []
	for job in os.listdir(sim_folder):
		if job.startswith("."): continue
		with open(os.path.join(sim_folder, job, "parameters.pickle"), "rb") as parameters:
			jobs.append(pickle.load(parameters))

	logger.log(f"Total number of jobs found: {len(jobs)}")
	logger.log(f"Total number of proccessors: {cpu_count()}")

	pool = Pool(processes = len(jobs))
	pool.map(analyse_spikes, jobs)
	pool.close()
	pool.join()
