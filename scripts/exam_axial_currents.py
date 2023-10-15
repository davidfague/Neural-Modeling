'''
Note: has been adapted to print values instead of plotting
'''

import sys
sys.path.append("../")

import pickle
import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss
from mpl_toolkits import mplot3d
import pdb #python debugger

from Modules.logger import Logger

from Modules.plotting_utils import plot_adjacent_segments
from Modules.segment import SegmentManager

import importlib

def determine_output_folder_and_cell_type(default_PT_Cell_folder, default_non_PT_Cell_folder):
    # Default values

    # Check if an argument is provided
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        return folder, "PTcell" in folder
    else:
        # If not, return defaults based on PT_Cell
        return (default_PT_Cell_folder, True) if PT_Cell else (default_non_PT_Cell_folder, False)


default_PT_Cell_folder = "output/FI_Neymotin4/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_500/"
default_non_PT_Cell_folder = "output/L5PCtemplate[0]_150min_195nseg_108nbranch_16071NCs_16071nsyn/"

# If PT_Cell is not defined earlier in the code, you can set its default value here
PT_Cell = False

output_folder, PT_Cell = determine_output_folder_and_cell_type(default_PT_Cell_folder, default_non_PT_Cell_folder)

## designate which cell to use if not sys.arv
#PT_Cell=False # true: Neymotin cell; false: Neymotin_Hay
#if PT_Cell:
#  output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/FI_Neymotin4/_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_500/"
#else:
#  output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/L5PCtemplate[0]_150min_195nseg_108nbranch_16071NCs_16071nsyn/"
#
## update in case using sys.arv
#if "PTcell" in output_folder:
#  PT_Cell = True
#else:
#  PT_Cell = False
  
#FI_Neymotin_Hay56/_seeds_130_90L5PCtemplate[0]_195nseg_108nbranch_0NCs_0nsyn_500/"
#FI_Neymotin/2023-10-12_21-10-22_seeds_130_90PTcell[0]_174nseg_102nbranch_0NCs_0nsyn_300"
#FI_Neymotin_Hay2/_seeds_130_90L5PCtemplate[0]_195nseg_108nbranch_0NCs_0nsyn_300/" 
#"output/BenModel/"

# load constants
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
#  constants.save_every_ms = 200
#  constants.h_tstop = 2500
dt=constants.h_dt

# settings
segs_to_plot = {
    'Soma': True,
    'Soma_Adj': False,
    'Axon': True,
    'Nexus': True,
    'Basal': True,
    'Tuft': True
}

how_to_plot = {
    'soma spikes': False, # index to plot
    'specific_time': False, # specific_time (ms)
    'values_at_specific_time': False,
    'seg_locations': True
}

soma_spike_settings = {
    'indices': [0,100],     # list of spike numbers (*not used currently*)
    'range': 100, # (ms) before and after
    'number': 5, # should probably change to use either number or indices # currently number
    'plot_adj_Vm': True, # whether to include adj Vm in plot
    'plot_total_AC': False # whether to include seg net Ax current.
}

specific_time_settings = {
    'time' : 500, # (ms)
    'range': 100, # (ms) before and after
    'plot_adj_Vm': True, # whether to include adj Vm in plot
    'plot_total_AC': False # whether to include seg net Ax current.
}

Hay_model = constants.build_L5_cell
current_types = constants.channel_names
if PT_Cell:
  dend_current_types = current_types
else:
  dend_current_types = constants.Base_channels + constants.Hay_channels
  if 'gNaTa_t_NaTa_t' in dend_current_types:
      dend_current_types.remove('gNaTa_t_NaTa_t')

#['ik', 'ica', 'ina', 'i_pas', 'i_hd']
#['ik_kdr','ik_kap','ik_kdmc','ina_nax','i_pas', 'ica', 'iampa','inmda','igaba']


#print(constants.h_dt, constants.save_every_ms, constants.h_tstop)

def get_segments_of_type(segments, segment_type):
    return [seg for seg in segments if segment_type in seg.seg]

def get_segment_with_specific_string(segments, substr):
    for seg in segments:
        if substr in seg.seg:
            return seg
    return None

def load_segment_indexes(output_folder):
    with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
        return pickle.load(file)

def subset_data(t, xlim):
    return np.where((t >= xlim[0]) & (t <= xlim[1]))[0]

def plot_all_segments(segments_to_plot, t, current_types, save_path, specific_time, sm):
    indices = subset_data(t, [specific_time - 100, specific_time + 100])
    for prefix, segments in segments_to_plot.items():
        for i, seg in enumerate(segments):
            plot_all(segment=seg, t=t, current_types=current_types, indices=indices, index=-1, save_to=save_path, title_prefix=prefix+str(i), ylim=[-1, 1] if prefix == "Nexus_" else None, vlines=np.array(sm.soma_spiketimes))
            plot_all(segment=seg, t=t, current_types=current_types, indices=None, index=None, save_to=save_path, title_prefix=prefix, ylim=[-1, 1] if prefix == "Nexus" else None, vlines=np.array(sm.soma_spiketimes))

def get_soma_adjacent_segments(soma):
    """
    Create a dictionary of segment types based on the segments that are adjacent to the soma
    and includes the soma itself.

    Parameters:
    - soma: The soma segment object, expected to have a property 'adj_segs' listing all its adjacent segments.

    Returns:
    A dictionary with keys being segment type names (inferred from segment names) and values being lists of segments.
    """
    segment_types = {"Soma": [soma]}  # Initialize with the soma

    for adj_seg in soma.adj_segs:
        if adj_seg.type.lower().startswith("dend"):
            seg_type = "Basal"
        else:
            seg_type = adj_seg.type.capitalize()

        # Prefix the type with "Soma_Adj_"
        soma_adj_type = "Soma_Adj_" + seg_type

        if seg_type not in segment_types:
            segment_types[seg_type] = []
        if soma_adj_type not in segment_types:
            segment_types[soma_adj_type] = []

        segment_types[seg_type].append(adj_seg)
        segment_types[soma_adj_type].append(adj_seg)

    return segment_types



    
def print_steady_state_values(segment, t, steady_state_time_index, data_types=[], title_prefix=None, return_values=False, show_individuals=False):
    '''
    Print (and optionally return) the steady state values of currents and axial currents 
    for a given segment at a specific time index.
    
    Parameters:
    - segment: SegmentManager Segment object
    - t: time vector
    - steady_state_time_index: Index at which the steady state values should be printed
    - title_prefix: Prefix for the title (typically denotes the segment type)
    - return_values: If set to True, return the currents as a dictionary instead of printing
    - show_individuals: If set to True, show individual axial currents. Otherwise, show only summed currents.
    '''

    values = {}  # Dictionary to hold the values if return_values is True

    # Current types present in the segment
    #data_types = ['v','ik_kdr','ik_kap','ik_kdmc','ina_nax','i_pas', 'ica', 'iampa','inmda','igaba']

    # Print title
    if title_prefix and not return_values:
        print(f"{title_prefix} - Steady State Values at time {t[steady_state_time_index]}ms:")
    elif not return_values:
        print(f"Steady State Values at time {t[steady_state_time_index]}ms:")

    for current in data_types:
        data = getattr(segment, current)
        if current == 'v':
            units = 'mV'
        else:
            units = 'nA'

        if return_values:
            values[current] = data[steady_state_time_index]
        else:
            print(f"{current}: {data[steady_state_time_index]} {units}")

    # If there are axial currents
    if hasattr(segment, 'axial_currents'):
        axial_current_by_type = {}  # Store summed axial currents by type
        for idx, adj_seg in enumerate(segment.adj_segs):
            axial_current_value = segment.axial_currents[idx][steady_state_time_index]

            # Sum the axial currents by type
            if adj_seg.type in axial_current_by_type:
                axial_current_by_type[adj_seg.type] += axial_current_value
            else:
                axial_current_by_type[adj_seg.type] = axial_current_value

            if show_individuals and not return_values:
                print(f"Axial current from {segment.name} to {adj_seg.name} (Type: {adj_seg.type}): {axial_current_value} nA")

        for seg_type, axial_current_sum in axial_current_by_type.items():
            if not return_values:
                print(f"Total axial current to {seg_type} type segments: {axial_current_sum} nA")

        total_AC = sum(axial_current_by_type.values())
        if not return_values:
            print(f"Total summed axial currents: {total_AC} nA")

    if not return_values:
        print("\n")  # For readability

    if return_values:
        return values
        
def plot_around_spikes(spiketimes, number_to_plot, segments_to_plot, t, current_types, save_path, sm, t_range, plot_adj_Vm, plot_total_AC):
    for i, AP_time in enumerate(np.array(spiketimes)):
        if i < number_to_plot:
            before_AP = AP_time - t_range  # ms
            after_AP = AP_time + t_range  # ms
            xlim = [before_AP, after_AP]  # time range
        
            # Subset the data for the time range
            indices = subset_data(t, xlim)
            for prefix, segments in segments_to_plot.items():
                for j, seg in enumerate(segments):
                    plot_all(segment=seg, t=t, current_types=current_types, indices=indices, index=i+1, save_to=save_path, title_prefix=prefix+str(j), ylim=[-1, 1] if prefix == "Nexus" else None, vlines=np.array(sm.soma_spiketimes), plot_adj_Vm=plot_adj_Vm, plot_total_AC=plot_total_AC)


def plot_all(segment, t, current_types=[], indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None, plot_adj_Vm=True, plot_total_AC=True):
    '''
    Plots axial current from target segment to adjacent segments, unless it the target segment is soma.
    Plots Vm of segment and adjacent segments,
    Plots Currents of segment
    
    Segment: SegmentManager Segment object
    t: time vector
    indices: indices for subsetting data
    xlim: limits for x axis (used to zoom in on AP)
    ylim: limits for y axis (used to zoom in on currents that may be minimized by larger magnitude currents)
    index: Used to label spike index of soma_spiketimes
    '''
    
    if indices is not None:
        t = t[indices]
        vlines = vlines[np.isin(np.round(vlines,1), np.round(t,1))]
        #print("t:",t)
        #print("vlines:", vlines)
        
    titles = [
        'Axial Current from [{}] to adjacent segments',
        'Vm from [{}] and adjacent segments',
        'Currents from [{}]'
    ]

    if index:
        for i, title in enumerate(titles):
            titles[i] = 'Spike ' + str(int(index)) + ' ' + title
            
    ylabels = ['nA', 'mV', 'nA']
    data_types = ['axial_currents', 'v', current_types]#['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]]

    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
    
    for j, ax in enumerate(axs):
        title = titles[j].format(segment.name)
        ylabel = ylabels[j]
        data_type = data_types[j]

        if type(data_type) == list: # membrane current plots
            for current in data_type:
                if '+' in current:
                    currents_to_sum = current.split('+')
                    max_index = np.max(indices)
                    array_length = len(getattr(segment, currents_to_sum[0]))
#                    print(np.shape(indices))
#                    print(max_index)
#                    print(array_length)
                    if max_index >= array_length:
                        print(f"Error: Trying to access index {max_index} in an array of size {array_length}")
                    indices = [i for i in indices if i < array_length]
                    data = getattr(segment, currents_to_sum[0])[indices] if indices is not None else getattr(segment, currents_to_sum[0])
                    for current_to_sum in currents_to_sum[1:]:
                        data += getattr(segment, current_to_sum)[indices] if indices is not None else getattr(segment, current_to_sum)
                else:
                    data = getattr(segment, current)[indices] if indices is not None else getattr(segment, current)
                if np.shape(t) != np.shape(data):
                  print(np.shape(t), np.shape(data))
                  print(data)
                  ax.plot(t[:-1], data, label=current)
                else:
                  ax.plot(t, data, label=current)
                #ax.set_ylim([-0.1,0.1])
        elif data_type == 'v': # Voltage plots
            v_data = segment.v[indices] if indices is not None else segment.v
            ax.plot(t, v_data, color=segment.color, label=segment.name)
            if plot_adj_Vm:
                for adj_seg in segment.adj_segs:
                    adj_v_data = adj_seg.v[indices] if indices is not None else adj_seg.v
                    
                    if adj_seg.color == segment.color:
                        ax.plot(t, adj_v_data, label=adj_seg.name, color='Magenta')
                    else:
                        ax.plot(t, adj_v_data, label=adj_seg.name, color=adj_seg.color)
        elif data_type == 'axial_currents':
            # For  axial currents 'Axial Current from [{}]'
            total_AC = np.zeros(len(segment.v))
            total_dend_AC = np.zeros(len(segment.v))
            total_to_soma_AC = np.zeros(len(segment.v))
            total_away_soma_AC = np.zeros(len(segment.v))
            for adj_seg_index, adj_seg in enumerate(segment.adj_segs): # gather axial currents
                total_AC += segment.axial_currents[adj_seg_index] # all dendrites
                if segment.type == 'soma': # plotting soma's ACs
                  if adj_seg.type == 'dend': # sum basal currents
                    total_dend_AC += segment.axial_currents[adj_seg_index] # sum AC from basal dendrites
                  else: # plot axon & apical trunk ACs
                    axial_current = segment.axial_currents[adj_seg_index][indices] if indices is not None else segment.axial_currents[adj_seg_index]
                    ax.plot(t, axial_current, label=adj_seg.name, color=adj_seg.color) # apical, axon
                else: # plotting any other segment's ACs, sum axial currents to or away soma.
                  if adj_seg in segment.parent_segs: # parent segs will be closer to soma with our model.
                    total_to_soma_AC += segment.axial_currents[adj_seg_index]
                  else:
                    total_away_soma_AC += segment.axial_currents[adj_seg_index]
                  
            if segment.type=='soma': # if we are plotting for soma segment, sum basal axial currents
              basal_axial_current = total_dend_AC[indices] if indices is not None else total_dend_AC
              ax.plot(t, basal_axial_current, label = 'Summed axial currents to basal segments', color = 'red')
              #ax.set_ylim([-0.2,0.1])
            else: #if not soma, plot axial currents to segments toward soma vs AC to segments away from soma.
              total_to_soma_AC = total_to_soma_AC[indices] if indices is not None else total_to_soma_AC
              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
              total_away_soma_AC = total_away_soma_AC[indices] if indices is not None else total_away_soma_AC
              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
              #ax.set_ylim([-0.75,0.75])
            if plot_total_AC:
              total_AC = total_AC[indices] if indices is not None else total_AC
              ax.plot(t, total_AC, label = 'Summed axial currents out of segment', color = 'Magenta')
        else:
          raise(ValueError("Cannot analyze {data_type}"))

        if vlines is not None: # indicate action potentials via dashed vertical lines
            if j==0: # only the axial currents plot
              for ap_index,vline in enumerate(vlines):
                if ap_index == 0: # only label one so that legend is not outrageous
                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', label='AP time', linestyle='dashed')
                else:
                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dashed')
        ax.axhline(0, color='grey')
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('time (ms)')
        ax.legend(loc='upper right')
        if title_prefix:
          ax.set_title(title_prefix+title)
        else:
          ax.set_title(title)
            
    plt.tight_layout()

    if save_to:
        if index is None:
            index = "wholesim_" # plotting entire sim.
        else:
            index = "AP" + "{}_".format(index)
        if title_prefix:
            fig.savefig(os.path.join(save_to, index + title_prefix + ".png"))
        else:
            fig.savefig(os.path.join(save_to, index + ".png"))
    plt.close()

def get_segments_of_type(segments, segment_type):
    return [seg for seg in segments if segment_type in seg.seg]

def get_segment_with_specific_string(segments, string):
    for seg in segments:
        if string in seg.name:  # Assuming the segment has a name attribute
            return seg
    return None



# Example usage
# print_steady_state_values(segment_object, t, 100, title_prefix="Segment 1")



def main():
  save_path = os.path.join(output_folder, "Analysis Currents")
  if os.path.exists(save_path):
    logger = Logger(output_dir = save_path, active = True)
    logger.log(f'Directory already exists: {save_path}')
  else:
    os.mkdir(save_path)
    logger = Logger(output_dir = save_path, active = True)
    logger.log(f'Creating Directory: {save_path}')
    
  step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
  steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps
  #print(steps)
  #print(type(steps))
  #print([type(step) for step in steps])

  #print(f"step_size: {step_size} |  steps: {[step for step in steps]}")
  t = []
  #for dir in os.listdir(output_folder): # list folders in directory
#  for step in steps:
#      dirname = os.path.join(output_folder, f"saved_at_step_{step}")
#      print(dirname)
#      with h5py.File(os.path.join(dirname, "t.h5")) as file:
#          t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
#  t = np.hstack(t) # (ms)
#  print(t)
#  t=np.append(t,(t[-1]+dt)) # fix for if t vec is one index short of the data # for some reason this fix changes the length of seg data too?
#  print(t)

  #random_state = np.random.RandomState(random_state)
#  try:sm = SegmentManager(output_folder=output_folder, steps = steps, dt = constants.h_dt, skip=constants.skip, transpose=transpose, channel_names=constants.channel_names)
#  except: sm = SegmentManager(output_folder=output_folder, steps = steps, dt = constants.h_dt, skip=300, transpose=transpose, channel_names=constants.channel_names)
  
  sm = SegmentManager(output_folder=output_folder, steps = steps, dt = constants.h_dt, skip=constants.skip, transpose=transpose, channel_names=constants.channel_names)
  t=np.arange(0,len(sm.segments[0].v)*dt,dt) # can probably change this to read the recorded t_vec
  #print(f"dir(sm.segments[0]: {dir(sm.segments[0])}")
  #Compute axial currents from each segment toward its adjacent segments.
  #compute axial currents between all segments
  sm.compute_axial_currents()
  
  logger.log(f"soma_spiketimes: {sm.soma_spiketimes}")
  
  logger.log(f'firing_rate: {len(sm.soma_spiketimes) / (len(sm.segments[0].v) * dt / 1000)}') # number of spikes / seconds of simulation
  
  
  # Finding segments
  soma_adj_segs = get_soma_adjacent_segments(sm.segments[0])
  
  soma_segs = get_segments_of_type(sm.segments, 'soma')
  if len(soma_segs) != 1:
      logger.log(f"Picking 1 out of {len(soma_segs)} Soma segments.")
      soma_segs = [soma_segs[3]]
  
  seg_indexes = load_segment_indexes(output_folder)
  if 'BenModel' in output_folder:
      nexus_seg_index, basal_seg_index = [], []
  else:
      nexus_seg_index, basal_seg_index, axon_seg_index, tuft_seg_index = seg_indexes["nexus"], seg_indexes["basal"], seg_indexes["axon"], seg_indexes["tuft"]
  
  nexus_segs, basal_segs, axon_segs, tuft_segs = [sm.segments[nexus_seg_index]], [sm.segments[basal_seg_index]], [sm.segments[axon_seg_index]], [sm.segments[tuft_seg_index]]
  axon_seg = get_segment_with_specific_string(sm.segments, '[0](0.5)') or get_segment_with_specific_string(sm.segments, '(0.5)')
  
  if constants.build_cell_reports_cell:
      nexus_seg = get_segment_with_specific_string(sm.segments, 'apic[24]')
      if nexus_seg:
          nexus_segs = [nexus_seg]
  
  # Combine the dictionaries
  segments_to_plot = {**soma_adj_segs}
  
  # Add the Nexus, Basal, Axon, and Tuft segments to segments_to_plot
  segments_to_plot['Nexus'] = nexus_segs
  if 'Basal' not in segments_to_plot:
      segments_to_plot['Basal'] = []
  segments_to_plot['Basal'].extend(basal_segs)
  segments_to_plot['Axon'] = axon_segs
  segments_to_plot['Tuft'] = tuft_segs
  
  # Filter out segment types based on segs_to_plot setting
  segments_to_plot = {seg_type: segments for seg_type, segments in segments_to_plot.items() 
                    if segs_to_plot.get(seg_type, False) or (seg_type.startswith('Soma_Adj_') and segs_to_plot.get('Soma_Adj', False))}
                    
  if how_to_plot['seg_locations']:
    for seg_type, segs in segment_types.items():
        plot_adjacent_segments(segs=segs, sm=sm, title_prefix=f"{seg_type}_", save_to=save_path)
                    
  if how_to_plot["values_at_specific_time"]:
      # Filter segments_to_plot to only include Soma_Adj segments
      filtered_segment_types = {k: v for k, v in segments_to_plot.items() if k.startswith('Soma_Adj')}
           
      steady_state_index = int(specific_time_settings['time'] / constants.h_dt)
      
      # Initializing the dictionary for summed dendritic currents
      summed_dend_currents = {}
      
      # Loop over the filtered segment types and call print_steady_state_values
      for title_prefix, segments in filtered_segment_types.items():
          for seg in segments:
              # Checking if it's a Basal dendrite segment
              if title_prefix == "Soma_Adj_Basal":
                  dend_currents = print_steady_state_values(seg, t, steady_state_index, data_types=current_types, return_values=True)
                  
                  for channel, current in dend_currents.items():
                      if channel not in summed_dend_currents:
                          summed_dend_currents[channel] = 0
                      summed_dend_currents[channel] += current
              else:
                  print_steady_state_values(seg, t, steady_state_index, data_types=current_types, title_prefix=title_prefix)
      
      # Print summed dendritic currents
      print("\nSummed dendritic currents:")
      for channel, current in summed_dend_currents.items():
          if channel == 'v':
              print(f"{channel}: {current:.4f} mV")
          else:
              print(f"{channel}: {current:.4f} nA")

  if how_to_plot['soma spikes']:
    print('number of spikes:',len(sm.soma_spiketimes))
    plot_around_spikes(sm.soma_spiketimes, number_to_plot=soma_spike_settings["number"], segments_to_plot=segments_to_plot, t=t, current_types=current_types, save_path=save_path, sm=sm, t_range=soma_spike_settings["range"], plot_adj_Vm=soma_spike_settings['plot_adj_Vm'], plot_total_AC=soma_spike_settings['plot_total_AC'])

  if how_to_plot["specific_time"]:
    plot_all_segments(segments_to_plot, t, current_types, save_path, specific_time=specific_time_settings["time"], sm=sm, plot_adj_Vm=specific_time_settings['plot_adj_Vm'], plot_total_AC=specific_time_settings['plot_total_AC'])


#  plot_adjacent_segments(segs=nexus_segs, sm=sm, title_prefix="Nexus_", save_to=save_path)
#  plot_adjacent_segments(segs=basal_segs, sm=sm, title_prefix="Basal_", save_to=save_path)
#  plot_adjacent_segments(segs=tuft_segs, sm=sm, title_prefix="Tuft_", save_to=save_path)


#  # taken from exam_NMDA
#  ca_inds=[71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 159, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
#  nmda_inds= [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
#  if process_ca_nmda_inds:
#    ca_inds = list(np.unique(ca_inds))
#    nmda_inds = list(np.unique(nmda_inds))
#    print(f"ca_inds: {ca_inds}")
#    print(f"nmda_inds: {nmda_inds}")
#    
#  ca_segs=[sm.segments[ca_ind] for ca_ind in ca_inds]
#  nmda_segs=[sm.segments[nmda_ind] for nmda_ind in nmda_inds]
#
#  if plot_CA_NMDA:
#    plot_adjacent_segments(segs=ca_segs, sm=sm, title_prefix="CA_", save_to=save_path) # segment with calcium spike
#    plot_adjacent_segments(segs=nmda_segs, sm=sm, title_prefix="NMDA_", save_to=save_path) # segment with NMDA spike

#  if plot_CA_NMDA:
#      for seg in ca_segs:
#          plot_all(seg, t, current_types=current_types, save_to=save_path, title_prefix = 'CA_')
#      for seg in nmda_segs:
#          plot_all(seg, t, current_types=current_types, save_to=save_path, title_prefix = 'NMDA_')


#def plot_all(segment, t, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines = None):
#    '''
#    Plots axial current from segment to adjacent segments,
#    Plots Vm of segment and adjacent segments,
#    Plots Currents of segment
#    
#    Segment: SegmentManager Segment object
#    t: time vector
#    xlim: limits for x axis (used to zoom in on AP)
#    ylim: limits for y axis (used to zoom in on currents that may be minimized by larger magnitude currents)
#    index: Used to label spike index of soma_spiketimes
#    '''
#    titles = [
#        'Axial Current from [{}] to adjacent segments',
#        'Vm from [{}] and adjacent segments',
#        'Currents from [{}]'
#    ]
#    if index:
#      for i,title in enumerate(titles):
#        titles[i] = 'Spike ' + str(int(index)) + ' ' + title
#    ylabels = ['nA', 'mV', 'nA']
#    data_types = ['axial_currents', 'v', ['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]] # can adjust this list to visualize a specific current
#
#    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
#
#    for j, ax in enumerate(axs):
#        title = titles[j].format(segment.name)
#        ylabel = ylabels[j]
#        data_type = data_types[j]
#
#        if type(data_type) == list: # For 'Currents from [{}]'
#            for current in data_type:
#                if '+' in current:
#                  currents_to_sum = current.split('+')
#                  data=getattr(segment,currents_to_sum[0])
#                  for current_to_sum in currents_to_sum[1:]:
#                    data+=getattr(segment,current_to_sum)
#                else:
#                  data = getattr(segment, current)
#                t=np.arange(0,len(data)*dt,dt)
#                ax.plot(t, data, label = current)
#            #if ylim is None:
#            #    ax.set_ylim([min(data), max(data)])
#        elif data_type == 'v': # For 'Vm from [{}]'
#            t=np.arange(0,len(segment.v)*dt,dt)
#            ax.plot(t, segment.v, color = segment.color, label = segment.name)
#            for i, adj_seg in enumerate(segment.adj_segs):
#                if adj_seg.color == segment.color:
#                    ax.plot(t, adj_seg.v, label = adj_seg.name, color = 'Magenta')
#                else:
#                    ax.plot(t, adj_seg.v, label = adj_seg.name, color = adj_seg.color)
#            if ylim is None:
#                ax.set_ylim([min(segment.v), max(segment.v)])
#        else: # For 'Axial Current from [{}]'
#            total_AC = np.zeros(len(segment.v))
#            total_dend_AC = np.zeros(len(segment.v))
#            total_to_soma_AC = np.zeros(len(segment.v))
#            total_away_soma_AC = np.zeros(len(segment.v))
#            for i, adj_seg in enumerate(segment.adj_segs):
#                total_AC += segment.axial_currents[i] # all dendrites
#                if adj_seg in segment.parent_segs: # if the adjacent segment is closer to soma
#                  total_to_soma_AC += segment.axial_currents[i]
#                else:
#                  total_away_soma_AC += segment.axial_currents[i]
#                if adj_seg.type == 'dend':
#                  basals=True
#                  total_dend_AC += segment.axial_currents[i] # basal dendrites
#                elif segment.type == 'soma':
#                  ax.plot(t, segment.axial_currents[i], label = adj_seg.name, color = adj_seg.color) # apical, axon
#            if segment.type=='soma':
#              ax.plot(t, total_dend_AC, label = 'Summed basal axial currents', color = 'red')
#              ax.set_ylim([-2,2])
#            else:
#              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
#              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
#              ax.set_ylim([-0.75,0.75])
#            ax.plot(t, total_AC, label = 'Summed axial currents', color = 'Magenta')
#
#        if vlines is not None:
#            if j==0: # only the axial currents plot
#              for ap_index,vline in enumerate(vlines):
#                if ap_index == 0: # only label one so that legend is not outrageous
#                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', label='AP time', linestyle='dashed')
#                else:
#                  ax.vlines(vline, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='dashed')
#        ax.axhline(0, color='grey')
#        if xlim:
#            ax.set_xlim(xlim)
#        ax.set_ylabel(ylabel)
#        ax.set_xlabel('time (ms)')
#        ax.legend(loc='upper right')
#        if title_prefix:
#          ax.set_title(title_prefix+title)
#        else:
#          ax.set_title(title)
#
#    plt.tight_layout()
#
#    if save_to:
#        if title_prefix:
#          fig.savefig(os.path.join(save_to, title_prefix + "AP_" + "_{}".format(index) + ".png"))
#        else:
#          fig.savefig(os.path.join(save_to, "AP_" + "_{}".format(index) + ".png"))
#          
#    plt.close()



if __name__ == "__main__":
    main()