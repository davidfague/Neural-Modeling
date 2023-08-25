'''
Note: segment manager computs axial currents from adjacent segment to the target segment. This code flips that directional relationship by multiplying by -1 when plotting axial currents.
'''

import sys
sys.path.append("../")

from collections import defaultdict
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
output_folder = sys.argv[1] if len(sys.argv) > 1 else "output/2023-08-23_13-06-55_seeds_130_90L5PCtemplate[0]_196nseg_108nbranch_16073NCs_16073nsyn" #"output/BenModel/"

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
#  constants.save_every_ms = 200
#  constants.h_tstop = 2500
dt=constants.h_dt

#print(constants.h_dt, constants.save_every_ms, constants.h_tstop)

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
  #print(step_size, steps)
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
  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=300, transpose=transpose)
  t=np.arange(0,len(sm.segments[0].v)*dt,dt) # can probably change this to read the recorded t_vec
  
  #Compute axial currents from each segment toward its adjacent segments.
  #compute axial currents between all segments
  sm.compute_axial_currents()
  
  logger.log(f"soma_spiketimes: {sm.soma_spiketimes}")
  
  logger.log(f'firing_rate: {len(sm.soma_spiketimes) / (len(sm.segments[0].v) * dt / 1000)}') # number of spikes / seconds of simulation
  
  #Find soma segments and group tuft segments.
  apical_segs_by_distance = defaultdict(list)  # defaultdict will automatically create lists for new keys
  soma_segs = []
  for seg in sm.segments:
    if seg.type == 'soma':
      soma_segs.append(seg)
    elif seg.type == 'apic':
      apical_segs_by_distance[seg.h_distance].append(seg)
      
  if len(soma_segs) != 1:
    logger.log(f"Picking 1 out of {len(soma_segs)} Soma segments.")
    #raise(ValueError("There should be only one soma segment."))
    soma_segs=[soma_segs[3]]
  
  #Plot segments adjacent to soma
  plot_adjacent_segments(segs=soma_segs, sm=sm, title_prefix="Soma_", save_to=save_path)
  #Plot segments adjacent to nexus
  with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
      seg_indexes = pickle.load(file)
  if 'BenModel' in output_folder:
    nexus_seg_index = []
    basal_seg_index = []
  else:
    nexus_seg_index=seg_indexes["nexus"]
    basal_seg_index=seg_indexes["basal"]
    tuft_seg_index=seg_indexes["tuft"]
    logger.log(f"NEXUS SEG: {sm.segments[nexus_seg_index].seg}") # to determine matching seg
  nexus_segs=[sm.segments[nexus_seg_index]]
  basal_segs=[sm.segments[basal_seg_index]]
  tuft_segs=[sm.segments[tuft_seg_index]]
  plot_adjacent_segments(segs=nexus_segs, sm=sm, title_prefix="Nexus_", save_to=save_path)
  plot_adjacent_segments(segs=basal_segs, sm=sm, title_prefix="Basal_", save_to=save_path)
  plot_adjacent_segments(segs=tuft_segs, sm=sm, title_prefix="Tuft_", save_to=save_path)
       
  
#  #Plot Axial Currents
#  for seg in soma_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix ='Soma_')
#  for seg in nexus_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Nexus_')
#  for seg in basal_segs:
#      plot_all(seg, t, save_to=save_path, title_prefix = 'Basal_')
      
  segments_to_plot = {
      "Soma_": soma_segs,
      "Nexus_": nexus_segs,
      "Basal_": basal_segs,
      "Tuft_": tuft_segs
  }

  plot_whole_data_length=False
  if plot_whole_data_length:
    for prefix, segments in segments_to_plot.items():
        for seg in segments:
            plot_all(seg, t, save_to=save_path, title_prefix=prefix)


  def subset_data(t, xlim):
      indices = np.where((t >= xlim[0]) & (t <= xlim[1]))
      return indices[0]
      
  # Plot around APs
  for i, AP_time in enumerate(np.array(sm.soma_spiketimes)):  # spike time (ms) 
      before_AP = AP_time - 100  # ms
      after_AP = AP_time + 100  # ms
      xlim = [before_AP, after_AP]  # time range
    
      # Subset the data for the time range
      indices = subset_data(t, xlim)

      for prefix, segments in segments_to_plot.items():
          for seg in segments:
              plot_all(segment=seg, t=t, indices=indices, index=i+1, save_to=save_path, title_prefix=prefix, ylim=[-1, 1] if prefix == "Nexus_" else None, vlines=np.array(sm.soma_spiketimes))
              
def plot_axial_currents(ax, t, segment, indices):
    total_AC = np.zeros(len(segment.v))
    if segment.type=='soma':
      total_dend_AC = np.zeros(len(segment.v))
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
              ax.plot(t, basal_axial_current, label = 'Summed axial currents from basal segments to soma', color = 'red')
              ax.set_ylim([-2,2])
            else: #if not soma, plot axial currents to segments toward soma vs AC to segments away from soma.
              total_to_soma_AC = total_to_soma_AC[indices] if indices is not None else total_to_soma_AC
              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
              total_away_soma_AC = total_away_soma_AC[indices] if indices is not None else total_away_soma_AC
              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
              ax.set_ylim([-0.75,0.75])
            total_AC = total_AC[indices] if indices is not None else total_AC
            ax.plot(t, total_AC, label = 'Summed axial currents', color = 'Magenta')
    pass

def plot_voltages(ax, t, segment, indices):
    # Your code for plotting voltages here
    pass

def plot_currents(ax, t, segment, indices):
    # Your code for plotting currents here
    pass
    
def plot_all(segments, t, indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None):
    '''
    Plots several types of data for a given segment over time.
    :param segments: List of SegmentManager Segment objects
    :param t: Time vector
    :param indices: Indices for subsetting data
    :param xlim: X-axis limits
    :param ylim: Y-axis limits
    :param index: Used to label spike index of soma_spiketimes
    :param save_to: Path to save plot
    :param title_prefix: Title prefix for plot
    :param vlines: Vertical lines for plot
    '''
    if indices is not None:
        t = t[indices]
        vlines = vlines[np.isin(np.round(vlines, 1), np.round(t, 1))]

    titles = [
        'Axial Current from [{}] to adjacent segments',
        'Vm from [{}] and adjacent segments',
        'Currents from [{}]'
    ]
    if index:
        titles = [f'Spike {int(index)} {title}' for title in titles]
        
    ylabels = ['nA', 'mV', 'nA']
    
    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))

    for j, (ax, title, ylabel) in enumerate(zip(axs, titles, ylabels)):
        if j == 0:
            plot_axial_currents(ax, t, segments, indices)
        elif j == 1:
            plot_voltages(ax, t, segments, indices)
        elif j == 2:
            plot_currents(ax, t, segments, indices)
        
        title = title.format(segment.name)
        if title_prefix:
            title = title_prefix + title
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (ms)')
        
        if xlim:
            ax.set_xlim(xlim)
        
        if vlines is not None:
            for vline in vlines:
                ax.axvline(x=vline, color='k', linestyle='--')
        
        ax.legend()

    plt.tight_layout()

    if save_to:
        filename = f"AP_{index}.png" if title_prefix is None else f"{title_prefix}AP_{index}.png"
        fig.savefig(os.path.join(save_to, filename))
        
    plt.close()
              
def plot_all(segments, t, indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None):
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
    # Initialize storage for accumulating data
    all_data = {
        'axial_currents': [],
        'v': [],
        'i': []  # Add more as needed
    }
    
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
    data_types = ['axial_currents', 'v', ['iampa+inmda', 'iampa+inmda+igaba','inmda', 'iampa','igaba', "imembrane"]]
    
    

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
                          if max_index >= array_length:
                              print(f"Error: Trying to access index {max_index} in an array of size {array_length}")
                          indices = [i for i in indices if i < array_length]
                          data = getattr(segment, currents_to_sum[0])[indices] if indices is not None else getattr(segment, currents_to_sum[0])
                          for current_to_sum in currents_to_sum[1:]:
                              data += getattr(segment, current_to_sum)[indices] if indices is not None else getattr(segment, current_to_sum)
                      else:
                          data = getattr(segment, current)[indices] if indices is not None else getattr(segment, current)
                      
                      ax.plot(t, data, label=current)
        elif data_type == 'v': # Voltage plots
            v_data = segment.v[indices] if indices is not None else segment.v
            ax.plot(t, v_data, color=segment.color, label=segment.name)
            
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
              ax.plot(t, basal_axial_current, label = 'Summed axial currents from basal segments to soma', color = 'red')
              ax.set_ylim([-2,2])
            else: #if not soma, plot axial currents to segments toward soma vs AC to segments away from soma.
              total_to_soma_AC = total_to_soma_AC[indices] if indices is not None else total_to_soma_AC
              ax.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
              total_away_soma_AC = total_away_soma_AC[indices] if indices is not None else total_away_soma_AC
              ax.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red')
              ax.set_ylim([-0.75,0.75])
            total_AC = total_AC[indices] if indices is not None else total_AC
            ax.plot(t, total_AC, label = 'Summed axial currents', color = 'Magenta')
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
        if title_prefix:
            fig.savefig(os.path.join(save_to, title_prefix + "AP_" + "_{}".format(index) + ".png"))
        else:
            fig.savefig(os.path.join(save_to, "AP_" + "_{}".format(index) + ".png"))
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all(segments, t, indices=None, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None, vlines=None):
    # ... (retain your docstring)
    
    # Initialize storage for accumulating data
    all_data = {
        'axial_currents': [],
        'v': [],
        'i': []  # Add more as needed
    }
    
    for segment in segments:  # Loop through each segment
        # ... (retain your original data processing code)
        
        # Example of accumulating data
        if segment.type != 'soma':
            all_data['axial_currents'].append(total_away_soma_AC)
            all_data['v'].append(segment.v[indices] if indices is not None else segment.v)
            # Add more as needed
    
    # Calculate average
    avg_axial_currents = np.mean(all_data['axial_currents'], axis=0)
    avg_v = np.mean(all_data['v'], axis=0)
    
    # Create your subplots (This stays the same)
    fig, axs = plt.subplots(len(titles), figsize=(12.8, 4.8 * len(titles)))
    
    for j, ax in enumerate(axs):
        # ... (Your original plot logic)
        
        # Add the average to the plot
        if j == 0:  # Assuming this is for 'axial_currents'
            ax.plot(t, avg_axial_currents, label='Average', linestyle='--', color='black')
        elif j == 1:  # Assuming this is for 'v'
            ax.plot(t, avg_v, label='Average', linestyle='--', color='black')
        
        # ... (rest of your code)
        
    # Save and close the figure
    # ...



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