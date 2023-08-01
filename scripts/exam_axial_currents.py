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

from Modules.plotting_utils import plot_adjacent_segments
from Modules.segment import SegmentManager
import constants
output_folder = 'output/2023-07-31_14-06-21_seeds_123_1L5PCtemplate[0]_642nseg_108nbranch_28918NCs_28918nsyn'
dt=constants.h_dt

print(constants.h_dt, constants.save_every_ms, constants.h_tstop)

def main():
  step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
  steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps
  print(step_size, steps)
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
  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt, skip=300)
  t=np.arange(0,len(sm.segments[0].v)*dt,dt) # can probably change this to read the recorded t_vec
  
  #Compute axial currents from each segment toward its adjacent segments.
  #compute axial currents between all segments
  sm.compute_axial_currents()
  
  #Find soma segments
  soma_segs = []
  for seg in sm.segments:
    if seg.type == 'soma':
      soma_segs.append(seg)
  if len(soma_segs) != 1:
    raise(ValueError("There should be only one soma segment."))
  
  
  AC_path = os.path.join(output_folder, "current_analysis")
  os.mkdir(AC_path)
  
  #Plot segments adjacent to soma
  plot_adjacent_segments(segs=soma_segs, sm=sm, title_prefix="Soma_", save_to=AC_path)
  #Plot segments adjacent to nexus
  with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
      seg_indexes = pickle.load(file)
  nexus_seg_index=seg_indexes["nexus"]
  nexus_segs=[sm.segments[nexus_seg_index]]
  plot_adjacent_segments(segs=nexus_segs, sm=sm, title_prefix="Nexus_", save_to=AC_path)
       
  
  #Plot Axial Currents
  for seg in soma_segs:
      plot_all(seg, t, save_to=AC_path, title_prefix ='Soma_')
  for seg in nexus_segs:
      plot_all(seg, t, save_to=AC_path, title_prefix = 'Nexus_')

  # Plot around APs # should update to include vlines indicating spike times.
  for i,AP_time in enumerate(np.array(sm.soma_spiketimes)):# spike time (ms) #TODO: check
      before_AP = AP_time - 100 #0.5 # ms
      after_AP = AP_time + 100 #3 # ms
      xlim = [before_AP, after_AP] # time range
      # plot around each AP
      for seg in soma_segs:
          plot_all(seg, t, xlim=xlim, index=i+1, save_to=AC_path, title_prefix="Soma_", ylim=[-1,1])
      before_AP = AP_time - 100 #0.5 # ms
      after_AP = AP_time + 100 #3 # ms
      xlim = [before_AP, after_AP] # time range
      for seg in nexus_segs:
          plot_all(seg, t, xlim=xlim, index=i+1, save_to=AC_path, title_prefix="Nexus_", ylim=[-1,1])

def plot_all(segment, t, xlim=None, ylim=None, index=None, save_to=None, title_prefix=None):
    '''
    Plots axial current from segment to adjacent segments,
    Plots Vm of segment and adjacent segments,
    Plots Currents of segment
    
    Segment: SegmentManager Segment object
    t: time vector
    xlim: limits for x axis (used to zoom in on AP)
    ylim: limits for y axis (used to zoom in on currents that may be minimized by larger magnitude currents)
    index: Used to label spike index of soma_spiketimes
    '''
    titles = [
        'Axial Current from [{}] to adjacent segments',
        'Vm from [{}] and adjacent segments',
        'Currents from [{}]'
    ]
    if index:
      for i,title in enumerate(titles):
        titles[i] = 'Spike ' + str(int(index)) + ' ' + title
    ylabels = ['nA', 'nA', 'mV', 'mV']
    data_types = ['axial_currents', 'v', ['iampa', 'icah', 'ical', 'ih', 'ina', 'inmda']] # igaba missing # can adjust this list to visualize a specific current

    for j, title in enumerate(titles):
        fig = plt.figure(figsize=(12.8,4.8))
        title = title.format(segment.name)
        ylabel = ylabels[j]
        data_type = data_types[j]

        if type(data_type) == list: # For 'Currents from [{}]'
            for current in data_type:
                data = getattr(segment, current)
                t=np.arange(0,len(data)*dt,dt)
                plt.plot(t, data, label = current)
            # plt.yscale('log') # Use symmetrical logarithmic scale # to visualize weaker currents
        elif data_type == 'v': # For 'Vm from [{}]'
            t=np.arange(0,len(segment.v)*dt,dt)
            plt.plot(t, segment.v, color = segment.color, label = segment.name)
            for i, adj_seg in enumerate(segment.adj_segs):
                plt.plot(t, adj_seg.v, label = adj_seg.name, color = adj_seg.color)
        else: # For 'Axial Current from [{}]'
            total_AC = np.zeros(len(segment.v))
            total_dend_AC = np.zeros(len(segment.v))
            total_to_soma_AC = np.zeros(len(segment.v))
            total_away_soma_AC = np.zeros(len(segment.v))
            for i, adj_seg in enumerate(segment.adj_segs):
                total_AC += segment.axial_currents[i] # all dendrites
                if adj_seg in segment.parent_segs: # if the adjacent segment is closer to soma
                  total_to_soma_AC += segment.axial_currents[i]
                else:
                  total_away_soma_AC += segment.axial_currents[i]
                if adj_seg.type == 'dend':
                  basals=True
                  total_dend_AC += segment.axial_currents[i] # basal dendrites
                #elif adj_seg.type == 'axon':
                #  pass
                elif segment.type == 'soma':
                  plt.plot(t, segment.axial_currents[i], label = adj_seg.name, color = adj_seg.color) # apical, axon
            if segment.type=='soma':
              plt.plot(t, total_dend_AC, label = 'Summed basal axial currents', color = 'red')
              plt.ylim([-2,2])
            else:
              plt.plot(t, total_to_soma_AC, label = 'Summed axial currents to segments toward soma', color = 'blue')
              plt.plot(t, total_away_soma_AC, label = 'Summed axial currents to segments away from soma', color = 'red') # can update label and color. # net current
              plt.ylim([-0.75,0.75])
            plt.plot(t, total_AC, label = 'Summed axial currents', color = 'Magenta') # can update label and color. # net current
              
            #plt.ylim([-0.75,0.75])#plt.ylim([-2,2])

        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.ylabel(ylabel)
        plt.xlabel('time (ms)')
        plt.legend()
        if title_prefix:
          plt.title(title_prefix+title)
          if save_to:
            fig.savefig(os.path.join(save_to, title_prefix+title + ".png"))
        else:
          plt.title(title)
          if save_to:
            fig.savefig(os.path.join(save_to, title + ".png"))
        plt.close()
        #plt.show()

if __name__ == "__main__":
    main()