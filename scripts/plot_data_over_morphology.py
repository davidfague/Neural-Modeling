import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
from ipywidgets import interactive_output, HBox, VBox, Layout
from IPython.display import display
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from cell_inference.config import params
from Modules.segment import SegmentManager
from Modules.plotting_utils import get_nested_property, plot_morphology
import constants


output_folder = "output/2023-08-02_14-00-11_seeds_123_87L5PCtemplate[0]_642nseg_108nbranch_28918NCs_28918nsyn"
output_folder = "output/BenModel"
constants.save_every_ms = 200
constants.h_tstop = 2500
if 'BenModel' in output_folder:
  constants.save_every_ms = 3000
  constants.h_tstop = 3000
  transpose =True
skip = 300 # (ms)

find_average=False
animate_plot=False
interactive=False
constants.cmap_type = 'cool'#'Greys'

constants.show_electrodes = False
if constants.show_electrodes:
  elec_pos = params.ELECTRODE_POSITION
else:
  elec_pos = None

# Default position parameters
loc_param_default = [0., 0., 45., 0., 1., 0.]
#[0., 0., 25., 0., 1., 0.] # resulted in 5 uV LFP
#[0., 0., 50., 0., 1., 0.] # resulted in 10 uV LFP
#[0., 0., 80., 0., 1., 0.] # resulted in 5 uV LFP # original

# Default view
elev, azim = 10, -45#90

new_property = None#['inmda','iampa','net_exc_i'] # set to None if not using existing property

time_index = 300
#identify the property being used #check cell.seg_info[0] for dictionary of properties (some are nested)
#if new_property is None:
#  if hasattr(constants, "property_list_to_analyze"):
#    property_list_to_analyze = constants.property_list_to_analyze
#  else:
property_list_to_analyze = ['netcon_density_per_seg', 'exc'] # can update here if it is not specified in constants.py
  # property_list_to_analyze = ['netcon_density_per_seg','exc']
  # property_list_to_analyze = ['seg_elec_info','beta','passive_soma']
  
def main(property_list_to_analyze):
  step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
  steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps

  #random_state = np.random.RandomState(random_state)
  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt)
  sm.compute_axial_currents()
  if new_property is not None:
    sm.sum_currents(currents=new_property[:-1], var_name = new_property[-1])
    property_list_to_analyze = [new_property[-1]]
 
  seg_prop = np.array([get_nested_property(seg, property_list_to_analyze, time_index) for seg in sm.segments]) # get seg property
  # try using average value of all time points
  if find_average:
    seg_prop = np.array([np.mean(get_nested_property(seg, property_list_to_analyze)) for seg in sm.segments]) # get seg property

  
  label = '_'.join(property_list_to_analyze) # get label
  if find_average:
    label = 'mean_' + label
  
  # Get Robust normalized properties
  # Define your percentile thresholds
  lower, upper = np.percentile(seg_prop, [1, 95])#90]) # you can adjust these percentiles to your needs
  # Define normalization function based on these percentiles
  robust_norm = plt.Normalize(vmin=lower, vmax=upper)
  # Apply robust normalization to segment property
  normalized_seg_prop = robust_norm(seg_prop)
  #normalized_seg_prop = (seg_prop - min(seg_prop)) / (max(seg_prop) - min(seg_prop)) # less robust normalization method
  # Generate Color map
  cmap = plt.get_cmap(constants.cmap_type)
  segment_colors = cmap(normalized_seg_prop)
  #print("segment_colors:",segment_colors)
  # Create a ScalarMappable object to represent the colormap
  smap = plt.cm.ScalarMappable(cmap=cmap, norm=robust_norm)
  # smap = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(seg_prop), vmax=max(seg_prop))) # another possible method
  
  # define interactive plot # update to include time data
  def interactive_plot(x, y, z, alpha, beta, phi, elev, azim):
      global loc_param
      loc_param = (x, y, z, np.pi/180*alpha, np.cos(np.pi/180*beta), np.pi/180*phi)
#      fig, ax = plot_morphology(cell=None, electrodes=elec_pos, move_cell=loc_param, elev=-elev, azim=-azim, figsize=(12, 8),
#                                clr = clr, seg_property = label, segment_colors = segment_colors, sm=smap, 
#                                seg_coords = seg_coords, sec_nseg = sec_nseg, type_id = type_id)
      fig, ax = plot_morphology(segments=sm.segments, electrodes=elec_pos, move_cell=loc_param, elev=-elev, azim=-azim, figsize=(12, 8),
                                seg_property = label, segment_colors = segment_colors, sm=smap)
      fig.savefig(os.path.join(output_folder, label+'.png'))
      plt.show()
  
  xslider = Layout(width='500px')
  yslider = Layout(height='350px')
  w_reset = widgets.Button(description='Reset', icon='history', button_style='primary')
  w_x = widgets.FloatSlider(value=loc_param_default[0], min=-200, max=200, step=1, description='x (um)', continuous_update=False, readout_format='.0f')
  w_y = widgets.FloatSlider(value=loc_param_default[1], min=-1000, max=1000, step=1, description='y (um)', continuous_update=False, readout_format='.0f')
  w_z = widgets.FloatSlider(value=loc_param_default[2], min=20, max=400, step=1, description='z (um)', continuous_update=False, readout_format='.0f')
  w_alpha = widgets.FloatSlider(value=180/np.pi*loc_param_default[3], min=-180, max=180, step=1, description='alpha (deg)', continuous_update=False, readout_format='.0f')
  w_beta = widgets.FloatSlider(value=180/np.pi*np.arccos(loc_param_default[4]), min=0, max=180, step=1, description='beta (deg)', continuous_update=False, readout_format='.0f')
  w_phi = widgets.FloatSlider(value=180/np.pi*loc_param_default[5], min=-180, max=180, step=1, description='phi (deg)', continuous_update=False, readout_format='.0f')
  w_elev = widgets.FloatSlider(value=-elev, min=-90, max=90, step=1, description='elev (deg)', continuous_update=False, readout_format='.0f', orientation='vertical', layout=yslider)
  w_azim = widgets.FloatSlider(value=-azim, min=-180, max=180, step=1, description='azim (deg)', continuous_update=False, readout_format='.0f', layout=xslider)
  
  def reset_default(*args):
      w_x.value, w_y.value, w_z.value = loc_param_default[:3]
      w_alpha.value = 180 / np.pi * loc_param_default[3]
      w_beta.value = 180 / np.pi * np.arccos(loc_param_default[4])
      w_phi.value = 180 / np.pi * loc_param_default[5]
      w_elev.value, w_azim.value = -elev, -azim
  w_reset.on_click(reset_default)
  
  loc_param = loc_param_default
  
  def animate(i, sm, robust_norm,cmap,ax, loc_param,label):
    seg_prop = np.array([get_nested_property(seg, property_list_to_analyze, i) for seg in sm.segments])
    normalized_seg_prop = robust_norm(seg_prop)
    segment_colors = cmap(normalized_seg_prop)
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=robust_norm)
    ax.clear()  # clear the plot for the new frame
    fig, ax = plot_morphology(segments=sm.segments, electrodes=elec_pos, move_cell=loc_param, elev=-elev, azim=-azim, figsize=(12, 8),
                            seg_property = label+str(i), segment_colors = segment_colors, sm=smap)
    return ax
  if animate_plot:
      # Create initial plot
      fig, ax = plot_morphology(segments=sm.segments, electrodes=elec_pos, move_cell=loc_param, elev=-elev, azim=-azim, figsize=(12, 8),
                                seg_property = label, segment_colors = segment_colors, sm=smap)
      # Create the animation
      ani = FuncAnimation(fig, animate, frames=range(len([get_nested_property(seg, property_list_to_analyze) for seg in [sm.segments[0]]])), interval=200, fargs=(sm,robust_norm,cmap,ax, loc_param,label,))
      plt.show()
  elif interactive: 
      out = interactive_output(interactive_plot, {'x': w_x, 'y': w_y, 'z': w_z, 'alpha': w_alpha, 'beta': w_beta, 'phi': w_phi, 'elev': w_elev, 'azim': w_azim})
      ui = VBox([ w_reset, HBox([ VBox([w_x, w_y, w_z]), VBox([w_alpha, w_beta, w_phi]) ]), HBox([ VBox([out, w_azim]), w_elev]) ])
      display(ui)
  else:
      print("saving figure")
      interactive_plot(x=w_x.value, y=w_y.value, z=w_z.value, alpha=w_alpha.value, beta=w_beta.value, phi=w_phi.value, elev=w_elev.value, azim=w_azim.value)


  
if __name__ == "__main__":
    main(property_list_to_analyze)
    
    