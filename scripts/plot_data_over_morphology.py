import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import os
import ipywidgets as widgets
from ipywidgets import interactive_output, HBox, VBox, Layout
from IPython.display import display

from cell_inference.config import params
from Modules.segment import SegmentManager
from Modules.plotting_utils import get_nested_property, plot_morphology
import constants


output_folder = "output/2023-07-31_14-06-21_seeds_123_1L5PCtemplate[0]_642nseg_108nbranch_28918NCs_28918nsyn"
skip = 300 # (ms)

constants.cmap_type = 'jet'#'Greys'

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
elev, azim = 10, 90

new_property = ['inmda','iampa','net_exc_i'] # set to None if not using existing property

time_index = 300
#identify the property being used #check cell.seg_info[0] for dictionary of properties (some are nested)
if not new_property:
  if hasattr(constants, "property_list_to_analyze"):
    property_list_to_analyze = constants.property_list_to_analyze
  else:
    property_list_to_analyze = ['inmda'] # can update here if it is not specified in constants.py
  # property_list_to_analyze = ['netcon_density_per_seg','exc']
  # property_list_to_analyze = ['seg_elec_info','beta','passive_soma']

def main():
  step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
  steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps

  #random_state = np.random.RandomState(random_state)
  sm = SegmentManager(output_folder, steps = steps, dt = constants.h_dt)
  sm.compute_axial_currents()
  print(new_property[:-1],new_property[-1])
  if new_property:
    sm.sum_currents(currents=new_property[:-1], var_name = new_property[-1])
    property_list_to_analyze = [new_property[-1]]
  #print(dir(sm.segments[0]))
  print(property_list_to_analyze)
  seg_prop = np.array([get_nested_property(seg, property_list_to_analyze, time_index) for seg in sm.segments]) # get seg property

  
  label = '_'.join(property_list_to_analyze) # get label
  
  # Get Robust normalized properties
  # Define your percentile thresholds
  lower, upper = np.percentile(seg_prop, [1, 99]) # you can adjust these percentiles to your needs
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
  
  out = interactive_output(interactive_plot, {'x': w_x, 'y': w_y, 'z': w_z, 'alpha': w_alpha, 'beta': w_beta, 'phi': w_phi, 'elev': w_elev, 'azim': w_azim})
  ui = VBox([ w_reset, HBox([ VBox([w_x, w_y, w_z]), VBox([w_alpha, w_beta, w_phi]) ]), HBox([ VBox([out, w_azim]), w_elev]) ])
  
  display(ui)
  
if __name__ == "__main__":
    main()