import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.cm as cm
import inspect
import os
import seaborn as sns
import json

def plot_sta(data, edges, title, x_ticks, x_tick_labels, xlim, 
			 norm_percentiles = (1, 99), save_to = None) -> None:
	# Adjust the data
	# lower, upper = np.percentile(data, norm_percentiles)
	# robust_norm = plt.Normalize(vmin = lower, vmax = upper)
	# normed_data = robust_norm(data) * 2 - 1 # Normalize to [-1, 1]
	fig = plt.figure(figsize = (10, 5))
	plt.imshow(data, cmap = sns.color_palette("coolwarm", as_cmap=True))
	plt.title(title)
	plt.xticks(ticks = x_ticks + 0.5, labels = x_tick_labels)
	plt.xlabel('Time (ms)')
	if xlim is not None:
		plt.xlim(*xlim)
	plt.yticks(ticks = np.arange(11) - 0.5, labels = np.round(edges, 3))
	plt.ylabel("Edge Quantile")
	# https://github.com/dbheadley/InhibOnDendComp/blob/master/src/mean_dendevt.py
	plt.colorbar(label = 'Percent Change from Mean')

	if save_to:
		fig.savefig(f'{save_to}', dpi = fig.dpi)

#TODO CHECK
def move_position(translate: list, rotate: list, old_position: list = None, move_frame: bool = False) -> np.ndarray:
	"""
	Rotate and translate an object with old_position and calculate its new coordinates.
	Rotate(alpha, h, phi): first rotate alpha about the y-axis (spin),
	then rotate arccos(h) about the x-axis (elevation),
	then rotate phi about the y-axis (azimuth).
	Finally translate the object by translate(x, y, z).
	If move_frame is True, use the object as reference frame and move the
	old reference frame, calculate new coordinates of the old_position.
	"""
	translate = np.asarray(translate)
	if old_position is None:
		old_position = [0., 0., 0.]
	old_position = np.asarray(old_position)
	rot = Rotation.from_euler('yxy', [rotate[0], np.arccos(rotate[1]), rotate[2]])
	if move_frame:
		new_position = rot.inv().apply(old_position - translate)
	else:
		new_position = rot.apply(old_position) + translate
	return new_position

def plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, 
							nexus_seg_index, trunk_seg_index, loc_param, lfp, elec_pos, plot_lfp_heatmap, 
							plot_lfp_traces, xlim = None, ylim = None, figsize: tuple = None, vlim = 'auto',
							show = True, save_dir = None):
	if xlim is None:
		xlim=t[[0, -1]]
	v_soma = Vm[soma_seg_index]
	v_tfut = Vm[tuft_seg_index]
	v_nexus = Vm[nexus_seg_index]
	v_axon = Vm[axon_seg_index]
	v_basal = Vm[basal_seg_index]
	v_trunk = Vm[trunk_seg_index]

	if figsize is None:
		plt.figure(figsize=(10, 4))
	else:
		plt.figure(figsize=figsize)
	plt.plot(t, v_soma, label='Soma')
	plt.plot(t, v_tfut, label='Tuft')
	plt.plot(t, v_nexus, label='Nexus')
	plt.plot(t, v_basal, label='Basal')
	plt.plot(t, v_axon, label='Axon')
	plt.plot(t, v_trunk, label='Trunk')
	plt.ylabel('Membrane potential (mV)')
	plt.xlabel('time (ms)')
	plt.xlim(xlim)

	plt.legend()
	if save_dir is not None:
		plt.savefig(os.path.join(save_dir, "Vm.png"))

	# Extracellular potential along y-axis
	if ylim is None:
		y_window = [-1000, 2500]
	else:
		y_window = ylim
	ylim = loc_param[1] + np.array(y_window)  # set range of y coordinate
	max_idx = np.argmax(np.amax(np.abs(lfp), axis=0))  # find the electrode that records maximum magnitude
	x_dist = elec_pos[max_idx, 0]  # x coordinate of the maximum magnitude electrode
	e_idx = (elec_pos[:, 0]==x_dist) & (elec_pos[:, 1] >= ylim[0]) & (elec_pos[:, 1] <= ylim[1])  # selected electrode indices

	fontsize = 15
	labelpad = -10
	ticksize = 12
	tick_length = 5
	nbins = 5
	if figsize is None:
		plt.figure(figsize=(12, 5))
	else:
		plt.figure(figsize=figsize)
	_ = plot_lfp_heatmap(t=t, elec_d=elec_pos[e_idx, 1], lfp=lfp[:, e_idx],
											fontsize=fontsize, labelpad=labelpad, ticksize=ticksize, tick_length=tick_length,
											nbins=nbins, vlim = vlim, axes=plt.gca()) #vlim='auto';normal range seems to be ~ [-.00722,.00722]
	#plt.hlines(0,xmin=min(t),xmax=max(t),linestyles='dashed') # create a horizontal line
	plt.title('Extracellular potential heatmap')
	plt.xlim(xlim)
	
	if save_dir is not None:
		plt.savefig(os.path.join(save_dir, "ecp_heatmap.png"))

	if figsize is None:
		plt.figure(figsize=(8, 5))
	else:
		plt.figure(figsize=figsize)
	_ = plot_lfp_traces(t, lfp[:, e_idx][:,1::3], electrodes=elec_pos[e_idx][1::3],
											fontsize=fontsize, labelpad=labelpad, ticksize=ticksize, tick_length=tick_length,
											nbins=nbins, axes=plt.gca())
	plt.title('Extracellular potential timecourse')
	plt.xlim(xlim)
	if save_dir is not None:
		plt.savefig(os.path.join(save_dir, "ecp_timecourse.png"))

	if show:
		plt.show()
							
def plot_LFP_Vm_currents(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, nexus_seg_index, trunk_seg_index,
							loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, xlim=None, ylim=None, 
			 			figsize: tuple = None, vlim = 'auto', data_dict: dict = None,
						cmap: str = 'tab20'):
							
	plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, nexus_seg_index, trunk_seg_index,
							loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, xlim, ylim, figsize, vlim)
	if xlim is None:
		xlim = t[[0, -1]]
	
	# Define a dictionary to associate segment indices with their names
	segments_dict = {
		"Soma": soma_seg_index,
		"Tuft": tuft_seg_index,
		"Nexus": nexus_seg_index,
		"Axon": axon_seg_index,
		"Basal": basal_seg_index,
		"Trunk": trunk_seg_index
	}
	
	# Loop over segments
	num_currents = len(data_dict)
	colormap = cm.get_cmap(cmap)  # or any other colormap
	for segment_name, segment_index in segments_dict.items():
		# Create a new figure for each segment
		if figsize is None:
			plt.figure(figsize=(10, 4))
		else:
			plt.figure(figsize=figsize)
	
		# Loop over currents within each segment
		for i,data_type in enumerate(data_dict):
			if (str(data_type) != 'spikes') and ('i' in data_type): # if it is a current data
				color = colormap(i / num_currents)  # get color from colormap
				data_segment = data_dict[data_type][segment_index]
				plt.plot(t, data_segment, label=str(data_type), color=color)
	
		plt.ylabel('Membrane Current (mA/cm2)')
		plt.xlabel('time (ms)')
		plt.xlim(xlim)
		plt.title('Segment Currents - ' + segment_name)  # Use segment_name in the title
		plt.legend()
		plt.show()
		plt.savefig('Currents_' + segment_name)  # Use segment_name in the file name
		plt.close()  # Close the figure after saving it

def plot_edges(edges, segments, output_folder, elec_dist_var='soma_passive', title = None, filename = None, seg_type = None):
	"""
	This function creates a plot of segments, colored according to the edge group they belong to.
	
	Parameters:
	edges (array): An array of edge values.
	segments (list): A list of segments.
	output_folder (str): Path to the output folder where the plot will be saved.
	title (str): The title of the plot. If None, the title will be the name of the 'edges' variable.
	filename (str): The filename for the saved plot. If None, the filename will be the title + '_Elec_distance.png'.
	seg_type (str): The type of segment you wish to plot ex. 'apic' or 'dend'. Used to filter segments.
	"""
	
	# If title is not provided, set title as the name of 'edges' argument
	if title is None:
		callers_local_vars = inspect.currentframe().f_back.f_locals.items()
		title = [var_name for var_name, var_val in callers_local_vars if var_val is edges][0]
	
	# If filename is not provided, set filename as title + '_Elec_distance.png'
	if filename is None:
		filename = title + '_Elec_distance.png'
	
	# Array to store edge indices
	edge_indices = []

	# Adjust edges array to include 0 and 1
	adjusted_edges = np.concatenate(([0], edges, [1]))

	# Iterate over each segment
	for seg in segments:
		seg_elec_distance = eval(seg.seg_elec_distance)['beta'][elec_dist_var]
		
		# Find the edge this segment is between
		for i in range(len(adjusted_edges) - 1):
			if adjusted_edges[i] <= seg_elec_distance <= adjusted_edges[i + 1]:  # include segments exactly at edge values
				edge_indices.append(i)
				break
		else:
			# if segment doesn't fall within any range, assign it to the last group
			edge_indices.append(len(adjusted_edges) - 2)

	# Normalize the edge_indices to range 0-1
	normalized_indices = np.array(edge_indices) / (len(adjusted_edges) - 2)  

	# Create colormap
	cmap = plt.get_cmap('jet', len(adjusted_edges) - 1)

	plt.figure(figsize=(4,10))

<<<<<<< HEAD
    # Filter segments to plot by segment type
    if seg_type is not None:
      new_segments = []
      for seg in segments:
        if (seg.type == seg_type) or (seg.type == 'soma'):
          new_segments.append(seg)
    else:
	    new_segments = segments
=======
	# Filter segments to plot by segment type
	if seg_type is not None:
		new_segments = []
		for seg in segments:
			if (seg.type == seg_type) or (seg.type == 'soma'):
				new_segments.append(seg)
	else:
		new_segments = segments
>>>>>>> refs/remotes/origin/main
		

	# Plot segments colored by normalized edge index
	for i, seg in enumerate(new_segments):
		plt.plot([seg.p0_x3d, seg.p0_5_x3d, seg.p1_x3d], [seg.p0_y3d, seg.p0_5_y3d, seg.p1_y3d], color=cmap(normalized_indices[i]))

	# Invisible scatter plot for the colorbar
	sc = plt.scatter([seg.p0_x3d for seg in segments], [seg.p0_y3d for seg in segments], c=normalized_indices, s=0, cmap=cmap)

	# Draw lines and labels
	plt.vlines(110,400,500)
	plt.text(0,450,'100 um')
	plt.hlines(400,110,210)
	plt.text(110,350,'100 um')
	plt.xticks([])
	plt.yticks([])

	# Set title
	plt.title(title)

	# Normalize adjusted_edges for colorbar ticks
	normalized_ticks = np.linspace(0, 1, len(adjusted_edges))

	# Create colorbar with ticks and labels matching adjusted edges
	cbar = plt.colorbar(sc, ticks=normalized_ticks, label='Edge index')
	cbar.ax.set_yticklabels(["{:.3f}".format(val) for val in adjusted_edges])
	cbar.ax.set_ylabel('Percentage of Somatic signal', rotation=270)

	plt.box(False)
	plt.savefig(os.path.join(output_folder, filename))

def plot_spikes(sm, seg=None, seg_index=None, dendritic_spike_times=[], spike_labels=[], start_time=0, end_time=None, output_folder="", title=None):
	"""
	This function plots the membrane potential of a given segment along with the spike times for different dendritic spikes. 
	It provides an option to specify a time range for the plot.

	Args:
	sm : Segment Manager object that holds all segments.
	seg : A Segment object that contains the membrane potential data to be plotted. Either seg or seg_index must be specified.
	seg_index : Index of the segment in the sm.segments list. Either seg or seg_index must be specified.
	dendritic_spike_times : A list of lists, each containing spike times for a type of dendritic spike (e.g. NMDA, CA, NA).
	spike_labels : A list of labels for the dendritic_spike_times.
	start_time : The start of the time range for the plot, in milliseconds. Default is 0.
	end_time : The end of the time range for the plot, in milliseconds. If not specified, the plot goes till the end of the segment.
	output_folder : The folder where to save the plot. If it does not exist, it will be created. Default is the current directory.
	title: The title for the plot. If not specified, the name of the segment will be used.

	Returns:
	None
	"""
	if seg is None and seg_index is None:
		raise ValueError("Either seg or seg_index must be provided")
	elif seg is None:
		seg = sm.segments[seg_index]
	elif seg_index is None:
		seg_index = sm.segments.index(seg)

	start_index = int(start_time * 10)
	if end_time is not None:
		end_index = int(end_time * 10)
	else:
		end_index = len(seg.v)

	time_range = np.arange(start_time, start_time + (end_index-start_index)*0.1, 0.1)
	plt.plot(time_range, seg.v[start_index:end_index], color='grey')

	colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black', 'white']

	if len(spike_labels) != len(dendritic_spike_times):
		raise ValueError("dendritic_spike_times and spike_labels must have the same length")

	for i, spike in enumerate(dendritic_spike_times):
		x_values = np.array([x for x in spike[seg_index] if start_index <= x < end_index]) * 0.1
		y_values = np.array([seg.v[x] for x in spike[seg_index] if start_index <= x < end_index])
		plt.scatter(x_values, y_values, marker='*', color=colors[i % len(colors)], s=50, label=spike_labels[i])

	# If no title is provided, use seg.name
	if title is None:
		title = seg.name

	plt.ylabel('Vm (mv)')
	plt.xlabel('Time (ms)')
	plt.title(title)
	plt.legend()

	# Create output folder if it doesn't exist
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	# Saving the plot as a .png file. The file name includes the title of the plot.
	safe_title = title.replace(' ', '_')
	file_name = f"Dendritic_Spikes_{safe_title}.png"
	full_path = os.path.join(output_folder, file_name)

	plt.savefig(full_path, dpi=300)
  plt.show()

##TODO: add docstirng
def plot_morphology(segments, 
                    electrodes: np.ndarray = None, 
                    axes: list = [2, 0, 1], 
                    elev: int = 20, azim: int = 10, 
                    move_cell: list = None, 
                    figsize: tuple = None, 
                    seg_property = None, 
                    segment_colors = None, 
                    sm = None) -> tuple:                 
    """
    Plot morphology in 3D.
    
    Parameters:
    ----------
    segments: SegmentManager Segment objects
    
    electrodes: np.ndarray = None
    	Electrode positions. Default: None, not shown.
    
    axes: list = [2, 0, 1]
    	Sequence of axes to display in 3d plot axes.
    	Default: [2,0,1] show z, x, y in 3d plot x, y, z axes, so y is upward.
    
    seg_property: label for color bar
    
    segment_colors: list
    	List of colors for each segment.
     
     sm: ScalarMappable
    
    Returns:
    ----------
    fig: plt.Figure
    
    ax: plt.Axis
    """ 
    # Initialize lists to hold the coordinates and radius
    p0_x3d, p0_y3d, p0_z3d = [], [], []
    p0_5_x3d, p0_5_y3d, p0_5_z3d = [], [], []
    p1_x3d, p1_y3d, p1_z3d = [], [], []
    r = []
    
    # Loop over all segments
    for seg in segments:
        # Append coordinates to the lists
        p0_x3d.append(seg.p0_x3d)
        p0_y3d.append(seg.p0_y3d)
        p0_z3d.append(seg.p0_z3d)
        
        p0_5_x3d.append(seg.p0_5_x3d)
        p0_5_y3d.append(seg.p0_5_y3d)
        p0_5_z3d.append(seg.p0_5_z3d)
        
        p1_x3d.append(seg.p1_x3d)
        p1_y3d.append(seg.p1_y3d)
        p1_z3d.append(seg.p1_z3d)
        
        r.append(seg.seg_diam/2)  # assumes seg.r is a radius
    # Convert lists to numpy arrays
    p0_x3d, p0_y3d, p0_z3d = np.array(p0_x3d), np.array(p0_y3d), np.array(p0_z3d)
    p0_5_x3d, p0_5_y3d, p0_5_z3d = np.array(p0_5_x3d), np.array(p0_5_y3d), np.array(p0_5_z3d)
    p1_x3d, p1_y3d, p1_z3d = np.array(p1_x3d), np.array(p1_y3d), np.array(p1_z3d)
    r = np.array(r)
    
    # Construct the seg_coords dictionary
    seg_coords = {
        'p0': np.vstack((p0_x3d, p0_y3d, p0_z3d)).T,
        'pc': np.vstack((p0_5_x3d, p0_5_y3d, p0_5_z3d)).T,
        'p1': np.vstack((p1_x3d, p1_y3d, p1_z3d)).T,
        'r': r,
        'dl': np.vstack((p1_x3d, p1_y3d, p1_z3d)).T - np.vstack((p0_x3d, p0_y3d, p0_z3d)).T,
    }
    #print(seg_coords)
    if move_cell is None:
    		move_cell = [0., 0., 0., 0., 1., 0.]
    
    move_cell = np.asarray(move_cell).reshape((2, 3))
    dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
    p0 = move_position(move_cell[0], move_cell[1], seg_coords['p0'])
    pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
    p1 = move_position(move_cell[0], move_cell[1], seg_coords['p1'])
    xyz = 'xyz'
    box = np.vstack([np.full(3, np.inf), np.full(3, np.NINF)])
    if electrodes is not None:
    	box[0, axes[0:2]] = np.amin(electrodes[:, axes[0:2]], axis=0)
    	box[1, axes[0:2]] = np.amax(electrodes[:, axes[0:2]], axis=0)
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    lb_ptr = 0
    # scale linewidth to seg diameter
    min_width = 1  # minimum linewidth
    max_width = 5    # maximum linewidth
    # Rescale segment diameters to the range [min_width, max_width]
    seg_diams = [seg.seg_diam for seg in segments]
    min_diam, max_diam = min(seg_diams), max(seg_diams)
    scale_factor = (max_width - min_width) / (max_diam - min_diam)
    for seg_index, seg in enumerate(segments):
      linewidth = (seg.seg_diam - min_diam) * scale_factor + min_width
      ax.plot3D([p0[seg_index, axes[0]], pc[seg_index, axes[0]]], 
              [p0[seg_index, axes[1]], pc[seg_index, axes[1]]], 
              [p0[seg_index, axes[2]], pc[seg_index, axes[2]]], 
              color=segment_colors[seg_index], linewidth=linewidth)

      ax.plot3D([pc[seg_index, axes[0]], p1[seg_index, axes[0]]], 
              [pc[seg_index, axes[1]], p1[seg_index, axes[1]]], 
              [pc[seg_index, axes[2]], p1[seg_index, axes[2]]], 
              color=segment_colors[seg_index], linewidth=linewidth)
    box[0, :] = np.minimum(box[0, :], np.minimum(p0.min(axis=0), p1.min(axis=0)))
    box[1, :] = np.maximum(box[1, :], np.maximum(p0.max(axis=0), p1.max(axis=0)))

    cbar_ax = fig.add_axes([0.75, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    if seg_property is not None:
    	cbar.set_label(seg_property)
    ctr = np.mean(box, axis=0)
    r = np.amax(box[1, :] - box[0, :]) / 2
    box = np.vstack([ctr - r, ctr + r])
    if electrodes is not None:
    	idx = np.logical_and(np.all(electrodes >= box[0, :], axis=1), np.all(electrodes <= box[1, :], axis = 1))
    	ax.scatter(*[(electrodes[idx, j], electrodes[idx, j]) for j in axes], color = 'orange', s = 5, label = 'electrodes')
    box = box[:, axes]
    ax.auto_scale_xyz(*box.T)
    ax.view_init(elev, azim)
    ax.set_xlabel(xyz[axes[0]])
    ax.set_ylabel(xyz[axes[1]])
    ax.set_zlabel(xyz[axes[2]])
    
    return fig, ax

def get_nested_property(seg, properties, time_index = None):
    #print(f"Debug: seg = {seg}, properties = {properties}")  # Debug line
    if properties:
        property = properties[0]
        remaining_properties = properties[1:]
        if isinstance(seg, dict) and property in seg:
            prop_value = seg[property]
        elif hasattr(seg, property):
            prop_value = getattr(seg, property)
        else:
            raise ValueError(f"Property '{property}' not found in segment. Please specify a proper path in seg_info.")
        
        # Attempt to convert string representation of dictionary into dictionary
        if isinstance(prop_value, str):
            try:
                prop_value = json.loads(prop_value.replace("'", "\""))
            except json.JSONDecodeError:
                pass

        #print(f"Debug: prop_value type = {type(prop_value)}")  # Additional debug line
        
        if remaining_properties:
            if isinstance(prop_value, dict):
                return get_nested_property(prop_value, remaining_properties)
            else:
                #print(f"Debug: prop_value = {prop_value}, remaining_properties = {remaining_properties}")  # Debug line
                raise ValueError("Property list contains too many items for the depth of the target")
        else:
            if (isinstance(prop_value, np.ndarray)) & (time_index is not None):
              if (len(prop_value) > 1):
                return prop_value[time_index]
            return prop_value
    else:
        raise ValueError("Empty property list")
