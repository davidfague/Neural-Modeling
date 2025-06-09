import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import warnings
import os

# generic function for plotting variable over morphology
def plot(seg_data, data_to_plot, ax, elevation=20, azimuth=-100, radius_scale=1.0, title='', clim_max=30, return_cbar = False, clims = None):
    """
    Plots a 3D morphology of segments colored based on a given variable.

    Parameters:
        seg_data (DataFrame): Data containing segment positions and radii.
        seg_id_column (str): The column name for segment IDs.
        data_to_plot is either assumed to be a dict mapping seg_id to values or a list of values. (dict): Dictionary containing variable values for each segment.
        ax (Axes3D): The 3D axes to plot on.
        elevation (int): Elevation angle for the 3D plot.
        azimuth (int): Azimuth angle for the 3D plot.
        radius_scale (float): Scaling factor for segment radius.
        title (str): Title for the plot.
        clim_max (float): Maximum limit for color normalization.
    """
    custom_vmin = 0
    custom_vmax = clim_max
    if clims:
        custom_vmin = clims[0]
        custom_vmax = clims[1]
    norm = plt.Normalize(vmin=custom_vmin, vmax=custom_vmax)

    # If data_to_plot is a list, convert it to a dictionary
    data_to_plot = data_to_plot if isinstance(data_to_plot, dict) else {i: data_to_plot[i] for i in range(len(data_to_plot))}

    # throw error if data_to_plot does not correspond to seg_data
    if not all([seg_id in data_to_plot for seg_id in seg_data.index]):
        raise ValueError('data_to_plot does not contain values for all segments')
                         
    for seg_id, seg in seg_data.iterrows():
        if seg_id != 0:
            x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
            y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
            z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]

            radius = seg['r'] * radius_scale
            value = data_to_plot.get(seg_id, 0)
            color = plt.cm.viridis(norm(value))

            ax.plot(x_points, z_points, y_points, linewidth=radius, color=color)
    ax.set_title(title)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])  # Required for the colorbar to work properly
    fig = ax.figure
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    if return_cbar:
        return cbar
    # cbar.set_label('Your Variable Label')

def plot_special_segments(seg_data, special_indices, special_colors, title_suffix=""): # used in notebooks/plot_reduction_morph and notebooks/build_load_synapses TODO: combine with plot_segments.
    if hasattr(seg_data, 'Coord X'):
        x_coord_name = 'Coord X'
        y_coord_name = 'Coord Y'
    elif hasattr(seg_data, 'pc_0'): # pc for center, 0 for x
        x_coord_name = 'pc_0'
        y_coord_name = 'pc_1'
    else:
        NotImplementedError('seg_data does not have a valid x_coord_name')

    if hasattr(seg_data, 'segmentID'):
        seg_id_attr_name = 'segmentID'
    elif hasattr(seg_data, 'Unnamed: 0'):
        seg_id_attr_name = 'Unnamed: 0'

    # Calculate the axis limits
    all_coords_x = seg_data[x_coord_name].tolist()
    all_coords_y = seg_data[y_coord_name].tolist()
    x_min, x_max = min(all_coords_x), max(all_coords_x)
    y_min, y_max = min(all_coords_y), max(all_coords_y)

    plt.figure()
    plt.scatter(seg_data[x_coord_name], seg_data[y_coord_name], s=0.1)
    for j, ind in enumerate(special_indices):
        plt.plot(seg_data.loc[getattr(seg_data, seg_id_attr_name).isin([ind]), x_coord_name], 
                    seg_data.loc[getattr(seg_data, seg_id_attr_name).isin([ind]), y_coord_name], special_colors[j])
    
    plt.title(f"Segments {title_suffix}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

def plot_segments(seg_data, special_indices, special_colors, title_suffix="", save_file=None, show=False, ax=None, elevation=0, azimuth=-100, radius_scale=1.0): # from notebooks/plot_voltages.ipynb TODO: combine with plot_special_segments.
    # Calculate the axis limits
    if hasattr(seg_data, 'Coord X'):
        x_coord_name = 'Coord X'
        y_coord_name = 'Coord Y'
    elif hasattr(seg_data, 'pc_0'):
        x_coord_name = 'pc_0'
        y_coord_name = 'pc_1'
    else:
        NotImplementedError('seg_data does not have a valid x_coord_name')

    all_coords_x = seg_data[x_coord_name].tolist()
    all_coords_y = seg_data[y_coord_name].tolist()
    x_min, x_max = min(all_coords_x), max(all_coords_x)
    y_min, y_max = min(all_coords_y), max(all_coords_y)

    for i, segs in enumerate([seg_data]):
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        ax.scatter(segs[x_coord_name], segs[y_coord_name], s=0.1)
        for j, ind in enumerate(special_indices):
            ax.plot(segs.loc[segs.segmentID.isin([ind]), x_coord_name], 
                   segs.loc[segs.segmentID.isin([ind]), y_coord_name], special_colors[j])
        
        ax.set_title(f"Segments {title_suffix}" if i == 0 else f"Segments {title_suffix}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if save_file:
            ax.figure.savefig(f"{save_file}.png", format='png', bbox_inches="tight", dpi=300)
        if show:
            plt.show()

def plot_reduced_morphology(seg_data, elevation=0, azimuth=-100, radius_scale=1.0, deleted_indices=[], show=True, color='red', figsize=(10,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for i, seg in seg_data.iterrows():
        x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
        y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
        z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]
        radius = seg['r'] * radius_scale
        if i in deleted_indices:
            c = color
            r = radius * 2
        else:
            c = 'black'
            r = radius
        ax.plot(x_points, z_points, y_points, linewidth=r, color=c)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    if show:
        plt.show()
    return fig, ax

def plot_morphology_with_highlighted_sec_type(sec_type, seg_data, color='red', **kwargs):
    if (sec_type not in np.unique(seg_data['sec_type_precise'])) and  (sec_type not in ['unlabeled', 'overlapping']):
        print(f"{sec_type} not found in seg_data. Instead, seg_data has sec_types: {np.unique(seg_data['sec_type_precise'])}")
    if sec_type == 'unlabeled': # segments without a seg_id.
        fig, ax = plot_reduced_morphology(seg_data,deleted_indices=seg_data[seg_data['sec_type_precise'].isna()]['seg_id'].tolist(), color=color, **kwargs)
    else:
        fig, ax = plot_reduced_morphology(seg_data,deleted_indices=seg_data[seg_data['sec_type_precise'] == sec_type]['seg_id'].tolist(), color=color, **kwargs)
    return fig, ax

def plot_clusters(seg_data, clustering_config, synapse_coords=None, ax=None, elevation=20, azimuth=-100, radius_scale=1.0, title=''):
    """
    Visualize the clustering configuration including functional groups and presynaptic cells.
    
    Parameters:
    -----------
    seg_data : pd.DataFrame
        DataFrame containing segment data
    clustering_config : dict
        Dictionary containing clustering configuration (from parameters)
    synapse_coords : np.ndarray, optional
        Array of synapse coordinates to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    elevation : float, optional
        Elevation angle for 3D view
    azimuth : float, optional
        Azimuth angle for 3D view
    radius_scale : float, optional
        Scale factor for segment radii
    title : str, optional
        Title for the plot
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot the cell morphology in 3D
    plot(seg_data, {i: 0 for i in seg_data.index}, ax, elevation=elevation, azimuth=azimuth, radius_scale=radius_scale)
    
    # Plot synapses if provided
    if synapse_coords is not None:
        ax.scatter(
            synapse_coords[:, 0],  # X
            synapse_coords[:, 2],  # Y # altered to match plot()
            synapse_coords[:, 1],  # Z
            c='gray', alpha=0.3, s=5, label='Synapses'
        )
    
    # Plot functional groups and presynaptic cells
    for sec_type, sec_config in clustering_config.items():
        for fg_idx, fg in enumerate(sec_config.get('functional_groups', [])):
            # Plot functional group center and radius
            fg_center = np.array(fg['center'])
            fg_radius = fg['radius']
            
            # Check if there are any synapses within this functional group
            if synapse_coords is not None:
                distances_to_fg = np.sqrt(np.sum((synapse_coords - fg_center)**2, axis=1))
                synapses_in_fg = np.sum(distances_to_fg <= fg_radius)
                if synapses_in_fg == 0:
                    warnings.warn(f"No synapses found in functional group {fg_idx} of section type {sec_type}")
            
            # Create a sphere for the functional group
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = fg_center[0] + fg_radius * np.outer(np.cos(u), np.sin(v))
            y = fg_center[2] + fg_radius * np.outer(np.sin(u), np.sin(v))
            z = fg_center[1] + fg_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, color='blue', alpha=0.1, label=f'FG {fg_idx}' if fg_idx == 0 else None)
            ax.scatter(fg_center[0], fg_center[2], fg_center[1], color='blue', s=50, 
                      label=f'FG Center {fg_idx}' if fg_idx == 0 else None)
            
            # Plot presynaptic cells
            for pc_idx, pc in enumerate(fg.get('presynaptic_cells', [])):
                pc_center = np.array(pc['center']) #+ fg_center
                pc_radius = pc['radius']
                
                # Check if there are any synapses within this presynaptic cell
                if synapse_coords is not None:
                    distances_to_pc = np.sqrt(np.sum((synapse_coords - pc_center)**2, axis=1))
                    synapses_in_pc = np.sum(distances_to_pc <= pc_radius)
                    if synapses_in_pc == 0:
                        warnings.warn(f"No synapses found in presynaptic cell {pc_idx} of functional group {fg_idx} in section type {sec_type}")
                
                # Create a sphere for the presynaptic cell
                x = pc_center[0] + pc_radius * np.outer(np.cos(u), np.sin(v))
                y = pc_center[2] + pc_radius * np.outer(np.sin(u), np.sin(v))
                z = pc_center[1] + pc_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x, y, z, color='red', alpha=0.1, 
                              label=f'PC {pc_idx}' if pc_idx == 0 and fg_idx == 0 else None)
                ax.scatter(pc_center[0], pc_center[2], pc_center[1], color='red', s=30,
                          label=f'PC Center {pc_idx}' if pc_idx == 0 and fg_idx == 0 else None)
    
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_zlabel('Z (um)')
    ax.set_title(title)
    ax.legend()
    
    return ax

def plot_morphology_with_highlighted_sec_types(sec_types, seg_data, colors=None, figsize=(12, 10), dpi=600, save_path=None):
    """
    Plot morphology with multiple section types highlighted in different colors.
    
    Parameters:
    -----------
    sec_types : list
        List of section types to highlight. Can include 'unlabeled' and 'overlapping'.
        If 'distal_apical' is included, it will automatically include 'tuft', 'oblique', 'trunk', and 'nexus'.
    seg_data : pd.DataFrame
        DataFrame containing segment data with 'sec_type_precise' and 'seg_id' columns
    colors : list, optional
        List of colors to use for highlighting. If None, will use default colors.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (12, 10).
    dpi : int, optional
        Dots per inch for the figure. Default is 600.
    save_path : str, optional
        If provided, the figure will be saved to this path with the specified DPI.
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    if len(sec_types) > len(colors):
        warnings.warn(f"More section types than colors provided. Some section types will share colors.")
    
    # Create figure and axes with specified size
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all segments in black first
    for i, seg in seg_data.iterrows():
        x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
        y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
        z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]
        radius = seg['r']
        ax.plot(x_points, z_points, y_points, linewidth=radius, color='black')
    
    # Track which section types we've already plotted to avoid duplicate legend entries
    plotted_types = set()
    
    # Highlight specified section types
    for idx, sec_type in enumerate(sec_types):
        color = colors[idx % len(colors)]
        
        # Handle distal_apical special case
        if sec_type == 'distal_apical':
            apical_types = ['tuft', 'oblique', 'trunk', 'nexus']
            # Only add the label once for distal_apical
            label_added = False
            for apical_type in apical_types:
                highlighted_indices = seg_data[seg_data['sec_type_precise'] == apical_type]['seg_id'].tolist()
                if highlighted_indices:
                    for i, seg in seg_data[seg_data['seg_id'].isin(highlighted_indices)].iterrows():
                        x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
                        y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
                        z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]
                        radius = seg['r'] * 2
                        ax.plot(x_points, z_points, y_points, linewidth=radius, color=color, 
                              label='distal_apical' if not label_added else None)
                    if not label_added:
                        label_added = True
        else:
            if sec_type == 'unlabeled':
                highlighted_indices = seg_data[seg_data['sec_type_precise'].isna()]['seg_id'].tolist()
            else:
                highlighted_indices = seg_data[seg_data['sec_type_precise'] == sec_type]['seg_id'].tolist()
                
            if not highlighted_indices:
                warnings.warn(f"No segments found for section type: {sec_type}")
                continue
                
            # Plot highlighted segments
            for i, seg in seg_data[seg_data['seg_id'].isin(highlighted_indices)].iterrows():
                x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
                y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
                z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]
                radius = seg['r'] * 2
                ax.plot(x_points, z_points, y_points, linewidth=radius, color=color, 
                       label=sec_type if sec_type not in plotted_types else None)
            plotted_types.add(sec_type)
    
    ax.view_init(elev=0, azim=-100)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 2, 1])
    
    # Remove duplicate labels from legend and position it inside the plot in top right
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
             loc='upper right',           # Position in top right
             bbox_to_anchor=(0.80, 0.70), # Slightly inset from the corner
             framealpha=0.8)             # Make the legend background slightly transparent
    
    # Adjust layout to minimize whitespace
    plt.tight_layout(pad=0.001)  # Reduce padding around the plot
    
    # Save figure if save_path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)  # Reduce padding when saving
    
    return fig, ax

def plot_morphology_with_y_range(seg_data, y_min=685, y_max=885, figsize=(12, 10), dpi=600, save_path=None):
    """
    Plot morphology with segments highlighted within a specific y-range.
    
    Parameters:
    -----------
    seg_data : pd.DataFrame
        DataFrame containing segment data with coordinates
    y_min : float, optional
        Minimum y-coordinate for highlighting. Default is 685.
    y_max : float, optional
        Maximum y-coordinate for highlighting. Default is 885.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (12, 10).
    dpi : int, optional
        Dots per inch for the figure. Default is 600.
    save_path : str, optional
        If provided, the figure will be saved to this path with the specified DPI.
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Create figure and axes with specified size
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all segments in black first
    for i, seg in seg_data.iterrows():
        x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
        y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
        z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]
        radius = seg['r']
        ax.plot(x_points, z_points, y_points, linewidth=radius, color='black')
    
    # Highlight segments within y-range
    highlighted_segments = seg_data[
        ((seg_data['p0_1'] >= y_min) & (seg_data['p0_1'] <= y_max)) |
        ((seg_data['pc_1'] >= y_min) & (seg_data['pc_1'] <= y_max)) |
        ((seg_data['p1_1'] >= y_min) & (seg_data['p1_1'] <= y_max))
    ]
    
    # Plot highlighted segments in red
    for i, seg in highlighted_segments.iterrows():
        x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
        y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
        z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]
        radius = seg['r'] * 2  # Make highlighted segments thicker
        ax.plot(x_points, z_points, y_points, linewidth=radius, color='red', 
               label='Y-range segments' if i == highlighted_segments.index[0] else None)
    
    ax.view_init(elev=0, azim=-100)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 2, 1])
    
    # Add legend
    if not highlighted_segments.empty:
        ax.legend(loc='upper right',
                 bbox_to_anchor=(0.80, 0.70),
                 framealpha=0.8)
    
    # Adjust layout to minimize whitespace
    plt.tight_layout(pad=0.001)
    
    # Save figure if save_path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    
    return fig, ax
    
def plot_morphology_flex( # flexible function to plot neuron morphology with various highlighting options
    seg_data,
    option='specific_sec_type',
    sec_types=None,
    y_min=None,
    y_max=None,
    out_dir=None,
    figsize=(10, 6),
    show=True,
    save=True,
    color=None,
    parameters=None,  # For auto-looping all types
    **kwargs
):
    """
    Flexible function to plot neuron morphology with various highlighting options.
    
    Parameters:
        seg_data: Data describing the neuron morphology.
        option: One of ['specific_sec_type', 'each_sec_type', 'y_range', 'single_type'].
        sec_types: List of section types (for 'specific_sec_type'); or single type as str for 'single_type'.
        y_min, y_max: For y_range option.
        out_dir: Directory to save plots; if None, no files saved unless save_path is passed directly.
        figsize: Tuple for figure size.
        show: Whether to display plots.
        save: Whether to save plots.
        color: Specify a color if desired (for same color).
        parameters: Used for auto-looping section types (must have inh_syn_properties as dict).
        **kwargs: Passed to the underlying plotting function.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    figs = []
    axs = []

    if option == 'specific_sec_type':
        # Highlight multiple section types, different colors (all on one plot)
        fig, ax = plot_morphology_with_highlighted_sec_types(
            sec_types, seg_data, figsize=figsize, **kwargs
        )
        if save and out_dir:
            fig.savefig(os.path.join(out_dir, 'highlighted_types.png'))
        if show:
            plt.show()
        figs.append(fig)
        axs.append(ax)

    elif option == 'each_sec_type':
        # Loop over types, all with same color (or auto)
        if parameters is None or not hasattr(parameters, "inh_syn_properties"):
            raise ValueError("parameters with inh_syn_properties required for this option.")
        for sec_type in parameters.inh_syn_properties.keys():
            print(f"Plotting {sec_type}:")
            fig, ax = plot_morphology_with_highlighted_sec_type(
                sec_type, seg_data, color=color if color else 'red', figsize=figsize, **kwargs
            )
            ax.set_title(sec_type)
            fig.tight_layout()
            if show:
                plt.show()
            if save and out_dir:
                fig.savefig(os.path.join(out_dir, f"{sec_type}.png"))
            figs.append(fig)
            axs.append(ax)

    elif option == 'y_range':
        # Highlight custom y-range
        fig, ax = plot_morphology_with_y_range(
            seg_data, y_min=y_min, y_max=y_max, figsize=figsize, **kwargs
        )
        if save and out_dir:
            fig.savefig(os.path.join(out_dir, f"y_range_{y_min}_{y_max}.png"))
        if show:
            plt.show()
        figs.append(fig)
        axs.append(ax)

    elif option == 'single_type':
        # Highlight a single section type
        sec_type = sec_types if isinstance(sec_types, str) else sec_types[0]
        fig, ax = plot_morphology_with_highlighted_sec_type(
            sec_type, seg_data, color=color if color else 'red', figsize=figsize, **kwargs
        )
        ax.set_title(sec_type)
        fig.tight_layout()
        if save and out_dir:
            fig.savefig(os.path.join(out_dir, f"{sec_type}.png"))
        if show:
            plt.show()
        figs.append(fig)
        axs.append(ax)
    
    else:
        raise ValueError("Unknown option. Choose from: 'specific_sec_type', 'each_sec_type', 'y_range', 'single_type'.")
    
    return figs, axs
