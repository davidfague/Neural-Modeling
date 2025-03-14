import matplotlib.pyplot as plt
import matplotlib as mpl

# generic function for plotting variable over morphology
def plot(seg_data, data_to_plot, ax, elevation=20, azimuth=-100, radius_scale=1.0, title='', clim_max=30, return_cbar = False):
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

def plot_special_segments(seg_data, special_indices, special_colors, title_suffix=""):
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

def plot_reduced_morphology(seg_data, elevation=0, azimuth=-100, radius_scale=1.0, deleted_indices=[]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, seg in seg_data.iterrows():
        # Extract x, y, z coordinates
        x_points = [seg['p0_0'], seg['pc_0'], seg['p1_0']]
        y_points = [seg['p0_1'], seg['pc_1'], seg['p1_1']]
        z_points = [seg['p0_2'], seg['pc_2'], seg['p1_2']]

        # Calculate line width and set color based on deleted_indices
        radius = seg['r'] * radius_scale
        color = 'red' if i in deleted_indices else 'black'
        if i in deleted_indices:
            radius *= 0.1  # adjust multiplier to change red line width

        # Note: the order is (x, z, y) to match the original orientation.
        ax.plot(x_points, z_points, y_points, linewidth=radius, color=color)

    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.show()
