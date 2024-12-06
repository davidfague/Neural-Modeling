import sys
sys.path.append("../")
sys.path.append("../Modules/")
sys.path.append("../cell_inference/")

from config import params

import os
import analysis
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from ecp import ECP

#from plotting_utils import plot_simulation_results

def plot_LFP(sim_directory, save_dir):
    i_membrane = analysis.DataReader.read_data(sim_directory, "i_membrane_")
    morph = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
    Vm = analysis.DataReader.read_data(sim_directory, "v")

    #dl: A NumPy array of shape [nseg, 3], representing the directional vectors from the start to the end of each segment.
    #pc: A NumPy array of shape [nseg, 3], representing the center points of each segment.
    #r: A NumPy array of shape [nseg], representing the radius of each segment.
    
    #morph['pc'] = morph.apply(lambda row: np.array([row['pc_0'], row['pc_1'], row['pc_2']]), axis=1)
    #pc_array = np.stack(morph['pc'].values)
    #morph['pc'] = pc_array
    #morph['dl'] = np.sqrt((morph['dl_0']**2) + (morph['dl_1']**2) + (morph['dl_2']**2))
    #morph['dl'] = morph.apply(lambda row: np.array([row['dl_0'], row['dl_1'], row['dl_2']]), axis=1)

    # Construct 'pc' and 'dl' arrays from the DataFrame
    pc = morph[['pc_0', 'pc_1', 'pc_2']].to_numpy()
    dl = morph[['dl_0', 'dl_1', 'dl_2']].to_numpy()
    r = morph['r'].to_numpy()

    # Create the seg_coords dictionary
    morph = {
        'pc': pc,
        'dl': dl,
        'r': r
    }
    print("morph['pc']: {morph['pc']}")
    print("morph['dl']: {morph['dl']}")

    elec_pos = params.ELECTRODE_POSITION
    ecp = ECP(i_membrane, seg_coords=morph, min_distance=params.MIN_DISTANCE)
    elec_pos = params.ELECTRODE_POSITION
    ecp.set_electrode_positions(elec_pos)
    ecp.calc_ecp()

    loc_param = [0., 0., 45., 0., 1., 0.]
    lfp = ecp.calc_ecp().T  # Unit: mV

    parameters = analysis.DataReader.load_parameters(sim_directory)
    t = np.arange(0, parameters.h_tstop + 1)  # ), parameters.h_dt)
    soma_seg_index = 0
    axon_seg_index = 10
    basal_seg_index = 20
    tuft_seg_index = 30
    nexus_seg_index = 40
    trunk_seg_index = 15
    plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, nexus_seg_index, trunk_seg_index,
                            loc_param, lfp, elec_pos, save_dir=save_dir)

def plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, 
							nexus_seg_index, trunk_seg_index, loc_param, lfp, elec_pos, 
				       xlim = None, ylim = None, figsize: tuple = None, vlim = 'auto',
							show = False, save_dir = None):
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

def plot_lfp_traces(t: np.ndarray, lfp: np.ndarray, electrodes: np.ndarray = None, vlim: str = 'auto',
                    fontsize: int = 40, labelpad: int = -30, ticksize: int = 40, tick_length: int = 15,
                    nbins: int = 3, savefig: str = None, axes = None):
    """
    Plot LFP traces.

    Parameters
    t: time points (ms). 1D array
    lfp: LFP traces (mV). If is 2D array, each column is a channel.
    electrodes: Electrode array coordinates. If specified, show waterfall plot following y coordinates.
    vlim: value limit for waterfall plot
    fontsize: size of font for display
    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
    ticksize: size of tick labels
    tick_length: length of ticks
    nbins: number of ticks
    savefig: if specified as string, save figure with the string as file name.
    axes: existing axes for the plot. If not specified, create new figure and axes.
    """
    t = np.asarray(t)
    lfp = np.asarray(lfp)
    if axes is None:
        fig = plt.figure()  # figsize=(15,15))
        ax = plt.gca()
    else:
        ax = axes
        fig = ax.get_figure()
        plt.sca(ax)
    if lfp.ndim == 2:
        if electrodes is None:
            for i in range(lfp.shape[1]):
                plt.plot(t, lfp[:, i])
        else:
            ind = np.lexsort(electrodes[:, [1, 0, 2][:electrodes.shape[1]]].T)[:lfp.shape[1]]
            clim = (electrodes[ind[0], 1], electrodes[ind[-1], 1])
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize())
            sm.set_clim(*clim)
            cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(clim[0], clim[1], nbins), pad=0.)
            cbar.ax.tick_params(length=tick_length, labelsize=ticksize)
            cbar.set_label('dist_y (\u03bcm)', fontsize=fontsize, labelpad=labelpad)
            if type(vlim) is str:
                if vlim == 'max':
                    vlim = np.amax(np.abs(lfp)) / 2 * np.array([-1, 1])
                elif vlim == 'minmax':
                    vlim = np.array([np.amin(lfp), np.amax(lfp)]) / 2
                else:
                    vlim = 1 * np.std(lfp) * np.array([-1, 1])
            xpos = (vlim[1] - vlim[0]) * np.arange(ind.size) - vlim[0]
            for j, i in enumerate(ind):
                plt.plot(t, xpos[j] + lfp[:, i], color=sm.to_rgba(electrodes[i, 1]))
            if np.isnan(xpos[-1]) or np.isinf(xpos[-1]) or np.isnan(vlim[1]) or np.isinf(vlim[1]):
                print("Invalid xpos or vlim value.")
                # Set to default values or handle the error appropriately
            else:
                ax.set_ylim([0, xpos[-1] + vlim[1]])
    else:
        plt.plot(t, lfp)
    ax.set_xlim(t[[0, -1]])
    plt.xlabel('ms', fontsize=fontsize)
    plt.ylabel('LFP (mV)', fontsize=fontsize, labelpad=labelpad)
    plt.locator_params(axis='both', nbins=nbins)
    plt.tick_params(length=tick_length, labelsize=ticksize)

    if savefig is not None:
        if type(savefig) is not str:
            savefig = 'LFP_trace.pdf'
        fig.savefig(savefig, bbox_inches='tight', transparent=True)
    return fig, ax


def plot_lfp_heatmap(t: np.ndarray, elec_d: np.ndarray, lfp: np.ndarray, 
                     vlim: str = 'auto', cbbox: list = None, cmap: str = 'viridis',
                     fontsize: int = 40, labelpad: int = -12, ticksize: int = 30, tick_length: int = 15,
                     nbins: int = 3, colorbar_label: str = None,
                     savefig: str = None, axes = None):
    """
    Plot LFP heatmap.

    t: time points (ms). 1D array
    elec_d: electrode distance (um). 1D array
    lfp: LFP traces (mV). If is 2D array, each column is a channel.
    vlim: value limit for color map. 'auto': using +/- 3-sigma of lfp for bounds as default.
          'max': use [-1, 1]*max(|lfp|). 'minmax': use [min(lfp), max(lfp)].
    cbbox: dimensions of colorbar
    cmap: A Colormap instance or registered colormap name. The colormap maps the C values to color.
    fontsize: size of font for display
    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
    ticksize: size of tick labels
    tick_length: length of ticks
    nbins: number of ticks
    savefig: if specified as string, save figure with the string as file name.
    axes: existing axes for the plot. If not specified, create new figure and axes.
    """
    if cbbox is None:
        cbbox = [.91, 0.118, .03, 0.76]
    lfp = np.asarray(lfp).T
    elec_d = np.asarray(elec_d) / 1000  # convert um to mm
    if type(vlim) is str:
        if vlim == 'max':
            vlim = np.amax(np.abs(lfp)) * np.array([-1, 1])
        elif vlim == 'minmax':
            vlim = [np.amin(lfp), np.amax(lfp)]
        else:
            vlim = 3 * np.std(lfp) * np.array([-1, 1])
    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes
        fig = ax.get_figure()
        plt.sca(ax)
    pcm = plt.pcolormesh(t, elec_d, lfp, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='auto')
    cbaxes = fig.add_axes(cbbox) if axes is None else None
    cbar = fig.colorbar(pcm, ax=ax, ticks=np.linspace(vlim[0], vlim[1], nbins), cax=cbaxes)
    cbar.ax.tick_params(length=tick_length, labelsize=ticksize)
    if colorbar_label is None: colorbar_label = 'LFP (mV)'
    cbar.set_label(colorbar_label, fontsize=fontsize, labelpad=labelpad)
    ax.set_xticks(np.linspace(t[0], t[-1], nbins))
    ax.set_yticks(np.linspace(elec_d[0], elec_d[-1], nbins))
    ax.tick_params(length=tick_length, labelsize=ticksize)
    ax.set_xlabel('time (ms)', fontsize=fontsize)
    ax.set_ylabel('dist_y (mm)', fontsize=fontsize)

    if savefig is not None:
        if type(savefig) is not str:
            savefig = 'LFP_heatmap.pdf'
        fig.savefig(savefig, bbox_inches='tight', transparent=True)
    return fig, ax

#def plot_morphology(
#                    seg_coords: dict = None, sec_nseg: List = None,
#                    type_id: Optional[List] = None, electrodes: Optional[np.ndarray] = None,
#                    axes: Union[List[int], Tuple[int]] = [2, 0, 1], clr: Optional[List[str]] = None,
#                    elev: int = 20, azim: int = 10, move_cell: Optional[Union[List,np.ndarray]] = None,
#                    figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
#    """
#    Plot morphology in 3D.
#
#    sim: simulation object
#    cellid: cell id in the simulation object. Default: 0
#    cell: stylized cell object. Ignore sim and cellid if specified
#    seg_coords: if not using sim or cell, a dictionary that includes dl, pc, r
#    sec_nseg: if not using sim or cell, list of number of segments in each section
#    type_id:  if not using sim or cell, list of the swc type id of each section/segment
#    electrodes: electrode positions. Default: None, not shown.
#    axes: sequence of axes to display in 3d plot axes.
#        Default: [2,0,1] show z,x,y in 3d plot x,y,z axes, so y is upward.
#    clr: list of colors for each type of section
#    Return Figure object, Axes object
#    """
#    if sim is None and cell is None:
#        if seg_coords is None or sec_nseg is None or type_id is None:
#            raise ValueError("If not using 'Simulation', input arguments 'seg_coords', 'sec_nseg', 'type_id' are required.")
#        if clr is None:
#            clr = ('g', 'r', 'b', 'c')
#        if move_cell is None:
#            move_cell = [0., 0., 0., 0., 1., 0.]
#        sec_id_in_seg = np.cumsum([0] + list(sec_nseg[:-1]))
#        type_id = np.asarray(type_id) - 1
#        if type_id.size != len(sec_nseg):
#            type_id = type_id = type_id[sec_id_in_seg]
#        type_id = type_id.tolist()
#        label_idx = np.array([type_id.index(i) for i in range(4)])
#        lb_odr = np.argsort(label_idx)
#        label_idx = label_idx[lb_odr].tolist()
#        sec_name = np.array(('soma','axon','dend','apic'))[lb_odr]
#    else:
#        if clr is None:
#            clr = ('g', 'b', 'pink', 'purple', 'r', 'c')
#        if cell is None:
#            if move_cell is None:
#                move_cell = sim.loc_param[cellid, 0]
#            cell = sim.cells[cellid]
#        elif move_cell is None:
#            move_cell = [0., 0., 0., 0., 1., 0.]
#        seg_coords = cell.seg_coords
#        sec_id_in_seg = cell.sec_id_in_seg
#        sec_nseg = []
#        sec_name = []
#        label_idx = []
#        type_id = []
#        for i, sec in enumerate(cell.all):
#            sec_nseg.append(sec.nseg)
#            name = sec.name().split('.')[-1]
#            if name not in sec_name:
#                sec_name.append(name)
#                label_idx.append(i)
#            type_id.append(sec_name.index(name))
#    label_idx.append(-1)
#
#    move_cell = np.asarray(move_cell).reshape((2, 3))
#    dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
#    pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
#    xyz = 'xyz'
#    box = np.vstack([np.full(3, np.inf), np.full(3, np.NINF)])
#    if electrodes is not None:
#        box[0, axes[0:2]] = np.amin(electrodes[:, axes[0:2]], axis=0)
#        box[1, axes[0:2]] = np.amax(electrodes[:, axes[0:2]], axis=0)
#
#    fig = plt.figure(figsize=figsize)
#    ax = plt.axes(projection='3d')
#    lb_ptr = 0
#    for i, itype in enumerate(type_id):
#        label = sec_name[lb_ptr] if i == label_idx[lb_ptr] else None
#        if label is not None: lb_ptr += 1
#        i0 = sec_id_in_seg[i]
#        i1 = i0 + sec_nseg[i] - 1
#        if sec_name[itype] == 'soma':
#            p05 = (pc[i0] + pc[i1]) / 2
#            ax.scatter(*[p05[j] for j in axes], c=clr[itype], s=20, label=label)
#        else:
#            p0 = pc[i0] - dl[i0] / 2
#            p1 = pc[i1] + dl[i1] / 2
#            ax.plot3D(*[(p0[j], p1[j]) for j in axes], color=clr[itype], label=label)
#            box[0, :] = np.minimum(box[0, :], np.minimum(p0, p1))
#            box[1, :] = np.maximum(box[1, :], np.maximum(p0, p1))
#    ctr = np.mean(box, axis=0)
#    r = np.amax(box[1, :] - box[0, :]) / 2
#    box = np.vstack([ctr - r, ctr + r])
#    if electrodes is not None:
#        idx = np.logical_and(np.all(electrodes >= box[0, :], axis=1), np.all(electrodes <= box[1, :], axis=1))
#        ax.scatter(*[(electrodes[idx, j], electrodes[idx, j]) for j in axes], color='orange', s=5, label='electrodes')
#    box = box[:, axes]
#    ax.auto_scale_xyz(*box.T)
#    ax.view_init(elev, azim)
#    ax.legend(loc=1)
#    ax.set_xlabel(xyz[axes[0]])
#    ax.set_ylabel(xyz[axes[1]])
#    ax.set_zlabel(xyz[axes[2]])
#    plt.show()
#    return fig, ax
#
#def plot_lfp_heatmap(t: np.ndarray, elec_d: np.ndarray, lfp: np.ndarray, 
#                     vlim: str = 'auto', cbbox: Optional[List[float]] = None, cmap: str = 'viridis',
#                     fontsize: int = 40, labelpad: int = -12, ticksize: int = 30, tick_length: int = 15,
#                     nbins: int = 3, colorbar_label: str = None,
#                     savefig: Optional[str] = None, axes: Axes = None) -> Tuple[Figure, Axes]:
#    """
#    Plot LFP heatmap.
#
#    t: time points (ms). 1D array
#    elec_d: electrode distance (um). 1D array
#    lfp: LFP traces (mV). If is 2D array, each column is a channel.
#    vlim: value limit for color map. 'auto': using +/- 3-sigma of lfp for bounds as default.
#          'max': use [-1, 1]*max(|lfp|). 'minmax': use [min(lfp), max(lfp)].
#    cbbox: dimensions of colorbar
#    cmap: A Colormap instance or registered colormap name. The colormap maps the C values to color.
#    fontsize: size of font for display
#    labelpad: Spacing in points from the Axes bounding box including ticks and tick labels.
#    ticksize: size of tick labels
#    tick_length: length of ticks
#    nbins: number of ticks
#    savefig: if specified as string, save figure with the string as file name.
#    axes: existing axes for the plot. If not specified, create new figure and axes.
#    """
#    if cbbox is None:
#        cbbox = [.91, 0.118, .03, 0.76]
#    lfp = np.asarray(lfp).T
#    elec_d = np.asarray(elec_d) / 1000  # convert um to mm
#    if type(vlim) is str:
#        if vlim == 'max':
#            vlim = np.amax(np.abs(lfp)) * np.array([-1, 1])
#        elif vlim == 'minmax':
#            vlim = [np.amin(lfp), np.amax(lfp)]
#        else:
#            vlim = 3 * np.std(lfp) * np.array([-1, 1])
#    if axes is None:
#        fig, ax = plt.subplots()
#    else:
#        ax = axes
#        fig = ax.get_figure()
#        plt.sca(ax)
#    pcm = plt.pcolormesh(t, elec_d, lfp, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='auto')
#    cbaxes = fig.add_axes(cbbox) if axes is None else None
#    cbar = fig.colorbar(pcm, ax=ax, ticks=np.linspace(vlim[0], vlim[1], nbins), cax=cbaxes)
#    cbar.ax.tick_params(length=tick_length, labelsize=ticksize)
#    if colorbar_label is None: colorbar_label = 'LFP (mV)'
#    cbar.set_label(colorbar_label, fontsize=fontsize, labelpad=labelpad)
#    ax.set_xticks(np.linspace(t[0], t[-1], nbins))
#    ax.set_yticks(np.linspace(elec_d[0], elec_d[-1], nbins))
#    ax.tick_params(length=tick_length, labelsize=ticksize)
#    ax.set_xlabel('time (ms)', fontsize=fontsize)
#    ax.set_ylabel('dist_y (mm)', fontsize=fontsize)
#
#    if savefig is not None:
#        if type(savefig) is not str:
#            savefig = 'LFP_heatmap.pdf'
#        fig.savefig(savefig, bbox_inches='tight', transparent=True)
#    return fig, ax

if __name__ == "__main__":
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
    else:
        raise RuntimeError

    if "-s" in sys.argv:
        save_dir = sys.argv[sys.argv.index("-s") + 1] # (global)
    else:
        raise RuntimeError
    plot_LFP(sim_directory, save_dir)