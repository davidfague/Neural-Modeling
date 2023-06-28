import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional, List, Tuple
from scipy.spatial.transform import Rotation
#TODO CHECK
def move_position(translate: Union[List[float],Tuple[float],np.ndarray],
                  rotate: Union[List[float],Tuple[float],np.ndarray],
                  old_position: Optional[Union[List[float], np.ndarray]] = None,
                  move_frame: bool = False) -> np.ndarray:
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

#TODO: add docstirng
def plot_morphology(sim = None, cellid: int = 0, cell: object = None,
					seg_coords: dict = None, sec_nseg: list = None,
					type_id: list = None, electrodes: np.ndarray = None,
					axes: list = [2, 0, 1], clr: list = None,
					elev: int = 20, azim: int = 10, move_cell: list = None,
					figsize: tuple = None, seg_property = None, segment_colors = None, sm = None) -> tuple:
	"""
	Plot morphology in 3D.

	Parameters:
	----------
	sim: object
		Simulation object

	cellid: int = 0
		Cell id in the simulation object.

	cell: object = None
		cell object. Ignore sim and cellid if specified.

	seg_coords: dict = None
		If not using sim or cell, a dictionary that includes dl, pc, r.

	sec_nseg: list = None
		If not using sim or cell, list of number of segments in each section.

	type_id: list = None
		If not using sim or cell, list of the swc type id of each section/segment.

	electrodes: np.ndarray = None
		Electrode positions. Default: None, not shown.

	axes: list = [2, 0, 1]
		Sequence of axes to display in 3d plot axes.
		Default: [2,0,1] show z, x, y in 3d plot x, y, z axes, so y is upward.

	clr: list
		List of colors for each type of section.

	Returns:
	----------
	fig: plt.Figure

	ax: plt.Axis
	"""
	if sim is None and cell is None:
		if seg_coords is None or sec_nseg is None or type_id is None:
			raise ValueError("If not using 'Simulation', input arguments 'seg_coords', 'sec_nseg', 'type_id' are required.")
		if clr is None:
			clr = ('g', 'r', 'b', 'c')
		if move_cell is None:
			move_cell = [0., 0., 0., 0., 1., 0.]
		sec_id_in_seg = np.cumsum([0] + list(sec_nseg[:-1]))
		type_id = np.asarray(type_id) - 1
		if type_id.size != len(sec_nseg):
			type_id = type_id = type_id[sec_id_in_seg]
		type_id = type_id.tolist()
		label_idx = np.array([type_id.index(i) for i in range(4)])
		lb_odr = np.argsort(label_idx)
		label_idx = label_idx[lb_odr].tolist()
		sec_name = np.array(('soma','axon','dend','apic'))[lb_odr]
	else:
		if clr is None:
			clr = ('g', 'b', 'pink', 'purple', 'r', 'c')
		if cell is None:
			if move_cell is None:
				move_cell = sim.loc_param[cellid, 0]
			cell = sim.cells[cellid]
		elif move_cell is None:
			move_cell = [0., 0., 0., 0., 1., 0.]
		seg_coords = cell.seg_coords
		sec_id_in_seg = cell.sec_id_in_seg
		sec_nseg = []
		sec_name = []
		label_idx = []
		type_id = []
		for i, sec in enumerate(cell.all):
			sec_nseg.append(sec.nseg)
			name = sec.name().split('.')[-1]
			if name not in sec_name:
				sec_name.append(name)
				label_idx.append(i)
			type_id.append(sec_name.index(name))
	label_idx.append(-1)

	move_cell = np.asarray(move_cell).reshape((2, 3))
	dl = move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
	pc = move_position(move_cell[0], move_cell[1], seg_coords['pc'])
	xyz = 'xyz'
	box = np.vstack([np.full(3, np.inf), np.full(3, np.NINF)])
	if electrodes is not None:
		box[0, axes[0:2]] = np.amin(electrodes[:, axes[0:2]], axis=0)
		box[1, axes[0:2]] = np.amax(electrodes[:, axes[0:2]], axis=0)

	fig = plt.figure(figsize=figsize)
	ax = plt.axes(projection='3d')
	lb_ptr = 0
	if segment_colors is None:
		for i, itype in enumerate(type_id):
			label = sec_name[lb_ptr] if i == label_idx[lb_ptr] else None
			if label is not None: lb_ptr += 1
			i0 = sec_id_in_seg[i]
			i1 = i0 + sec_nseg[i] - 1
			if sec_name[itype] == 'soma':
				p05 = (pc[i0] + pc[i1]) / 2
				ax.scatter(*[p05[j] for j in axes], c=clr[itype], s=20, label=label)
			else:
				p0 = pc[i0] - dl[i0] / 2
				p1 = pc[i1] + dl[i1] / 2
				ax.plot3D(*[(p0[j], p1[j]) for j in axes], color=clr[itype], label=label)
				box[0, :] = np.minimum(box[0, :], np.minimum(p0, p1))
				box[1, :] = np.maximum(box[1, :], np.maximum(p0, p1))
	else:
		for i, itype in enumerate(type_id):
			label = sec_name[lb_ptr] if i == label_idx[lb_ptr] else None
			if label is not None: lb_ptr += 1
			i0 = sec_id_in_seg[i] #segments list index of first segment of this section
			i1 = i0 + sec_nseg[i] - 1 #segments list index of last segment of this section
			if sec_name[itype] == 'soma':
				p05 = (pc[i0] + pc[i1]) / 2
				ax.scatter(*[p05[j] for j in axes], c=clr[itype], s=20, label=label)
			else:
				for seg_index in range(i0,i1+1):
					p0 = pc[seg_index] - dl[seg_index]/2
					p1 = pc[seg_index] + dl[seg_index]/2
					ax.plot3D(*[(p0[j], p1[j]) for j in axes], color=segment_colors[seg_index], label=label)
					box[0, :] = np.minimum(box[0, :], np.minimum(p0, p1))
					box[1, :] = np.maximum(box[1, :], np.maximum(p0, p1))
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


def plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, nexus_seg_index,
			    			loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, xlim=None, ylim=None, figsize: tuple =None):
	if xlim is None:
		xlim=t[[0, -1]]
		
	v_soma = Vm[soma_seg_index]
	v_tfut = Vm[tuft_seg_index]
	v_nexus = Vm[nexus_seg_index]
	v_axon = Vm[axon_seg_index]
	v_basal = Vm[basal_seg_index]

	if figsizeis None:
		plt.figure(figsize=(10, 4))
	else:
		plt.figure(figsize=figsize)
	plt.plot(t, v_soma, label='Soma')
	plt.plot(t, v_tfut, label='Tuft')
	plt.plot(t, v_nexus, label='Nexus')
	plt.plot(t, v_basal, label='Basal')
	#plt.plot(t, v_axon, label='axon')
	plt.ylabel('Membrane potential (mV)')
	plt.xlabel('time (ms)')
	plt.xlim(xlim)

	plt.legend()
	plt.savefig('Vm')

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
	if figsizeis None:
		plt.figure(figsize=(12, 5))
	else:
		plt.figure(figsize=figsize)
	_ = plot_lfp_heatmap(t=t, elec_d=elec_pos[e_idx, 1], lfp=lfp[:, e_idx],
											fontsize=fontsize, labelpad=labelpad, ticksize=ticksize, tick_length=tick_length,
											nbins=nbins, vlim = "auto", axes=plt.gca()) #vlim='auto';normal range seems to be ~ [-.00722,.00722]
	#plt.hlines(0,xmin=min(t),xmax=max(t),linestyles='dashed') # create a horizontal line
	plt.title('Extracellular potential heatmap')
	plt.xlim(xlim)
	plt.savefig('ECP heatmap')
	if figsizeis None:
		plt.figure(figsize=(8, 5))
	else:
		plt.figure(figsize=figsize)
	_ = plot_lfp_traces(t, lfp[:, e_idx][:,1::3], electrodes=elec_pos[e_idx][1::3],
											fontsize=fontsize, labelpad=labelpad, ticksize=ticksize, tick_length=tick_length,
											nbins=nbins, axes=plt.gca())
	plt.title('Extracellular potential timecourse')
	plt.xlim(xlim)
	plt.savefig('ECP timecourse')

	plt.show()
