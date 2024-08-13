import numpy as np
import os
import pickle, h5py

from tqdm import tqdm
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from matplotlib import colormaps, colors, cm
import matplotlib.animation as animation

class DataReader:

    @staticmethod
    def load_parameters(sim_folder):
        with open(os.path.join(sim_folder, "parameters.pickle"), "rb") as file:
            parameters = pickle.load(file)
        return parameters

    @staticmethod
    def read_data(sim_folder, sim_file_name, parameters=None):
        
        # For convenience
        if sim_file_name.endswith(".h5"):
            sim_file_name = sim_file_name[:-3]

        if parameters is None:
            with open(os.path.join(sim_folder, "parameters.pickle"), "rb") as file:
                parameters = pickle.load(file)

        step_size = int(parameters.save_every_ms / parameters.h_dt) 
        steps = range(step_size, int(parameters.h_tstop / parameters.h_dt) + 1, step_size)

        data = []
        for step in steps:
            with h5py.File(os.path.join(sim_folder, f"saved_at_step_{step}", sim_file_name + ".h5"), 'r') as file:
                retrieved_data = np.array(file["data"])

                # Spikes
                if len(retrieved_data.shape) == 1:
                    data.append(retrieved_data)

                # Traces
                elif len(retrieved_data.shape) == 2:
                    # Neuron saves traces inconsistently; sometimes the trace length is (t) and sometimes it is (t+1)
                    # Thus, cut the trace at parameters.save_every_ms
                    data.append(retrieved_data[:, :])#parameters.save_every_ms*parameters.record_every_time_steps])
        data = np.concatenate(data, axis = 1)

        return data

class SummaryStatistics:

    # http://www.columbia.edu/cu/appliedneuroshp/Spring2018/Spring18SHPAppliedNeuroLec4.pdf
    @staticmethod
    def spike_triggered_average(trace: np.ndarray, spike_times: np.ndarray, win_length: int):
        if len(trace.shape) != 2:
            raise ValueError("trace should be a 2d array; if it is a 1d array, try trace.reshape(1, -1)")
        
        # Delete spike times within the first window
        spike_times = np.delete(spike_times, np.where(spike_times < win_length))

        # Delete spikes which occured after the trace's end
        spike_times = np.delete(spike_times, np.where(spike_times > trace.shape[1] - win_length))

        sta = np.zeros(win_length)
        # Add trace[spike - window: spike] to the trace
        for sp_time in spike_times:
            sta = sta + trace[:, int(sp_time - win_length // 2): int(sp_time + win_length // 2)]

        # Average over all spikes
        sta = sta / (len(spike_times) + 1e-15)

        return sta

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4205553/#:~:text=To%20quantify%20the%20correlation%20between,is%20insensitive%20to%20firing%20rate.
    @staticmethod
    def spike_time_tiling_coefficient(
            spike_times_A: np.ndarray, 
            spike_times_B: np.ndarray, 
            time_A: int, 
            time_B: int, 
            win_length: int = 1):
        
        def get_T(spike_times: np.ndarray, total_time: int, win_length: int):
            T = np.zeros(total_time)
            for sp_time in spike_times:
                T[int(sp_time - win_length) : int(sp_time + win_length)] = 1
            return np.mean(T)
        
        def get_P(spike_times_A: np.ndarray, spike_times_B: np.ndarray, win_length: int):
            counter = 0
            for sp_time_A in spike_times_A:
                for sp_time_B in spike_times_B:
                    if np.abs(sp_time_A - sp_time_B) < win_length:
                        counter += 1
                        continue
            return counter / len(spike_times_A)

        
        TA = get_T(spike_times_A, time_A, win_length)
        TB = get_T(spike_times_B, time_B, win_length)
        PA = get_P(spike_times_A, spike_times_B, win_length)
        PB = get_P(spike_times_B, spike_times_A, win_length)

        STTC = 0.5 * ((PA - TB) / (1 - PA * TB) + (PB - TA) / (1 - PB * TA))
        return STTC

    @staticmethod
    def get_quantiles_based_on_elec_dist(morph, elec_dist, spikes, section):
        filtered_elec_dist = elec_dist.loc[morph.section == section, "beta_passive"]
        filtered_spikes = [spikes[i] for i in np.where(morph.section == section)[0]]

        if len(filtered_spikes) < 10:
            raise RuntimeError(f"Found less than 10 spikes when computing quantiles for {section}.")
        
        q = np.quantile(filtered_elec_dist, np.arange(0, 1.1, 0.1))
        return q

    @staticmethod
    def bin_matrix_to_quantiles(matrix, quantiles, var_to_bin, cumulative=False):
        out = np.zeros((len(quantiles) - 1, matrix.shape[1]))

        if cumulative:
          for i in range(len(quantiles) - 1):
             inds = np.where((var_to_bin > quantiles[i]) & ((var_to_bin < quantiles[i + 1])))[0]
             out[i] = np.sum(matrix[inds], axis = 0)
        else:
          for i in range(len(quantiles) - 1):
             inds = np.where((var_to_bin > quantiles[i]) & ((var_to_bin < quantiles[i + 1])))[0]
             out[i] = np.mean(matrix[inds], axis = 0)
        
        return out


class Trace:

    @staticmethod
    def get_crossings(data, threshold):

        # Find threshold crossings
        threshold_crossings = np.diff(data > threshold)

        # Determine if the trace starts above or below threshold to get upward crossings
        if data[0] < threshold:
            upward_crossings = np.argwhere(threshold_crossings)[::2]
            downward_crossings = np.argwhere(threshold_crossings)[1::2]
        else:
            upward_crossings = np.argwhere(threshold_crossings)[1::2]
            downward_crossings = np.argwhere(threshold_crossings)[::2]
        
        if len(downward_crossings) < len(upward_crossings): 
            upward_crossings = upward_crossings[:-1]

        return upward_crossings, downward_crossings
    
    def get_duration(spike_times, downward_crossings):
        # For each spike time, find the closest downward crossing which occured after that time
        durations = []
        for st in spike_times:
            closest_dc = downward_crossings[np.where(downward_crossings - st > 0)[0][0]]
            durations.append(float(closest_dc - st))
        return durations
    
class CurrentTrace(Trace):
    
    @staticmethod
    def compute_axial_currents(v, seg_data, adj_matrix):
        ac_matrix = np.zeros((len(adj_matrix), len(adj_matrix), v.shape[-1]))
        for i in range(len(adj_matrix)):
            # Children
            for j in np.where(adj_matrix[i, :] == 1)[0]:
                # Child - parent
                ac = -(v[i] - v[j]) / (seg_data.loc[i, "seg_half_seg_RA"] + seg_data.loc[j, "seg_half_seg_RA"])
                ac_matrix[i, j] = ac
        return ac_matrix

def plot_spike_windows(spike_times, v_Na, v_Soma, window_size=10):
    num_spikes = len(spike_times)
    fig, axes = plt.subplots(num_spikes, 1, figsize=(12, 6 * num_spikes))

    for i, spike_time in enumerate(spike_times):
        start = max(0, spike_time - window_size)
        end = min(len(v_Na), spike_time + window_size)

        axes[i].plot(range(start, end), v_Na[start:end], label='v_Na', color='blue')
        axes[i].plot(range(start, end), v_Soma[start:end], label='v_Soma', color='red')
        axes[i].axvline(x=spike_time, color='green', linestyle='--', label='Spike Time')
        
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Voltage')
        axes[i].legend()
        axes[i].set_title(f'Spike at Time {spike_time}')

    plt.tight_layout()
    plt.show()

class VoltageTrace(Trace):

    @staticmethod
    def get_Na_spikes(g_Na: np.ndarray, threshold: float, spikes: np.ndarray, ms_within_spike: float, v_Na: np.ndarray, v_Soma: np.ndarray):
        upward_crossings, downward_crossings = VoltageTrace.get_crossings(g_Na, threshold)

        if len(upward_crossings) == 0:
            return np.array([]), np.array([]), np.array([])

        Na_spikes = []
        Na_spike_ends = []
        backprop_AP = []

        for sp_time in upward_crossings:
            # Time of APs before this Na spike
            spikes_before_sodium_spike = spikes[spikes < sp_time]
            spikes_after_sodium_spike = spikes[spikes > sp_time]

            # Find the corresponding end time for the current spike
            spike_end = next((down_time for down_time in downward_crossings if down_time > sp_time), None)

            # Na spike starts less than x ms before first AP after Na spike, and Na spike voltage is less than soma voltage at time of Na spike start (bAP that was being counted as Na spike)
            if (len(spikes_after_sodium_spike) > 0):
                # if ((spikes_after_sodium_spike[0] - sp_time) < 5) and (v_Na[sp_time] < v_Soma[sp_time]):
                #     backprop_AP.append(sp_time)

                # Na spike has no AP before
                if len(spikes_before_sodium_spike) == 0:#*******elif len(spikes_before_sodium_spike) == 0:
                    Na_spikes.append(sp_time)
                    Na_spike_ends.append(spike_end)
    
                # Na spike is more than x ms after last AP
                elif (sp_time - spikes_before_sodium_spike[-1] > ms_within_spike):
                    Na_spikes.append(sp_time)
                    Na_spike_ends.append(spike_end)
                
                # Na spike is within x ms after latest AP and counted as a back propagating AP
                else:
                    backprop_AP.append(sp_time)
            else:
                if len(spikes_before_sodium_spike) == 0:
                    Na_spikes.append(sp_time)
                    Na_spike_ends.append(spike_end)
    
                # Na spike is more than x ms after last AP
                elif (sp_time - spikes_before_sodium_spike[-1] > ms_within_spike):
                    Na_spikes.append(sp_time)
                    Na_spike_ends.append(spike_end)
                
                # Na spike is within x ms after latest AP and counted as a back propagating AP
                else:
                    backprop_AP.append(sp_time)

        return np.array(Na_spikes), np.array(Na_spike_ends), np.array(backprop_AP)
    
    @staticmethod
    def get_Ca_spikes(v, threshold, ica, dur_threshold_in_timesteps = 20):
        upward_crossings, downward_crossings = VoltageTrace.get_crossings(v, threshold)
        upward_crossings, downward_crossings = VoltageTrace.time_criterion(upward_crossings, downward_crossings, dur_threshold_in_timesteps)
        left_bounds, right_bounds, sum_currents = VoltageTrace.current_criterion(upward_crossings, downward_crossings, ica)
        left_bounds, right_bounds, sum_currents = VoltageTrace.time_criterion(left_bounds, right_bounds, dur_threshold_in_timesteps, sum_currents=sum_currents, return_list = True)
        # left_bounds, right_bounds, sum_currents = VoltageTrace.Hausser_current_criterion(upward_crossings, downward_crossings, ica)
        # checking if we can use time_criterion after current_criterion
        # print(f"shape legit crossings: {upward_crossings.shape, downward_crossings.shape}") # crossings are shape (n, 1)
        # print(f"shape bounds: {left_bounds.shape, right_bounds.shape}") # bounds are lists
        # print(len(left_bounds))
        return left_bounds, right_bounds, sum_currents
    
    @staticmethod
    def get_NMDA_spikes(v, threshold, inmda):
        return VoltageTrace.get_Ca_spikes(v, threshold, inmda, dur_threshold_in_timesteps = 26)
    
    @staticmethod
    def current_criterion(upward_crossings, downward_crossings, control_current):
        left_bounds = []
        right_bounds = []
        sum_current = []

        for crossing_index in np.arange(len(upward_crossings)):
            # Get current for this upward crossing
            e1 = control_current[upward_crossings[crossing_index]]

            # All the indices within an arch where current is less than 130% of e1
            x30 = np.argwhere(
                np.diff(
                    control_current[int(upward_crossings[crossing_index]):int(downward_crossings[crossing_index])] < 1.3 * e1, 
                    prepend = False))
            
            if len(x30) == 0: continue

            # All the indices within an arch where current is less than 115% of e1
            x15 = np.argwhere(
                np.diff(
                    control_current[int(upward_crossings[crossing_index]):int(downward_crossings[crossing_index])] < 1.15 * e1, 
                    prepend = False))
            
            stop = False
            while stop == False:
                left_bound = x30[0]

                # There are both x30 and x15
                if len(x15[x15 > left_bound]) != 0:
                    right_bound = np.sort(x15[x15 > left_bound])[0]
                else: # There is only x30
                    right_bound = (downward_crossings[crossing_index] - upward_crossings[crossing_index])
                    stop = True
                
                # calculate the charge
                current_sum = np.sum(control_current[int(upward_crossings[crossing_index] + left_bound) : int(upward_crossings[crossing_index] + right_bound)])
                if abs(current_sum) > 0.1: # charge threshold
                    left_bounds.append(upward_crossings[crossing_index] + left_bound)
                    right_bounds.append(upward_crossings[crossing_index] + right_bound)
                    sum_current.append(current_sum) # @MARK there was a divide by 10 in Ben's code, possibly for 0.1 ms dt.

                x30 = x30[x30 > upward_crossings[crossing_index] + right_bound]
                if len(x30) == 0: stop = True

        return left_bounds, right_bounds, sum_current
    
    @staticmethod
    def Hausser_current_criterion(upward_crossings, downward_crossings, control_current, charge_threshold=0.1):
        spike_start_times = []
        spike_end_times = []
        spike_charges = []

        for crossing_index in range(len(upward_crossings)):
            # e1 = control_current[upward_crossings[crossing_index]]

            # detect peak current and time index within 6 s after Vm_threshold upward crossing
            crossing_idx = upward_crossings[crossing_index] # Get the index of the crossing
            values_after_crossing = control_current[int(crossing_idx):int(crossing_idx)+6] # Get the values at the 6 indices after the crossing
            # @MARK 6 indices are for dt=1ms; use 60 indices for dt=0.1ms
            # peak_value = min(values_after_crossing) # Find the peak (negative values)
            # peak_index = values_after_crossing.index(peak_value) + crossing_idx # Get the index where the maximum value occurs
            peak_index = np.argmin(values_after_crossing) + crossing_idx
            peak_current = control_current[peak_index] # Set e1 to the maximum value

            # current trace between peak and downward crossing
            current_after_peak = control_current[int(peak_index):int(downward_crossings[crossing_index])]

            # Boolean masks for currents less than 30% and 15% of peak (a negative value)
            mask_30 = abs(current_after_peak) < abs(0.3 * peak_current) # where the current magnitiude is less than 30% the peak current magnitude
            mask_15 = abs(current_after_peak) > abs(0.15 * peak_current)

            if not np.any(mask_30) or not np.any(mask_15):
                # print("never less than 30%")
                continue

            left_bound = peak_index # official start of spike is at the peak current

            # Find the first index in the segment where the current is less than 15% of e1
            x15_indices = np.where(mask_15)[0]
            if len(x15_indices) == 0:
                # print('never less than 30%')
                continue
            right_bound = left_bound + x15_indices[0]

            # Find the first index in the segment where the current is less than 30% of e1
            x30_indices = np.where(mask_30)[0]
            if len(x30_indices) == 0:
                # print('suspicious never less than 30%')
                continue
            charge_bound = left_bound + x30_indices[0] # use 30% crossing to calculate total charge

            # Calculate the total charge from left_bound to current_bound
            # print(left_bound, current_bound)
            total_charge = np.sum(np.abs(control_current[int(left_bound):int(charge_bound)])) #@MARK divide by 10 if dt=0.l ms instead of dt=1ms
            # print(total_current)
            
            if total_charge >= charge_threshold:
                # print('successful detection')
                spike_start_times.append(left_bound)
                spike_end_times.append(right_bound)
                spike_charges.append(total_charge)
            else:
                pass# print('current threshold not met')

        return spike_start_times, spike_end_times, spike_charges
    @staticmethod
    def time_criterion(upward_crossings, downward_crossings, dur_threshold_in_timesteps = 26, sum_currents=None, return_list = False, dur_max_threshold_in_timesteps = 250): # rihgt-now, timestep is in ms. 26 ms-NMDA; 20 ms-Ca
        
        # Check if the length of crossings is even
        if len(upward_crossings) != len(downward_crossings):
            raise ValueError("Mismatch in length of upward and downward crossings")
        elif (len(upward_crossings) == 0) and (sum_currents is not None):
            return upward_crossings, downward_crossings, sum_currents
        elif (len(upward_crossings) == 0):
            return upward_crossings, downward_crossings

        if isinstance(downward_crossings, list):
            upward_crossings = np.array(upward_crossings)
            downward_crossings = np.array(downward_crossings)
            
        # Calculate durations
        durations = downward_crossings - upward_crossings
        
        # Apply time threshold
        valid_indices = (durations > dur_threshold_in_timesteps) & (durations < dur_max_threshold_in_timesteps)
        if return_list:
            legit_up_crossings = list(upward_crossings[valid_indices])
            legit_down_crossings = list(downward_crossings[valid_indices])
        else:
            legit_up_crossings = upward_crossings[valid_indices].reshape(-1,1)
            legit_down_crossings = downward_crossings[valid_indices].reshape(-1,1)
            
        if sum_currents is not None:
            sum_currents = list(np.array(sum_currents).reshape(-1,1)[valid_indices])
            return legit_up_crossings, legit_down_crossings, sum_currents
        else:
            return legit_up_crossings, legit_down_crossings
        
        
    

class CellGraph:

    def __init__(self, seg_data):
        self.N = len(seg_data)
        self.start_coords = seg_data.loc[:, ["p0_0", "p0_1", "p0_2"]].to_numpy()
        self.center_coords = seg_data.loc[:, ["pc_0", "pc_1", "pc_2"]].to_numpy()
        self.end_coords = seg_data.loc[:, ["p1_0", "p1_1", "p1_2"]].to_numpy()

    def animate_cell(self, variable, step):
        plots = []
        fig = plt.figure(figsize = (10, 20))
        for i in tqdm(range(0, variable.shape[1], step)):
            fig_i = self.plot_cell(color = variable[:, i])
            ax_i = fig_i.gca(); ax_i.set(animated = True); ax_i.remove()
            ax_i.figure = fig; fig.add_axes(ax_i); plt.close(fig_i)
            plots.append([ax_i])
        cmap = colormaps.get_cmap("Spectral")
        # Normalize because color maps are defined in [0, 1]
        norm = colors.Normalize(np.min(variable), np.max(variable))
        fig.colorbar(cm.ScalarMappable(norm = norm, cmap = cmap), ax = fig.gca(), fraction = 0.026, pad = 0.04)
        anime = animation.ArtistAnimation(fig, plots)
        return anime

    def plot_cell(self, color = None):
        fig = plt.figure(figsize = (10, 20))
        ax = fig.add_subplot(projection = '3d')

        # All these values are heuristics
        ax.view_init(elev = 20, azim = 10)
        r = Rotation.from_euler('yx', (90, 90), degrees = True)
        start_coords = r.apply(self.start_coords)
        end_coords = r.apply(self.end_coords)

        if color is not None:
            cmap = colormaps.get_cmap("Spectral")
            # Normalize because color maps are defined in [0, 1]
            norm = colors.Normalize(np.min(color), np.max(color))
            fig.colorbar(cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax, fraction = 0.026, pad = 0.04)

        for i in range(self.N):
            ax.plot(
                np.hstack([start_coords[i, 0], end_coords[i, 0]]),
                np.hstack([start_coords[i, 1], end_coords[i, 1]]),
                np.hstack([start_coords[i, 2], end_coords[i, 2]]),
                color = cmap(norm(color[i])) if color is not None else 'teal'
            )
        return fig

    def compute_adjacency_matrix(self):
        adj_matrix = np.zeros((self.N, self.N))
        for i in tqdm(range(self.N)):
            for j in range(self.N):
                adj_matrix[i, j] = self._find_zero_diff_in_two_coord_lists(
                    [self.start_coords[i], self.center_coords[i], self.end_coords[i]],
                    [self.start_coords[j], self.center_coords[j], self.end_coords[j]])
        return adj_matrix - np.eye(self.N)

    def _find_zero_diff_in_two_coord_lists(self, list0, list1):
        for c0 in list0:
            for c1 in list1:
                if np.sum(np.abs(c0 - c1)) < 1e-10:
                    return 1
        return 0

# UNFINISHED BUSINESS
class ECP:
    # Adapted from
    # https://github.com/chenziao/Stylized-Single-Cell-and-Extracellular-Potential/tree/main/cell_inference

    def calc_transfer_resistance(
            self,
            seg_coords,
            move_cell: list = None,
            scale: float = 1.0, 
            min_distance: float = None,
            move_elec: bool = False, 
            sigma: float = 0.3) -> None:
        """
        Precompute mapping from segment to electrode locations
        move_cell: list/tuple/2-by-3 array of (translate,rotate), rotate the cell followed by translating it
        scale: scaling factor of ECP magnitude
        min_distance: minimum distance allowed between segment and electrode, if specified
        sigma: resistivity of medium (mS/mm)
        move_elec: whether or not to relatively move electrodes for calculation
        """
        if move_cell is not None:
            move_cell = np.asarray(move_cell).reshape((2, 3))
        if move_elec and move_cell is not None:
            elec_coords = self.move_position(move_cell[0], move_cell[1], self.elec_coords, True)
        else:
            elec_coords = self.elec_coords
        if not move_elec and move_cell is not None:
            dl = self.move_position([0., 0., 0.], move_cell[1], seg_coords['dl'])
            pc = self.move_position(move_cell[0], move_cell[1], seg_coords['pc'])
        else:
            dl = seg_coords['dl']
            pc = seg_coords['pc']
        if min_distance is None:
            r = seg_coords['r']
        else:
            r = np.fmax(seg_coords['r'], min_distance)
        rr = r ** 2
        
        tr = np.empty((self.nelec, self.cell._nseg))
        for j in range(self.nelec):  # calculate mapping for each site on the electrode
            rel_pc = elec_coords[j, :] - pc  # distance between electrode and segment centers
            # compute dot product row-wise, the resulting array has as many rows as original
            r2 = np.einsum('ij,ij->i', rel_pc, rel_pc)
            rlldl = np.einsum('ij,ij->i', rel_pc, dl)
            dlmag = np.linalg.norm(dl, axis=1)  # length of each segment
            rll = abs(rlldl / dlmag)  # component of r parallel to the segment axis it must be always positive
            r_t2 = r2 - rll ** 2  # square of perpendicular component
            up = rll + dlmag / 2
            low = rll - dlmag / 2
            np.fmax(r_t2, rr, out=r_t2, where=low - r < 0)
            num = up + np.sqrt(up ** 2 + r_t2)
            den = low + np.sqrt(low ** 2 + r_t2)
            tr[j, :] = np.log(num / den) / dlmag  # units of (um) use with im_ (total seg current)
        tr *= scale / (4 * np.pi * sigma)
        return tr
    
    def move_position(
            translate: np.ndarray,
            rotate: np.ndarray,
            old_position: np.ndarray = None,
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
                


            
            

            
            