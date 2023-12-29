import numpy as np
import os
import pickle, h5py
import pandas as pd

class DataReader:

    @staticmethod
    def load_parameters(sim_folder):
        with open(os.path.join(sim_folder, "parameters.pickle"), "rb") as file:
            parameters = pickle.load(file)
        return parameters

    @staticmethod
    def read_data(sim_folder, sim_file_name):
        
        # For convenience
        if sim_file_name.endswith(".h5"):
            sim_file_name = sim_file_name[:-3]

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
                    data.append(retrieved_data[:, :parameters.save_every_ms])
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
        spike_times = np.delete(spike_times, np.where(spike_times > trace.shape[1]))

        sta = np.zeros(win_length)
        # Add trace[spike - window: spike] to the trace
        for sp_time in spike_times:
            sta = sta + trace[:, int(sp_time - win_length // 2): int(sp_time + win_length // 2)]

        # Average over all spikes
        sta = sta / len(spike_times)

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
    def get_quantiles_based_on_elec_dist(morph, elec_dist, spikes, elec_dist_var):
        filtered_elec_dist = elec_dist.loc[morph.section == elec_dist_var, "beta_passive"]
        filtered_spikes = [spikes[i] for i in np.where(morph.section == elec_dist_var)[0]]

        if len(filtered_spikes) < 10:
            raise RuntimeError(f"Found less than 10 spikes when computing quantiles for {elec_dist_var}.")
        
        q = np.quantile(filtered_elec_dist, np.arange(0, 1.1, 0.1))
        return q

    @staticmethod
    def bin_matrix_to_quantiles(matrix, quantiles, var_to_bin):
        out = np.zeros((len(quantiles), matrix.shape[1]))

        for i in range(len(quantiles) - 1):
            inds = np.where((var_to_bin > quantiles[i]) & ((var_to_bin < quantiles[i + 1])))[0]
            out[i] = np.sum(matrix[inds], axis = 0)
        
        return out[:-1]


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
    

class VoltageTrace(Trace):

    @staticmethod
    def get_Na_spikes(g_Na: np.ndarray, threshold: float, spikes: np.ndarray, ms_within_spike: float) -> np.ndarray:

        upward_crossings, _ = VoltageTrace.get_crossings(g_Na, threshold)

        if len(upward_crossings) == 0:
            return np.array([]), np.array([])

        Na_spikes = []
        backprop_AP = []
        for sp_time in upward_crossings:
            # Time of APs before this na spike
            spikes_before_sodium_spike = spikes[spikes < sp_time]

            # Na spike has no AP before
            if len(spikes_before_sodium_spike) == 0:
                Na_spikes.append(sp_time)

            # Na spike is more than x ms after last AP
            elif (sp_time - spikes_before_sodium_spike[-1] > ms_within_spike):
                Na_spikes.append(sp_time)

            # Na spike is within x ms after latest AP and counted as a back propagating AP
            else:
                backprop_AP.append(sp_time)

        return np.array(Na_spikes), np.array(backprop_AP)
    
    @staticmethod
    def get_Ca_spikes(v, threshold, ica):
        upward_crossings, downward_crossings = VoltageTrace.get_crossings(v, threshold)
        left_bounds, right_bounds, sum_currents = VoltageTrace.current_criterion(upward_crossings, downward_crossings, ica)
        return left_bounds, right_bounds, sum_currents
    
    @staticmethod
    def get_NMDA_spikes(v, threshold, inmda):
        return VoltageTrace.get_Ca_spikes(v, threshold, inmda)
    
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
                
                left_bounds.append(upward_crossings[crossing_index] + left_bound)
                right_bounds.append(upward_crossings[crossing_index] + right_bound)
                sum_current.append(
                    np.sum(control_current[int(upward_crossings[crossing_index] + left_bound) : int(upward_crossings[crossing_index] + right_bound)])
                    )

                x30 = x30[x30 > upward_crossings[crossing_index] + right_bound]
                if len(x30) == 0: stop = True

        return left_bounds, right_bounds, sum_current

            
            

            
            
