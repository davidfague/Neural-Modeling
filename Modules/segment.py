import numpy as np
import pandas as pd
import h5py

def voltage_criterion(data = None, v_thresh: float = -40, time_thresh: int = 260):
    threshold_crossings = np.diff(data > v_thresh, prepend = False)
    upward_crossings = np.argwhere(threshold_crossings)[::2, 0]
    downward_crossings = np.argwhere(threshold_crossings)[1::2, 0]
    # If length of threshold_crossings is not even
    if np.mod(np.argwhere(threshold_crossings).reshape(-1,).shape[0],2)!= 0:
        legit_up_crossings = upward_crossings[:-1][np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2] > time_thresh]
        legit_down_crossings = downward_crossings[np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2] > time_thresh]
    else:
        legit_up_crossings = upward_crossings[np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2] > time_thresh]
        legit_down_crossings = downward_crossings[np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2] > time_thresh]
    return upward_crossings, legit_up_crossings, legit_down_crossings

def current_criterion(legit_uc_iso, legit_dc_iso, control_inmda = np.array([1])):
    bounds = []
    sum_current = []

    for ind1 in np.arange(0, len(legit_uc_iso)):
        e1 = control_inmda[legit_uc_iso[ind1]] # Current @ up_crossing[ind1]
        # All the indices where current crosses 130% of e1
        x30 = np.argwhere(np.diff(control_inmda[legit_uc_iso[ind1]:legit_dc_iso[ind1]] < 1.3*e1, prepend=False))
        # All the indices where current crosses 115% of e1
        x15 = np.argwhere(np.diff(control_inmda[legit_uc_iso[ind1]:legit_dc_iso[ind1]] < 1.15*e1, prepend=False))

        if len(x30)>0:
            x15_copy = x15
            x30_copy = x30
            try:
                i = x30[0][0]
            except:
                import pdb; pdb.set_trace()
            n = 0
            while n==0:
                if len(np.sort(x15[x15 > i]))!=0:
                    b1 = i
                    b2 = np.sort(x15[x15 > i])[0]
                    bounds.append([legit_uc_iso[ind1]+b1,legit_uc_iso[ind1] + b2])
                    sum_current.append(np.sum(control_inmda[legit_uc_iso[ind1]+b1:legit_uc_iso[ind1] + b2]) / 10)
                else:
                    b1 = i
                    b2 = (legit_dc_iso[ind1] - legit_uc_iso[ind1])
                    bounds.append([legit_uc_iso[ind1]+b1,legit_uc_iso[ind1] + b2])
                    sum_current.append(np.sum(control_inmda[legit_uc_iso[ind1] + b1:legit_uc_iso[ind1] + b2]) / 10)
                    n=1

                x30_copy = x30_copy[x30_copy>legit_uc_iso[ind1]+b2]

                if len(x30_copy)!=0:
                    i = x30_copy[x30_copy>b2][0]
                else:
                    n = 1

    return bounds, sum_current

class Segment:
    '''
    class for storing segment info and recorded data
    '''
    def __init__(self, seg_info: dict = None, seg_data: dict = None):
        if seg_info is None or seg_data is None:
            raise ValueError("seg_info and seg_data cannot be None.")

        # seg_info is a row from a dataframe
        # set seg_info into Segment attributes
        for info_key, info_value in seg_info.items():
            clean_key = str(info_key).replace(" ", "_").replace(".", "_")  # change space and '.' both to '_'
            setattr(self, clean_key, info_value)

        # Assign the segment name
        self.name = self.seg  # for clarity

        # set seg_data into Segment attributes
        for data_type in seg_data:
            setattr(self, str(data_type), seg_data[data_type])

        # set segment color based on the type
        if self.type == 'soma':
            self.color = 'purple'
        elif self.type == 'dend':
            self.color = 'red'
        elif self.type == 'apic':
            self.color = 'blue'
        elif self.type == 'axon':
            self.color = 'green'
        else:
            raise ValueError("Section type not implemented", self.type)

        # initialize lists for later
        self.axial_currents = []
        self.adj_segs = []  # adjacent segments list
        self.child_segs = []
        self.parent_segs = []
        self.parent_axial_currents = []
        self.child_axial_currents = []

class SegmentManager:

    def __init__(self, output_folder: str, dt: float = 0.1):

        self.segments = []
        self.dt = dt

        # Read datafiles
        # The last element of data is always seg_info.csv
        # The second-to-last element of data is always spikes_report.h5
        data = self.read_data(output_folder)

        self.num_segments = len(data[-1])

        for i in range(self.num_segments):
            # Build seg_data
            seg_data = {}
            for ind, name in enumerate(["v", "gNaTa", "iampa", "icah", "ical", "ih", "ina"]):
                seg_data[name] = data[ind][i, :]

            seg = Segment(seg_info = data[-1].iloc[i], seg_data = seg_data)
            self.segments.append(seg)

        # Soma spikes (ms)
        self.soma_spiketimes = data[-2][:]
        # Soma spikes (inds)
        self.soma_spiketimestamps = np.sort((self.soma_spiketimes / dt).astype(int))

    def read_data(self, output_folder: str) -> list:
        file_names = ["Vm_report", "gNaTa_T_data_report", "i_AMPA_report",
                      "i_NMDA_report", "icah_data_report", "ical_data_report",
                      "ih_data_report", "ina_data_report", "spikes_report"]
        ext = 'h5'
        data = []

        for name in file_names:
            with h5py.File(f"{output_folder}/{name}.{ext}", 'r') as file:
                data.append(np.array(file["report"]["biophysical"]["data"]))

        data.append(pd.read_csv(f"{output_folder}/seg_info.csv"))

        return data

    def get_na_lower_bounds_and_peaks(self, threshold: float, ms_within_somatic_spike: float) -> tuple:
        na_lower_bounds = []
        peak_values = []
        flattened_peak_values = []
    
        for seg in self.segments:
            lb = self.get_na_lower_bounds_for_seg(seg, threshold, ms_within_somatic_spike)
            na_lower_bounds.append(lb)
    
            # Calculate the index for the peak
            peak_indices = (lb + 1 / self.dt).astype(int)
    
            # Iterate over each index in peak_indices
            for index in peak_indices:
                # Check if the index is within the bounds of the data
                if index < len(seg.gNaTa):
                    peak = seg.gNaTa[index]
                    peak_values.append(peak)
                    flattened_peak_values.append(peak)
                else:
                    print(f"Warning: peak index {index} exceeds the size of the data. Skipping this index.")
    
        return na_lower_bounds, peak_values, flattened_peak_values

    def get_na_lower_bounds_for_seg(self, seg, threshold: float, ms_within_somatic_spike: float) -> np.ndarray:
        # Find bounds (crossings)
        threshold_crossings = np.diff(seg.gNaTa > threshold)

        # Assume the trace starts below threshold
        upward_crossings = np.argwhere(threshold_crossings)[::2]

        # Only count if within 2 ms after a somatic spike
        na_spks = [int(i) for i in upward_crossings if ~np.any((i - self.soma_spiketimestamps < ms_within_somatic_spike / self.dt))]
        if len(threshold_crossings) % 2 != 0:
            na_spks = na_spks[:-1]

        return np.array(na_spks)

    def get_ca_lower_bounds_durations_and_peaks(self, lowery, uppery, random_state: np.random.RandomState):
        # Filter Ca and get bounds
        CAsegIDs, ca_lower_bounds, ca_upper_bounds, ca_mag, ca_segments_for_condition = [], [], [], [], []
        for i, seg in enumerate(self.segments):
            #if (seg.type == "apic") & (seg.p0_5_y3d > lowery) & (seg.p0_5_y3d < uppery):
            if (seg.type == "dend") | (seg.type == "apic"):
                CAsegIDs.append(i)
                bounds = self.get_ca_lower_bounds_for_seg(seg, i)
                ca_lower_bounds.append(bounds[0]), ca_upper_bounds.append(bounds[1]), ca_mag.append(bounds[2])

                # Prepare for the next step, peak calculation
                for bound in bounds[0]:
                    if (bound > 20) & (bound < 1400000): #TODO: ca_upper_bounds??
                        ca_segments_for_condition.append(i) # (i) will be appended multiple times, that's intended
            else:
                ca_lower_bounds.append([]), ca_upper_bounds.append([]), ca_mag.append([])

        random_segments_ids = random_state.choice(ca_segments_for_condition, 100)

        duration_low, duration_high, peak_values = [], [], []
        for rand_seg_id in random_segments_ids:
            spike_times = ca_lower_bounds[rand_seg_id]
            duration_low_seg, duration_high_seg, peak_values_seg = self.get_duration_and_peak_for_seg(self.segments[rand_seg_id], spike_times)
            duration_low.append(duration_low_seg), duration_high.append(duration_high_seg), peak_values.append(peak_values_seg)

        return ca_lower_bounds, ca_upper_bounds, ca_mag, duration_low, duration_high, peak_values

    def get_duration_and_peak_for_seg(self, seg, spike_times):
        duration_low, duration_high, peak_values = [], [], []
        for spike_time in spike_times:
            if spike_time > 100:
                trace = seg.icah[spike_time - 100 : spike_time + 200] + seg.ical[spike_time - 100 : spike_time + 200] +\
                seg.ih[spike_time - 100 : spike_time + 200]
                peak_value = np.max(trace)
                half_peak = peak_value / 2
                duration = np.arange(0, 300)[trace > half_peak] + spike_time - 10
                if len(duration) > 2:
                    duration_low.append(duration[0])
                    duration_high.append(duration[-1])
                    peak_values.append(peak_value)
        return duration_low, duration_high, peak_values

    def get_ca_lower_bounds_for_seg(self, seg, seg_ind):

        ca_lower_bound, ca_upper_bound, ca_mag = [], [], []

        trace = seg.icah + seg.ical + seg.ih
        m, s = np.mean(trace), np.std(trace)

        legit_uc_iso = voltage_criterion(seg.v, v_thresh = -40, time_thresh = 200)[1]
        legit_dc_iso = voltage_criterion(seg.v, v_thresh = -40, time_thresh = 200)[-1]

        if (len(legit_uc_iso) != 0) & (np.min(trace) != 0):
            bnds, sum_curr = current_criterion(legit_uc_iso = legit_uc_iso, legit_dc_iso = legit_dc_iso,
                                               control_inmda = seg.icah)
            ca_lower_bound = np.array(bnds).reshape(-1, 2)[:, 0]
            ca_upper_bound = np.array(bnds).reshape(-1, 2)[:, 1]
            ca_mag = sum_curr

        return ca_lower_bound, ca_upper_bound, ca_mag

    def get_edges(self, lower_bounds, edge_type = "apic"):
        edges = []
        for i in range(self.num_segments):
            if (len(lower_bounds[i]) > 0) & (edge_type in self.segments[i].sec):
                edges.append(eval(self.segments[i].seg_elec_distance)['beta']['passive_soma'])

        if len(edges) > 10:
            edges = np.quantile(edges, np.arange(0, 1.1, 0.1))

        return edges

    def get_sta(self, spiketimes, lower_bounds, edges, sec_indicator):
        sta = np.zeros((len(spiketimes), 39))

        c = 0
        for i in range(self.num_segments):
            for s_times in np.sort(spiketimes):
                # Exclude bursts
                if s_times - c > 10:
                    for e in np.arange(0, len(edges)):
                        if len(lower_bounds[i]) > 0:
                            dist = eval(self.segments[i].seg_elec_distance)['beta']['passive_soma']
                            if (sec_indicator in self.segments[i].sec):
                                if edges[e] < dist <= edges[e + 1]:
                                    na_inds = lower_bounds[i].astype(int)
                                    x2, _ = np.histogram(na_inds * self.dt, bins = np.arange(np.floor(s_times) - 2 / self.dt, np.floor(s_times) + 2 / self.dt, 1))
                                    sta[e] += x2
                c = s_times

        return sta
