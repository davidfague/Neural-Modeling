import numpy as np
import pandas as pd
import os, h5py
import warnings

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
        # initialize attributes for functional groups and presynaptic cells
        self.functional_group_indices = []
        self.functional_group_names = []
        self.presynaptic_cell_indices = []
        self.presynaptic_cell_names = []

class SegmentManager:

    def __init__(self, output_folder: str, steps: list, dt: float = 0.1, skip: int = 0, transpose = False, build_detailed_seg_info: bool = False, no_data=False):
        '''
        skip: ms of simulation to skip
        build_detailed_seg_info: Whether or not to try to read detailed_seg_info.csv, a csv containing much more segmentation
        '''
        if no_data:
          filenames = ["Vm_report", "spikes_report"]
          current_names = ["v"]
        else:
          #filenames = ["Vm_report", "gNaTa_t_NaTa_t_data_report", "i_AMPA_report",
          #             "i_NMDA_report", "i_GABA_report", "ica_Ca_HVA_data_report", "ica_Ca_LVAst_data_report",
          #             "ihcn_Ih_data_report", "ina_NaTa_t_data_report", "i_membrane_report","spikes_report"]
          #current_names = ["v", "gNaTa", "iampa", "inmda", "igaba","icah", "ical", "ih", "ina", "imembrane"]
          #filenames = ["Vm_report", "ina_nax_data_report", "i_AMPA_report",
          #             "i_NMDA_report", "i_GABA_report", "ik_kap_data_report", "ik_kdmc_data_report", "ik_kdr_data_report",
          #             "ihcn_Ih_data_report", "i_pas_data_report","spikes_report"]
          filenames = ["Vm_report", "ina_nax_data_report", "i_AMPA_report",
                       "i_NMDA_report", "i_GABA_report", "ik_kap_data_report", "ik_kdmc_data_report", "ik_kdr_data_report",
                       "i_hd_data_report", "i_pas_data_report","spikes_report"]
          current_names = ["v", "ina_nax", "iampa", "inmda", "igaba", "ik_kap","ik_kdmc","ik_kdr", "ih", "i_pas"]
          
        self.segments = []
        self.dt = dt

        # Read datafiles
        data = self.read_data(filenames, output_folder, steps)
        
        #debugging
        print(f" len(data.keys()): {len(data.keys())}")
        print(f" len(filenames): {len(filenames)}")
        print(f" len(data[filenames[0]]): {len(data[filenames[0]])}")
        for key in data.keys():
          print(f" len(data[{key}]): {len(data[key])}")
#        if build_detailed_seg_info: # not yet implemented
#          build_detailed_segments(output_folder, steps)
        #print(data["Vm_report"].shape)

        self.num_segments = len(data["seg_info"])
        print("NUM seg", self.num_segments)

        for i in range(self.num_segments):
            # Build seg_data
            seg_data = {}
            for filename, current_name in zip(filenames[:-1], current_names):
                if transpose:
                    seg_data[current_name] = data[filename].T[i, int(skip / dt):]
                else:
                    seg_data[current_name] = data[filename][i, int(skip / dt):]

            seg = Segment(seg_info = data["seg_info"].iloc[i], seg_data = seg_data)
            self.segments.append(seg)
            
        self.compute_adj_segs_from_pseg_indices()
        
        # Soma spikes (ms)
        self.soma_spiketimes = np.array([(i-skip) for i in data[filenames[-1]][:] if i >= skip])
        # Soma spikes (inds)
        self.soma_spiketimestamps = np.sort((self.soma_spiketimes / dt).astype(int))

    def compute_adj_segs_from_pseg_indices(self):
      '''
      Uses parent segment indices to assign lists of adjacent segments to each segment
      '''
      for i, seg in enumerate(self.segments):  # iterate through segment index
        psegid = seg.pseg_index
        if not np.isnan(psegid):
            psegid = int(psegid)
            self.segments[psegid].adj_segs.append(seg)  # add child seg to this seg's adj_segs list
            self.segments[psegid].child_segs.append(seg)  # add child seg to this seg's child_segs list
            seg.parent_segs.append(self.segments[psegid])
            seg.adj_segs.append(self.segments[psegid])

    def read_data(self, filenames: list, output_folder: str, steps: list) -> list:

        data = {name : [] for name in filenames}
        
        for step in steps:
            dirname = os.path.join(output_folder, f"saved_at_step_{step}")
            for name in filenames:
              with h5py.File(os.path.join(dirname, name + ".h5"), 'r') as file:
                if name == 'spikes_report':
                    if "spikes" in file and "biophysical" in file["spikes"]:
                        data[name].append(np.array(file["spikes"]["biophysical"]["timestamps"][:]))
                    elif "report" in file and "biophysical" in file["report"]:
                        data[name].append(np.array(file["report"]["biophysical"]["data"]))
                    else:
                        print(f"No expected key found in file {name}.h5 for spikes_report")
                else:
                    if "report" in file and "biophysical" in file["report"]:
                        data[name].append(np.array(file["report"]["biophysical"]["data"][:]))
                    else:
                        print(f"No expected key found in file {name}.h5 for report")

        # Merge data
        for name in data.keys():
            data[name] = np.hstack(data[name])

        data["seg_info"] = pd.read_csv(os.path.join(output_folder, f"saved_at_step_{steps[0]}", "seg_info.csv"))

        return data

    def get_na_lower_bounds_and_peaks(self, threshold: float, ms_within_somatic_spike: float) -> tuple:
        na_lower_bounds = []
        bAP_lower_bounds = []
        peak_values = []
        flattened_peak_values = []
    
        for seg in self.segments:
            lb, bAPs = self.get_na_lower_bounds_for_seg(seg, threshold, ms_within_somatic_spike)
            na_lower_bounds.append(lb)
            bAP_lower_bounds.append(bAPs)
    
            # Calculate the index for the peak
            peak_indices = (lb + 1 / self.dt).astype(int)
    
            # Iterate over each index in peak_indices
            failed_indices = []
            for index in peak_indices:
                # Check if the index is within the bounds of the data
                if index < len(seg.gNaTa):
                    peak = seg.gNaTa[index]
                    peak_values.append(peak)
                    flattened_peak_values.append(peak)
                else:
                    failed_indices.append(index)

            if len(failed_indices) > 0:
                warining_text = f"Skipped {len(failed_indices)} indicies, since they exceeded the data length."
                warnings.warn(warining_text)
    
        return na_lower_bounds, peak_values, flattened_peak_values, bAP_lower_bounds

    def get_na_lower_bounds_for_seg(self, seg, threshold: float, ms_within_somatic_spike: float) -> np.ndarray:
        # Find bounds (crossings)
        threshold_crossings = np.diff(seg.gNaTa > threshold)

        # Determine if the trace starts above or below threshold to get upward crossings
        if seg.gNaTa[0] < threshold:
            upward_crossings = np.argwhere(threshold_crossings)[::2]
        else:
            upward_crossings = np.argwhere(threshold_crossings)[1::2]

        # Only count if within 2 ms after a somatic spike
        # na_spks = [int(i) for i in upward_crossings if ~np.any((np.abs(i - self.soma_spiketimestamps) < ms_within_somatic_spike / self.dt))]
        na_spks = []
        bAPs = [] # list of na spikes classified as action potentials
        for na_spikestamp in upward_crossings: # filter out na spikes right after 
            soma_spikes_before_na_spike = self.soma_spiketimestamps[self.soma_spiketimestamps < na_spikestamp] # time of APs before this na spike
            if len(soma_spikes_before_na_spike) == 0: # na spike has no AP before
                na_spks.append(na_spikestamp)
            elif (na_spikestamp - soma_spikes_before_na_spike[-1] > ms_within_somatic_spike / self.dt): # na spike is more than x ms after last AP
                na_spks.append(na_spikestamp)
            else: # na spike is within x ms after latest AP # na spike is less than x ms after latest AP and counted as a back propagating AP
                bAPs.append(na_spikestamp)

        return np.array(na_spks), np.array(bAPs)

    def get_ca_nmda_lower_bounds_durations_and_peaks(self, lowery, uppery, random_state: np.random.RandomState):
        # Filter and get bounds
        segIDs, lower_bounds, upper_bounds, mag, segments_for_condition = [], [], [], [], []
        for i, seg in enumerate(self.segments):
            current_type = None
            if (lowery is not None) & (uppery is not None):
                cond = (seg.type == "apic") & (seg.p0_5_y3d > lowery) & (seg.p0_5_y3d < uppery)
                current_type = "icah"
            else:
                cond = (seg.type == "dend") | (seg.type == "apic")
                current_type = "inmda"

            if cond:
                segIDs.append(i)
                bounds = self.get_ca_lower_bounds_for_seg(seg, i, current_type)
                lower_bounds.append(bounds[0]), upper_bounds.append(bounds[1]), mag.append(np.array(bounds[2]))

                # Prepare for the next step, peak calculation
                for bound in bounds[0]:
                    if (bound > 20) & (bound < 1400000): #TODO: ca_upper_bounds??
                        segments_for_condition.append(i) # (i) will be appended multiple times, that's intended
            else:
                lower_bounds.append([]), upper_bounds.append([]), mag.append([])

        random_segments_ids = random_state.choice(segments_for_condition, 100)

        # Not used in notebooks @DEPRECATION
        duration_low, duration_high, peak_values = [], [], []
        if current_type == 'icah':
            for rand_seg_id in random_segments_ids:
                spike_times = lower_bounds[rand_seg_id]
                duration_low_seg, duration_high_seg, peak_values_seg = self.get_duration_and_peak_for_seg(self.segments[rand_seg_id], spike_times)
                duration_low.append(duration_low_seg), duration_high.append(duration_high_seg), peak_values.append(peak_values_seg)

        return lower_bounds, upper_bounds, mag, duration_low, duration_high, peak_values

    def get_duration_and_peak_for_seg(self, seg, spike_times):
        duration_low, duration_high, peak_values = [], [], []
        for spike_time in spike_times:
            if spike_time > 100:
                trace = seg.icah[spike_time - 100 : spike_time + 200] + seg.ical[spike_time - 100 : spike_time + 200] +\
                seg.ih[spike_time - 100 : spike_time + 200]
                peak_value = np.max(trace)
                half_peak = peak_value / 2
                duration = np.arange(len(trace))[trace > half_peak] + spike_time - 10
                if len(duration) > 2:
                    duration_low.append(duration[0])
                    duration_high.append(duration[-1])
                    peak_values.append(peak_value)
        return duration_low, duration_high, peak_values

    def get_ca_lower_bounds_for_seg(self, seg, seg_ind, current_type):

        ca_lower_bound, ca_upper_bound, ca_mag = [], [], []
        
        if current_type == "icah":
            v_thresh, time_thresh = -40, 200
            trace = seg.icah + seg.ical + seg.ih
        elif current_type == "inmda":
            v_thresh, time_thresh = -40, 260
            trace = seg.inmda
        else:
            raise ValueError("current_type not defined")

        legit_uc_iso = voltage_criterion(seg.v, v_thresh = v_thresh, time_thresh = time_thresh)[1]
        legit_dc_iso = voltage_criterion(seg.v, v_thresh = v_thresh, time_thresh = time_thresh)[-1]

        if (len(legit_uc_iso) != 0) & (np.min(trace) != 0):
            bnds, sum_curr = current_criterion(legit_uc_iso = legit_uc_iso, legit_dc_iso = legit_dc_iso,
                                               control_inmda = getattr(seg, current_type))
            ca_lower_bound = np.array(bnds).reshape(-1, 2)[:, 0]
            ca_upper_bound = np.array(bnds).reshape(-1, 2)[:, 1]
            ca_mag = sum_curr

        return ca_lower_bound, ca_upper_bound, ca_mag

    def get_edges(self, lower_bounds, edge_type = "apic", elec_dist_var = 'soma_passive', mag = None, mag_th = None):
        edges = []
        for i in range(self.num_segments):
            if (edge_type in self.segments[i].sec) & (len(lower_bounds[i]) > 0):
                if mag is None:
                    edges.append(eval(self.segments[i].seg_elec_distance)['beta'][elec_dist_var])
                else:
                    if np.any(mag[i] < mag_th):
                        edges.append(eval(self.segments[i].seg_elec_distance)['beta'][elec_dist_var])

        if len(edges) > 10:
            edges = np.quantile(edges, np.arange(0, 1.1, 0.1))
        else:
            raise(RuntimeError(f"len({edges}) < 10 for edge_type:{edge_type}| elec_dist_var:{elec_dist_var}. This is probably due to a lack of reported dendritic spikes in lower_bounds[seg_index]."))

        return edges

    def get_sta(self, spiketimes, lower_bounds, edges, sec_indicator, current_type, elec_dist_var = 'soma_passive', mag = None, mag_th = None):
        
        bin_start, bin_end, step_size, interval = 5, 5, 2, 49

        sta = np.zeros((len(edges), interval))
        c = 0
        for i in range(self.num_segments):
            for s_times in np.sort(spiketimes):
                # Exclude bursts
                if s_times - c > 10:
                    for e in np.arange(0, len(edges)-1):
                        if len(lower_bounds[i]) > 0:
                            dist = eval(self.segments[i].seg_elec_distance)['beta'][elec_dist_var]
                            if (sec_indicator in self.segments[i].sec):
                                if edges[e] < dist <= edges[e + 1]:
                                    na_inds = lower_bounds[i].astype(int)
                                    if current_type == "inmda": na_inds = na_inds[mag[i] < mag_th]
                                    x2, _ = np.histogram(na_inds * self.dt, bins = np.arange(np.floor(s_times) - bin_start / self.dt, np.floor(s_times) + bin_end / self.dt, step_size))
                                    sta[e] += x2
                c = s_times

        return sta[:-1, :]
        
    def sum_currents(self, currents: list, var_name: str):
      '''
      function for summing a list of currents, as new attribute of Segment class
      '''
      for seg in self.segments:
        i = 0
        for current in currents:
          i += getattr(seg, current)
        setattr(seg, var_name, i)
      
    def compute_axial_currents(self):
      '''
      function for computing axial currents as new attribute of Segment class
      '''
      #compute axial currents between all segments
      for seg in self.segments:
        for adj_seg in seg.adj_segs:
          axc = (seg.v - adj_seg.v) / (seg.seg_half_seg_RA + adj_seg.seg_half_seg_RA) #compute axial current using (v_in-v_out)/(halfsegRa+halfsegRa)
          seg.axial_currents.append(axc)
          if adj_seg in seg.parent_segs:
            seg.parent_axial_currents.append(axc)
          elif adj_seg in seg.child_segs:
            seg.child_axial_currents.append(axc)
    
    def build_detailed_segments(self):
      pass