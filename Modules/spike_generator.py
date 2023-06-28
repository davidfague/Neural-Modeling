import numpy as np
from scipy.signal import lfilter
from neuron import h
import warnings
from Modules.cell_model import CellModel

def minmax(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

class SpikeGenerator:
  
	def __init__(self):
		self.spike_trains = []
		self.vecstims = []
		self.netcons = []
	
	#TODO: add docstring, check typing
	def generate_inputs(self, synapses: list, t: np.ndarray, mean_firing_rate: object, method: str, 
			 			origin: str, 
					  	rhythmicity: bool = False, 
						rhythmic_mod = None, rhythmic_f = None,
					  	spike_trains_to_delay = None, fr_time_shift = None, spike_train_dt: float = 1e-3) -> None:
		'''
		Generate spike trains.

		Parameters:
		----------
		synapses: list
			List of synapse objects.

		t: np.ndarray
			Time aray.

		mean_firing_rate: float or distribution
			Mean firing rate of the spike train.
		
		method: str
			How to vary the profile over time. One of ['1f_noise', 'delay'].

		mean_firing_rate:  of mean firing rate of spike train
		spike_trains_to_delay: list of time stamps where spikes occured for delay modulation

		origin: str
			..., one of ['same_presynaptic_cell', 'same_presynaptic_region']

		returns: temporary lists for this go around. Mainly for appending objects to cell
		'''
	
		spike_trains = []
		netcons_list = [] # returned temporary list for this go around

		#TODO: check the order of arguments in each function
		if origin == "same_presynaptic_cell": # same fr profile # same spike train # same mean fr
			# Ensure the firing rate is a float
			mean_fr = self.get_mean_fr(mean_firing_rate)
			fr_profile = self.get_firing_rate_profile(method=method, t = t,
						 							  rhythmicity = rhythmicity, rhythmic_f = rhythmic_f,
													  rhythmic_mod = rhythmic_mod, spike_trains_to_delay = spike_trains_to_delay,
													  fr_time_shift = fr_time_shift)
			spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
			for synapse in synapses:
				spike_trains.append(spikes)
				netcon = self.set_spike_train(synapse, spikes)
				netcons_list.append(netcon)
		  
		elif origin == "same_presynaptic_region": # same fr profile # unique spike train # unique mean fr
			fr_profile = self.get_firing_rate_profile(method=method, t = t,
						 							  rhythmicity = rhythmicity, rhythmic_f = rhythmic_f,
													  rhythmic_mod = rhythmic_mod, spike_trains_to_delay = spike_trains_to_delay,
													  fr_time_shift = fr_time_shift)
			for synapse in synapses:
				mean_fr = self.get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
				spike_trains.append(spikes)
				netcon = self.set_spike_train(synapse, spikes)
				netcons_list.append(netcon)
			
		else: # unique fr profile # unique spike train # unqiue mean fr
			for synapse in synapses:
				fr_profile = self.get_firing_rate_profile(method=method, t = t,
														  rhythmicity = rhythmicity, rhythmic_f = rhythmic_f,
														  rhythmic_mod = rhythmic_mod, spike_trains_to_delay = spike_trains_to_delay,
														  fr_time_shift = fr_time_shift)
				mean_fr = self.get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
				netcon = self.set_spike_train(synapse, spikes)
				spike_trains.append(spikes)
				netcons_list.append(netcon)
			
		return netcons_list, spike_trains
	#TODO: check definition of t
	#TODO: add docstring
	def get_firing_rate_profile(self, t, method: str, 
								rhythmicity: bool = False, rhythmic_f = None, rhythmic_mod = None,
								spike_trains_to_delay = None, fr_time_shift = None, spike_train_dt: float = 1e-3):

		# Create the firing rate profile
		#TODO: add bounds, etc.
		if method == '1f_noise':
			fr_profile = self.noise_modulation(num_obs = len(t))
		elif method == 'delay':
			fr_profile = self.delay_modulation(spike_trains_to_delay = spike_trains_to_delay, fr_time_shift = fr_time_shift, 
				      						  spike_train_t = t, spike_train_dt = spike_train_dt)
		else:
			raise NotImplementedError
		
		if rhythmicity:
			fr_profile = self.rhythmic_modulation(fr_profile, rhythmic_f, rhythmic_mod, t)
		
		#TODO: check if fr_profile can even be negative
		if np.sum(fr_profile < 0) != 0:
			warnings.warn("Found zeros in fr_profile.")

		fr_profile[fr_profile < 0] = 0 # Can't have negative firing rates.
		
		return fr_profile
	
	def noise_modulation(self, num_obs: int, A: list = None, B: list = None, bounds: tuple = (0.5, 1.5)) -> np.ndarray:
		'''
		Produce pink ("1/f") noise out of the white noise.
		The idea is to generate a white noise and then filter it to impose autocovariance structure with
		1/f psd. The filter used here is the scipy.signal's FIR / IIR filter which replaces observation t
		with an AR(k) sum of the previous k unfiltered AND filtered observations.

		The resulting pink noise is minmaxed and shifted to be in [shift[0], shift[1]] region.

		Parameters:
		----------
		num_obs: int
			Length of the profile.

		A: list[float]
			AR coefficients of the filtered observations.

		B: list[float]
			AR coefficients of the unfiltered (original) observations.

		bounds: tuplew
			The profile bounds: [min, max]. 

		Returns:
		----------
		fr_profile: np.ndarray
			Firing rate profile.
		'''
		# Default values from a previous implementation.
		if A is None:
			A = [1, -2.494956002, 2.017265875, -0.522189400]
		if B is None:
			B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
		
		white_noise = np.random.normal(loc = 1, scale = 0.5, size = num_obs + 2000)

		# Apply the FIR/IIR filter to create the 1/f noise, minmax and shift to bounds
		fr_profile = minmax(lfilter(B, A, white_noise)[2000:]) * (bounds[1] - bounds[0]) + bounds[0]

		return fr_profile
	
	#TODO: fix typing, potentially won't work with spike_trains_to_delay.shape[0] != 1
	def delay_modulation(self, spike_trains_to_delay, fr_time_shift: int, spike_train_t: int, 
			  			 spike_train_dt: float = 1e-3, bounds = (0, 2)) -> np.ndarray:
		'''
		Compute a firing rate profile from a target spike train and shift it to create a new profile.

		Parameters:
		----------
		spike_trains_to_delay: #TODO: add type
			Target spike train.

		fr_time_shift: int
			Shift of the target spike train's rate profile.

		spike_train_t: int
			Time length that was used to generate spike_trains_to_delay.

		spike_train_dt: float
			Time discretization that was used to generate spike_trains_to_delay.

		bounds: tuple
			The profile bounds: [min, max]. 

		Returns:
		----------
		fr_profile: np.ndarray
			Firing rate profile.

		'''
		if fr_time_shift is None:
			raise TypeError('fr_time_shift must be an integer.')
		
		# Flatten all the spike trains, because otherwise np.histogram doesn't work
		# Ignore spike trains without spikes
		times_where_spikes = [sp_time for sp_train in spike_trains_to_delay for sp_time in sp_train if len(sp_train) > 0]
				
		# Compute the firing rate profile from the histogram
		hist, _ = np.histogram(times_where_spikes, bins = spike_train_t)

		#TODO: check if redundant, and can just use hist
		fr_profile = hist / (spike_train_dt * (len(spike_trains_to_delay) + 1))

		# Shift by fr_time_shift
		wrap = fr_profile[-fr_time_shift:]
		fr_profile[fr_time_shift:] = fr_profile[0:-fr_time_shift]
		fr_profile[0:fr_time_shift] = wrap

		# Minmax and shift to bounds
		fr_profile = minmax(fr_profile) * (bounds[1] - bounds[0]) + bounds[0]

		return fr_profile

	#TODO: why this equation (source?)
	def rhythmic_modulation(self, fr_profile: np.ndarray, rhythmic_f: int, P: int, rhythmic_mod: float, t: np.ndarray):
		'''
		
		Parameters:
		----------
		fr_profile: np.ndarray
			Firing rate profile to modulate.

		rhythmic_f: int
			The multiplicative period of the sin function, k in a * sin(2 * pi * k + b).

		P: int
			The additive period of the sin function, b in a * sin(2 * pi * k + b).

		rhythmic_mod: float
			Modulation strength, a in a * sin(2 * pi * k + b).

		t: np.ndarray
			Time used to generate fr_profile.

		Returns:
		----------
		fr_profile: np.ndarray
			Modulated profile.

		'''
		A = fr_profile / (1 / rhythmic_mod - 1)
		fr_profile[0, :] = A * np.sin(2 * np.pi * rhythmic_f * t + P) + fr_profile

		return fr_profile
	
	#TODO: fix call
	#TODO: check division by 1000
	def generate_spikes_from_profile(self, fr_profile, mean_fr):
		''' sample spikes '''
		fr_profile = fr_profile * mean_fr
		sample_values = np.random.poisson(fr_profile / 1000)
		spike_times = np.where(sample_values > 0)[0]
		return spike_times
	
	def set_spike_train(self, synapse, spikes) -> h.NetCon:
		self.spike_trains.append(spikes)
		stim = self.set_vecstim(spikes)
		nc = self.set_netcon(synapse, stim)
		return nc
	
	def set_vecstim(self, stim_spikes) -> h.Vector:
		vec = h.Vector(stim_spikes)
		stim = h.VecStim()
		stim.play(vec)
		self.vecstims.append(stim)
		return stim
	
	def set_netcon(self, synapse, stim) -> h.NetCon:
		nc = h.NetCon(stim, synapse.synapse_neuron_obj, 1, 0, 1)
		self.netcons.append(nc)
		synapse.ncs.append(nc)
		return nc
	
	def get_mean_fr(self, mean_firing_rate: object) -> float:
		if callable(mean_firing_rate): # mean_firing_rate is a distribution
			 # Sample from the distribution
			mean_fr = mean_firing_rate(size = 1)
		else: # mean_firing_rate is a float
			mean_fr = mean_firing_rate

		if mean_fr <= 0:
			raise ValueError("mean_fr <= 0.")
		
		return mean_fr

	@staticmethod
	def remove_inputs(synapses = None, netcons = None) -> None:
		''' 
		Makes netcons inactive (do not deliver their spike trains).
		
		Parameters:
		----------
		synapses: list
			List of synapses to deactivatie netcons for.

		netcons: list
			List of netcons to deactivate.
		'''
		if synapses:
			for synapse in synapses:
				for netcon in synapse.ncs:
					netcon.active(False)
		if netcons:
			for netcon in netcons:
				netcon.active(False)
	def generate_inputs_to_cell(self, cell: CellModel, synapses: list, t: np.ndarray, mean_firing_rate: object, method: str, 
				 			origin: str, 
						  	rhythmicity: bool = False, 
							rhythmic_mod = None, rhythmic_f = None,
						  	spike_trains_to_delay = None, fr_time_shift = None, spike_train_dt: float = 1e-3) -> None:
		'''
		Generate spike trains on an existing cell object

		Parameters:
		----------
  		cell: CellModel
			Cell to add to.
   
		synapses: list
			List of synapse objects.

		t: np.ndarray
			Time aray.

		mean_firing_rate: float or distribution
			Mean firing rate of the spike train.
		
		method: str
			How to vary the profile over time. One of ['1f_noise', 'delay'].

		mean_firing_rate:  of mean firing rate of spike train
		spike_trains_to_delay: list of time stamps where spikes occured for delay modulation

		origin: str
			..., one of ['same_presynaptic_cell', 'same_presynaptic_region']

		returns: temporary lists for this go around. Mainly for appending objects to cell
		'''
		
		netcons, spike_trains = self.generate_inputs(synapses=synapses, t=t, mean_firing_rate=mean_firing_rate, method=method, 
			 			origin=origin, 
					  	rhythmicity=rhythmicity, 
						rhythmic_mod = rhythmic_mod, rhythmic_f = rhythmic_f,
					  	spike_trains_to_delay = spike_trains_to_delay, fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
		for netcon in netcons:
			cell.netcons.append(netcon)
		for spike_train in spike_trains:
			cell.spike_trains.append(spike_trains)
