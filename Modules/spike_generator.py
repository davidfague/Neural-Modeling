import numpy as np
from scipy.signal import lfilter
from neuron import h

def minmax(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

class SpikeGenerator:
  
	def __init__(self):
		self.netcons = []
		self.spike_trains = []
	
	#TODO: add docstring, check typing
	def generate_inputs(self, synapses: list, t: np.ndarray, mean_firing_rate: object, method: str, 
		     			origin: str, 
					  	rhythmicity: bool = False, 
						rhythmic_mod = None, rhythmic_f = None,
					  	spike_trains_to_delay = None, time_shift = None) -> None:
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
		'''
	
		spike_trains = []
		netcons_list = []

		#TODO: check the order of arguments in each function
		if origin == "same_presynaptic_cell": # same fr profile # same spike train # same mean fr
			# Ensure the firing rate is a float
			mean_fr = self.get_mean_fr(mean_firing_rate)
			fr_profile = self.get_firing_rate_profile(method, t, rhythmicity, spike_trains_to_delay)
			spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
			for synapse in synapses:
				netcons_list.append(self.set_spike_train(synapse, spikes))
				spike_trains.append(spikes)
		  
		elif origin == "same_presynaptic_region": # same fr profile # unique spike train # unique mean fr
			fr_profile = self.firing_rate_profile(method, t, rhythmicity, spike_trains_to_delay)
			for synapse in synapses:
				mean_fr = self.get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
				netcons_list.append(self.set_spike_train(synapse, spikes))
				spike_trains.append(spikes)
			
		else: # unique fr profile # unique spike train # unqiue mean fr
			for synapse in synapses:
				fr_profile = self.firing_rate_profile(method, t, rhythmicity, spike_trains_to_delay)
				mean_fr = self.get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr)
				netcons_list.append(self.set_spike_train(synapse, spikes))
				spike_trains.append(spikes)
			
		self.netcons.append(netcons_list)
		self.spike_trains.append(spike_trains)
	
	#TODO: check definition of t
	#TODO: add docstring
	def get_firing_rate_profile(self, t, mean_firing_rate: float, method: str, 
								rhythmicity: bool = False, rhythmic_f = None, rhythmic_mod = None,
								spike_trains_to_delay = None, time_shift = None):

		# Create the firing rate profile
		#TODO: add bounds, etc.
		if method == '1f_noise':
			fr_profile = self.noise_modulation(num_obs = len(t))
		elif method == 'delay':
			fr_profile = self.delay_modulation(fr_profile, spike_trains_to_delay, time_shift, t)
		else:
			raise NotImplementedError
		
		if rhythmicity:
			fr_profile = self.rhythmic_modulation(fr_profile, rhythmic_f, rhythmic_mod, t)
		
		#TODO: check if fr_profile can even be negative
		fr_profile[fr_profile < 0] = 0 # Can't have negative firing rates.
		
		return fr_profile
	
	def noise_modulation(self, num_obs: int, A: list = None, B: list = None, bounds: tuple = (0, 2)) -> np.ndarray:
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
	
	#TODO: fix typing, potentially won't work with spike_train_to_delay.shape[0] != 1
	def delay_modulation(self, spike_train_to_delay, fr_time_shift: int, spike_train_t: int, 
		      			 spike_train_dt: float = 1e-3, bounds = (0, 2)) -> np.ndarray:
		'''
		Compute a firing rate profile from a target spike train and shift it to create a new profile.

		Parameters:
		----------
		spike_train_to_delay: #TODO: add type
			Target spike train.

		fr_time_shift: int
			Shift of the target spike train's rate profile.

		spike_train_t: int
			Time length that was used to generate spike_train_to_delay.

		spike_train_dt: float
			Time discretization that was used to generate spike_train_to_delay.

		bounds: tuple
			The profile bounds: [min, max]. 

		Returns:
		----------
		fr_profile: np.ndarray
			Firing rate profile.

		'''

		# Compute the firing rate profile from the histogram
		hist, _ = np.histogram(spike_train_to_delay, bins = spike_train_t)

		#TODO: check if redundant, and can just use hist
		fr_profile = hist / (spike_train_dt * (len(spike_train_to_delay) + 1))

		# Shift by time_shift
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
	def generate_spikes_from_profile(self, fr_profile):
		''' sample spikes '''
		sample_values = np.random.poisson(fr_profile / 1000)
		spike_times = np.where(sample_values > 0)[0]
		return spike_times
	
	def set_spike_train(self, synapse, spikes):
		stim = self.set_vecstim(spikes)
		nc = self.set_netcon(stim, synapse)
		return nc
	
	def set_vecstim(self, stim_spikes):
		vec = h.Vector(stim_spikes)
		stim = h.VecStim()
		stim.play(vec)
	
	#TODO: check relation to synapse
	def set_netcon(self, synapse, stim):
		nc = h.NetCon(stim, synapse.pp_obj, 1, 0, 1)
		self.netcons_list.append(nc)
		synapse.ncs.append(nc)
		return nc
	
	#TODO: add rationale for exp
	def get_mean_fr(self, mean_firing_rate: object, exp = False) -> float:
		if callable(mean_firing_rate): # mean_firing_rate is a distribution
			 # Sample from the distribution
			mean_fr = mean_firing_rate(size = 1)
		else: # mean_firing_rate is a float
			mean_fr = mean_firing_rate
		
		if exp:
			mean_fr = np.exp(mean_fr)

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
