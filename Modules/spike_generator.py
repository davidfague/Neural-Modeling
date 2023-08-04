import numpy as np
from scipy.signal import lfilter
from neuron import h
from Modules.cell_model import CellModel
import warnings
import math

def minmax(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

class SpikeGenerator:
  
	def __init__(self):
		self.spike_trains = []
		self.vecstims = []
		self.netcons = []
	
	def generate_inputs(self, synapses: list, t: np.ndarray, mean_firing_rate: object, method: str,
			 			origin: str, random_state: np.random.RandomState, 
						rhythmicity: bool = False, rhythmic_mod: float = None, 
						rhythmic_f: int = None, spike_trains_to_delay: list = None, 
						fr_time_shift: int = None, spike_train_dt: float = 1e-3) -> tuple:
		'''
		Generate netcons and spike trains.

		Parameters:
		----------
		synapses: list
			List of synapse objects.

		t: np.ndarray
			Time array.

		mean_firing_rate: float or distribution
			Mean firing rate of the spike train.
		
		method: str
			How to vary the profile over time. One of ['1f_noise', 'delay'].

		origin: str
			Origin of input, one of ['same_presynaptic_cell', 'same_presynaptic_region'].

		random_state: np.random.RandomState
			RNG.

		rhythmicity: bool = False
			Whether to apply rhythmic modulation.

		rhythmic_mod: float = None
			For rhythmic modulation, modulation strength, a in a * sin(2 * pi * k + b).

		rhythmic_f: int = None
			For rhythmic modulation, the multiplicative period of the sin function, k in a * sin(2 * pi * k + b).

		spike_trains_to_delay: list = None
			For delay modulation, list of time stamps where spikes occured for delay modulation.

		fr_time_shift: int
			For delay modulation, shift of the target spike train's rate profile.

		spike_train_dt: float = 1e-3
			For delay modulation, step used to generate the spike train.

		Returns:
		----------
		netcons_list: list
		
		spike_trains: list
		'''
	
		spike_trains = []
		netcons_list = [] # returned temporary list for this go around

		if origin == "same_presynaptic_cell": # Same fr profile, Same spike train, Same mean fr
			# Ensure the firing rate is a float
			mean_fr = self.get_mean_fr(mean_firing_rate)
			fr_profile = self.get_firing_rate_profile(t = t, method = method, random_state = random_state,
													  rhythmicity = rhythmicity, rhythmic_mod = rhythmic_mod,
					     							  rhythmic_f = rhythmic_f, 
													  spike_trains_to_delay = spike_trains_to_delay,
													  fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
			spikes = self.generate_spikes_from_profile(fr_profile, mean_fr, random_state)
			for synapse in synapses:
				spike_trains.append(spikes)
				netcon = self.set_spike_train(synapse, spikes)
				netcons_list.append(netcon)
		  
		elif origin == "same_presynaptic_region": # Same fr profile, Unique spike train, Unique mean fr
			fr_profile = self.get_firing_rate_profile(t = t, method = method, random_state = random_state, rhythmicity = rhythmicity, 
					     							  rhythmic_mod = rhythmic_mod, rhythmic_f = rhythmic_f, 
													  spike_trains_to_delay = spike_trains_to_delay,
													  fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
			for synapse in synapses:
				mean_fr = self.get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr, random_state)
				spike_trains.append(spikes)
				netcon = self.set_spike_train(synapse, spikes)
				netcons_list.append(netcon)
			
		else: # Unique fr profile, Unique spike train, Unqiue mean fr
			for synapse in synapses:
				fr_profile = self.get_firing_rate_profile(t = t, method = method, random_state = random_state, 
					      								  rhythmicity = rhythmicity,
					      								  rhythmic_mod = rhythmic_mod, rhythmic_f = rhythmic_f,
														  spike_trains_to_delay = spike_trains_to_delay,
														  fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
				mean_fr = self.get_mean_fr(mean_firing_rate)
				spikes = self.generate_spikes_from_profile(fr_profile, mean_fr, random_state)
				netcon = self.set_spike_train(synapse, spikes)
				spike_trains.append(spikes)
				netcons_list.append(netcon)
			
		return netcons_list, spike_trains

	def get_firing_rate_profile(self, t: np.ndarray, method: str, random_state: np.random.RandomState,
								rhythmicity: bool = False, rhythmic_mod = None, rhythmic_f = None,
								spike_trains_to_delay = None, fr_time_shift = None, 
								spike_train_dt: float = 1e-3, bounds: tuple = (0.5, 1.5),
								tiesinga_params: tuple = ()) -> np.ndarray:
		'''
		Parameters:
		----------
		t: np.ndarray
			Time array.
		
		method: str
			How to vary the profile over time. One of ['1f_noise', 'delay', 'gaussian'].
		
		random_state: np.random.RandomState
			RNG.
		
		rhythmicity: bool = False
			Whether to apply rhythmic modulation.

		rhythmic_mod: float = None
			For rhythmic modulation, modulation strength, a in a * sin(2 * pi * k + b).

		rhythmic_f: int = None
			For rhythmic modulation, the multiplicative period of the sin function, k in a * sin(2 * pi * k + b).

		spike_trains_to_delay: list = None
			For delay modulation, list of time stamps where spikes occured for delay modulation.

		fr_time_shift: int
			For delay modulation, shift of the target spike train's rate profile.

		spike_train_dt: float = 1e-3
			For delay modulation, step used to generate the spike train.
			
		'''

		# Create the firing rate profile
		if method == '1f_noise':
			fr_profile = self.noise_modulation(num_obs = len(t), random_state = random_state)
		elif method == 'delay':
			fr_profile = self.delay_modulation(spike_trains_to_delay = spike_trains_to_delay, fr_time_shift = fr_time_shift, 
				      						   spike_train_t = t, spike_train_dt = spike_train_dt)
		elif method == "gaussian":
			a_iv, P, CV_t, sigma_iv, pad_aiv = tiesinga_params 
			fr_profile = self.gaussian_modulation(random_state = random_state, train_length = len(t), dt = spike_train_dt,
					 							  a_iv = a_iv, P = P, CV_t = CV_t, sigma_iv = sigma_iv, bounds = bounds,
												  pad_aiv = pad_aiv)
		else:
			raise NotImplementedError
		
		if rhythmicity:
			fr_profile = self.rhythmic_modulation(fr_profile, rhythmic_f, rhythmic_mod, t)
		
		# Can't have negative firing rates (precision reasons)
		if np.sum(fr_profile < 0) != 0:
			warnings.warn("Found zeros in fr_profile.")
		fr_profile[fr_profile < 0] = 0 
		
		return fr_profile
	
	def noise_modulation(self, random_state: np.random.RandomState, 
		      			 num_obs: int, A: list = None, B: list = None, 
						 bounds: tuple = (0.5, 1.5)) -> np.ndarray:
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
		
		white_noise = random_state.normal(loc = 1, scale = 0.5, size = num_obs + 2000)

		# Apply the FIR/IIR filter to create the 1/f noise, minmax and shift to bounds
		fr_profile = minmax(lfilter(B, A, white_noise)[2000:]) * (bounds[1] - bounds[0]) + bounds[0]

		return fr_profile
	
	def delay_modulation(self, spike_trains_to_delay: list, fr_time_shift: int, spike_train_t: int, 
			  			 spike_train_dt: float = 1e-3, bounds = (0, 2)) -> np.ndarray:
		'''
		Compute a firing rate profile from a target spike train and shift it to create a new profile.

		Parameters:
		----------
		spike_trains_to_delay: list
			Target spike trains.

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

		fr_profile = hist / (spike_train_dt * (len(spike_trains_to_delay) + 1))

		# Shift by fr_time_shift
		wrap = fr_profile[-fr_time_shift:]
		fr_profile[fr_time_shift:] = fr_profile[0:-fr_time_shift]
		fr_profile[0:fr_time_shift] = wrap

		# Minmax and shift to bounds
		fr_profile = minmax(fr_profile) * (bounds[1] - bounds[0]) + bounds[0]

		return fr_profile

	def gaussian_density(self, x: np.ndarray, mu: float, sigma: float):
		return (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
	
	def gaussian_modulation(self, random_state: np.random.RandomState, train_length: float, 
			 				dt: float, a_iv: int, P: float, CV_t: float, sigma_iv: float,
							bounds: tuple, pad_aiv: int = 0) -> np.ndarray:
		'''
		Paramters:
		----------

		random_state: np.random.RandomState
			RNG to use.

		train_length: float
			Train length in ms.
		
		dt: float
			Time step to discretize with.

		a_iv: int
			Number of spike trains to generate.

		P: float
			Mean of distance between volleys in ms.

		CV_t: float
			Coefficient of variation of distance between volleys in ms.

		sigma_iv: float
			Volley's std.

		bounds: tuple
			The profile bounds: [min, max].

		Returns:
		----------
		fr_profile: np.ndarray
			Firing rate profile.
		'''
		num_timestamps = int(train_length / dt)
		num_trains = int(a_iv + pad_aiv * 2)

		fr_profile = np.zeros((num_trains, num_timestamps)) # In timestamps

		timestamp = P
		while timestamp < num_timestamps:
			# Place a Gaussian at stamp centered at a_iv // 2 with std = sigma_iv
			gaus = self.gaussian_density(np.arange(num_trains), int(num_trains // 2), sigma_iv)

			# Normalize to get probability and place
			prob = gaus / np.max(gaus)
			fr_profile[:, int(timestamp)] = minmax(prob) * (bounds[1] - bounds[0]) + bounds[0]
			timestamp += int(random_state.normal(int(P / dt), int(CV_t * P / dt)))

		return fr_profile


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
	
	#TODO: check division by 1000
	def generate_spikes_from_profile(self, fr_profile: np.ndarray, mean_fr: float, random_state: np.random.RandomState) -> np.ndarray:
		fr_profile = fr_profile * mean_fr
		sample_values = random_state.poisson(fr_profile / 1000)
		spike_times = np.where(sample_values > 0)[0]
		return spike_times
   
	def create_netstim(self, frequency=None, interval=None, number=None, duration=None, start=None, noise=0):
		'''
    function for creating a NetStim. Provide either frequency or interval and number or duration.
    frequency: float
      desired frequency for stimulus (Hz)
    interval: float
      mean time between spikes (ms)
    number: int
      average number of spikes
    duration: float
      desired duration of stimulus (ms)
    start: float
      most likely start time of first spike (ms)
    noise: range 0 to 1 
      Fractional randomness. 0 deterministic, 1 intervals have negexp distribution.
      an interval between spikes consists of a fixed interval of duration (1 - noise)*interval plus a negexp interval of mean duration noise*interval. 
      Note that the most likely negexp interval has duration 0.
    '''
		if start is None:
			raise(ValueError("Pass start argument."))  
    
		if (frequency is not None) & (interval is not None):
			raise(ValueError("Pass either frequency or interval argument. Not Both."))
		elif (frequency is None) & (interval is None):
			raise(ValueError("Pass either frequency or interval argument."))
		elif (frequency is not None):
			interval = 1000/frequency # calculate interval from frequency # convert from Hz to ms.
    
		if (number is not None) & (duration is not None):
			raise(ValueError("Pass either number or duration argument. Not Both."))
		elif (number is None) & (duration is None):
			raise(ValueError("Pass either number or duration argument."))
		elif (duration is not None):
			number = math.floor(duration/interval) # calculate number of spike from duration and spike interval
    
    #create NetStim hoc object
		stim = h.NetStim()
		stim.interval = interval
		stim.number = number
		stim.start = start
		stim.noise = noise
    
		return stim
	
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
				 				origin: str, random_state: np.random.RandomState, rhythmicity: bool = False, rhythmic_mod = None, 
								rhythmic_f = None, spike_trains_to_delay = None, fr_time_shift = None, spike_train_dt: float = 1e-3) -> None:
		'''
		Generate spike trains on an existing cell object.

		Parameters:
		----------
		cell: CellModel
			Cell to generate on.

		synapses: list
			List of synapse objects.

		t: np.ndarray
			Time array.

		mean_firing_rate: float or distribution
			Mean firing rate of the spike train.
		
		method: str
			How to vary the profile over time. One of ['1f_noise', 'delay'].

		origin: str
			Origin of input, one of ['same_presynaptic_cell', 'same_presynaptic_region'].

		random_state: np.random.RandomState
			RNG.

		rhythmicity: bool = False
			Whether to apply rhythmic modulation.

		rhythmic_mod: float = None
			For rhythmic modulation, modulation strength, a in a * sin(2 * pi * k + b).

		rhythmic_f: int = None
			For rhythmic modulation, the multiplicative period of the sin function, k in a * sin(2 * pi * k + b).

		spike_trains_to_delay: list = None
			For delay modulation, list of time stamps where spikes occured for delay modulation.

		fr_time_shift: int
			For delay modulation, shift of the target spike train's rate profile.

		spike_train_dt: float = 1e-3
			For delay modulation, step used to generate the spike train.
		'''
		
		netcons, spike_trains = self.generate_inputs(synapses = synapses, t = t, mean_firing_rate = mean_firing_rate, method = method, 
			 										 origin = origin, random_state = random_state, rhythmicity = rhythmicity, 
													 rhythmic_mod = rhythmic_mod, rhythmic_f = rhythmic_f, spike_trains_to_delay = spike_trains_to_delay, 
													 fr_time_shift = fr_time_shift, spike_train_dt = spike_train_dt)
		for netcon in netcons:
			cell.netcons.append(netcon)
		for spike_train in spike_trains:
			cell.spike_trains.append(spike_train)
