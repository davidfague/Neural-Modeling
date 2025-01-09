import numpy as np
from scipy.signal import lfilter
import warnings

def minmax(x):
	if np.min(x) == np.max(x):
		return x
	return (x - np.min(x)) / (np.max(x) - np.min(x))

class SpikeTrain:

	def __init__(self, spike_times, T, mean_fr):
		self.spike_times = spike_times
		self.T = T
		self.mean_fr = mean_fr

class PoissonTrainGenerator:
	# Win_size = 1 ms
	# labmdas refers to an array of firing_rates over time
 
	@staticmethod
	def generate_lambdas_by_delaying(num: int,
			spike_trains: list) -> np.ndarray:
			# num is the number of samples
  
		# Get lambdas
		lambdas = PoissonTrainGenerator.generate_delayed_fr_profile(num=num, spike_trains = spike_trains)

		# # Can't have negative firing rates (precision reasons)
		# if np.sum(lambdas < 0) != 0:
		# 	warnings.warn("Found non-positive lambdas when generating a spike train.")
		# 	lambdas[lambdas < 0] = 0 
		
		return lambdas

	@staticmethod
	def generate_lambdas_from_pink_noise(
			num: int,
			random_state: np.random.RandomState,
			lambda_mean: float = 1.0,
			rhythmic_modulation: bool = False) -> np.ndarray:
		
		# Get lambdas
		lambdas = PoissonTrainGenerator.generate_pink_noise(num_obs = num, random_state = random_state, mean = lambda_mean)

		# Apply modulation
		if rhythmic_modulation:
			lambdas = PoissonTrainGenerator.rhythmic_modulation(lambdas)
		
		# # Can't have negative firing rates (precision reasons)
		# if np.sum(lambdas < 0) != 0:
		# 	warnings.warn("Found non-positive lambdas when generating a spike train.")
		# 	lambdas[lambdas < 0] = 0 
		
		return lambdas
	
	@staticmethod
	def generate_pink_noise(
			random_state: np.random.RandomState, 
			num_obs: int, 
			mean: float = 1,
			std: float = 0.5,
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
		# These values produce stable pink noise
		A = [1, -2.494956002, 2.017265875, -0.522189400]
		B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
		
		white_noise = random_state.normal(loc = mean, scale = std, size = num_obs + 2000)

		# Apply the FIR/IIR filter to create the 1/f noise, minmax and shift to bounds
		fr_profile = minmax(lfilter(B, A, white_noise)[2000:])# * (bounds[1] - bounds[0]) + bounds[0]

		return fr_profile
	
	@staticmethod
	def generate_delayed_fr_profile(num, spike_trains):
		'''trains_matrix is n_pre_cells x n_timepoints'''
		# all of the following are 1x n _timepoints
		old_spike_probs = PoissonTrainGenerator.compute_probs_from_spike_trains(num, spike_trains)
		new_spike_probs = PoissonTrainGenerator.shift_wrap_array(old_spike_probs, 4) # delay the exc spike train
		new_frs = PoissonTrainGenerator.compute_frs_from_probs(new_spike_probs)
		return new_frs

	@staticmethod
	def compute_probs_from_spike_trains(num, spike_trains):
		# Assuming exc_spike_trains is a list of lists containing spike times for each synapse
		nsyn = len(spike_trains)
		# Initialize an array to count spikes at each time point
		spike_counts = np.zeros(num)
		# Count spikes at each time point
		for train in spike_trains:
			spike_counts[train] += 1
		# Calculate the mean spike train
		mean_spike_train = spike_counts / nsyn # mean_spike_train[0] is the probability of observing a spike in a train at t[0]
  
		# Check if any probabilities are greater than 1 and get their indices and values
		indices = np.where(mean_spike_train > 1)[0]
		values = mean_spike_train[indices]
		if len(indices) > 0:
			print(f"Warning: Probabilities greater than 1 detected at indices {indices} with values {values}")

		return mean_spike_train

	@staticmethod
	def shift_wrap_array(arr, by):
		wrap = arr[-by:].copy()
		arr[by:] = arr[0:-by]
		arr[0:by] = wrap
		return arr

	@staticmethod
	def compute_frs_from_probs(spike_probs):
		'''in units mHz'''
		# Clamp spike_probs to avoid 0 and 1
		epsilon = 1e-15
		spike_probs = np.clip(spike_probs, epsilon, 1 - epsilon)
		return -np.log(1 - spike_probs)

	@staticmethod
	def shift_mean_of_lambdas(lambdas, desired_mean):
		'''frs: np.array of firing rates that will be lamda
		units mHz'''
  		# Can't have negative firing rates (precision reasons)
		if np.sum(lambdas < 0) != 0:
			warnings.warn("Found negative lambdas before shifting a mean.")
			lambdas[lambdas < 0] = 1e-15 
		# print(np.sum(lambdas < 0),lambdas)
		# if divide_1000:
		# 	desired_mean = desired_mean / 1000 # shift
		lambdas = lambdas + (desired_mean - np.mean(lambdas))

		# print(np.sum(lambdas < 0),lambdas)
  
  		# Can't have negative firing rates (precision reasons)
		if np.sum(lambdas < 0) != 0:
			warnings.warn("Found negative lambdas after shifting a mean.")
			lambdas[lambdas < 0] = 1e-15 
   
		return lambdas

	@staticmethod
	def delay_spike_trains(spike_trains_to_delay: list, shift: int = 4) -> list:
		delayed_trains = []
		for train in spike_trains_to_delay:
			delayed_times = train.spike_times - shift
			delayed_times[delayed_times < 0] = 0
			delayed_trains.append(SpikeTrain(delayed_times, train.T))

		return delayed_trains

	@staticmethod
	def rhythmic_modulation(lambdas: np.ndarray, frequency: float, depth_of_mod: float, delta_t: float):
		'''mod_trace = mean_fr * (1 + depth_of_mod * np.sin((2 * np.pi * f * t ) + P))
		assynes that delta_t in ms, frequency in hz.'''
		assert 0<=depth_of_mod<=1
		if depth_of_mod == 0: # same as no modulation
			return lambdas
		t = np.linspace(0, len(lambdas) * delta_t * 1e-3, len(lambdas)) # Time array in seconds
		lambdas = lambdas + lambdas * depth_of_mod * np.sin(2 * np.pi * frequency * t)
		return lambdas
	
	@staticmethod
	def generate_spike_train(
			lambdas: np.ndarray,
			random_state: np.random.RandomState) -> np.ndarray:
		
		t = np.zeros(len(lambdas))
		for i, lambd in enumerate(lambdas):
			# Because we are using win-size = 1 ms; lambda = num_spikes_per_second * win_length / 1000
			num_points = random_state.poisson(lambd / 1000)
			if num_points > 0: t[i] = 1

		# print(np.mean(t)*1000)

		spike_times = np.where(t > 0)[0]
		return SpikeTrain(spike_times, len(lambdas), np.mean(t)*1000)

	# @staticmethod
	# def delay_modulation(spike_trains_to_delay: list, fr_time_shift: int, spike_train_t: int, 
	# 		  			 spike_train_dt: float = 1e-3, bounds = (0, 2)) -> np.ndarray:
	# 	'''
	# 	Compute a firing rate profile from a target spike train and shift it to create a new profile.

	# 	Parameters:
	# 	----------
	# 	spike_trains_to_delay: list
	# 		Target spike trains.

	# 	fr_time_shift: int
	# 		Shift of the target spike train's rate profile.

	# 	spike_train_t: int
	# 		Time length that was used to generate spike_trains_to_delay.

	# 	spike_train_dt: float
	# 		Time discretization that was used to generate spike_trains_to_delay.

	# 	bounds: tuple
	# 		The profile bounds: [min, max]. 

	# 	Returns:
	# 	----------
	# 	fr_profile: np.ndarray
	# 		Firing rate profile.

	# 	'''
	# 	if fr_time_shift is None:
	# 		raise TypeError('fr_time_shift must be an integer.')
   
	# 	# Flatten all the spike trains, because otherwise np.histogram doesn't work
	# 	# Ignore spike trains without spikes
	# 	times_where_spikes = [sp_time for sp_train in spike_trains_to_delay for sp_time in sp_train if len(sp_train) > 0]
	# 	# Compute the firing rate profile from the histogram
	# 	hist, _ = np.histogram(times_where_spikes, bins = spike_train_t)

	# 	fr_profile = hist / (spike_train_dt * (len(spike_trains_to_delay) + 1))

	# 	# Shift by fr_time_shift
	# 	wrap = fr_profile[-fr_time_shift:]
	# 	fr_profile[fr_time_shift:] = fr_profile[0:-fr_time_shift]
	# 	fr_profile[0:fr_time_shift] = wrap

	# 	# Minmax and shift to bounds
	# 	fr_profile = minmax(fr_profile) * (bounds[1] - bounds[0]) + bounds[0]

	# 	return fr_profile