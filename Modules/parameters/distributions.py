"""
Distribution functions for synaptic parameters.

This module provides various probability distribution functions used for
initializing synaptic weights, firing rates, and release probabilities.
"""

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize


def norm_dist(mean, std, size, clip):
	"""Normal distribution with clipping."""
	val = np.random.normal(mean, std, size)
	s = float(np.clip(val, clip[0], clip[1]))
	return s


def log_norm_dist(mean, std, scalar, size, clip):
	"""
	Lognormal distribution with clipping.
	
	Converts desired mean and std to underlying normal distribution parameters.
	For a lognormal distribution with desired mean m and std s:
	μ (underlying normal mean) = ln(m² / √(m² + s²))
	σ (underlying normal std) = √(ln(1 + (s/m)²))
	"""
	if mean <= 0:
		raise ValueError(f"log_norm_dist: mean must be positive, got {mean}")
	if std < 0:
		raise ValueError(f"log_norm_dist: std must be non-negative, got {std}")
	
	variance = std ** 2
	mean_squared = mean ** 2
	mu = np.log(mean_squared / np.sqrt(mean_squared + variance))
	sigma = np.sqrt(np.log(1 + variance / mean_squared))
	
	val = np.random.lognormal(mu, sigma, size)
	s = scalar * float(np.clip(val, clip[0], clip[1]))
	return s


def precompute_bin_means(gmax_mean, gmax_std, gmax_scalar, clip, large_sample_size=10000):
	"""Precompute bin means for lognormal distribution (legacy, use create_binned_version instead)."""
	if gmax_mean <= 0:
		raise ValueError(f"precompute_bin_means: gmax_mean must be positive, got {gmax_mean}")
	if gmax_std < 0:
		raise ValueError(f"precompute_bin_means: gmax_std must be non-negative, got {gmax_std}")
	
	variance = gmax_std ** 2
	mean_squared = gmax_mean ** 2
	mu = np.log(mean_squared / np.sqrt(mean_squared + variance))
	sigma = np.sqrt(np.log(1 + variance / mean_squared))
	
	val = np.random.lognormal(mu, sigma, large_sample_size)
	s = gmax_scalar * np.clip(val, clip[0], clip[1])
	
	num_bins = 10
	bin_edges = np.percentile(s, np.linspace(0, 100, num_bins + 1))
	bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
	
	return bin_means


def binned_log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size, clip, bin_means):
	"""Lognormal distribution with values assigned to nearest bin mean (legacy, use create_binned_version instead)."""
	if gmax_mean <= 0:
		raise ValueError(f"binned_log_norm_dist: gmax_mean must be positive, got {gmax_mean}")
	if gmax_std < 0:
		raise ValueError(f"binned_log_norm_dist: gmax_std must be non-negative, got {gmax_std}")
	
	variance = gmax_std ** 2
	mean_squared = gmax_mean ** 2
	mu = np.log(mean_squared / np.sqrt(mean_squared + variance))
	sigma = np.sqrt(np.log(1 + variance / mean_squared))
	
	val = np.random.lognormal(mu, sigma, size)
	s = gmax_scalar * np.clip(val, clip[0], clip[1])
	
	binned_values = np.zeros_like(s)
	for i in range(size):
		bin_index = np.digitize(s[i], bin_means) - 1
		binned_values[i] = bin_means[bin_index]
	return binned_values


def create_binned_version(base_function, params, large_sample_size=10000, num_bins=10):
	"""Wrap any distribution function to return binned values (assigns to nearest bin mean)."""
	sample_params = params.copy()
	sample_params['size'] = large_sample_size
	samples = base_function(**sample_params)
	
	if isinstance(samples, (list, np.ndarray)):
		bin_edges = np.percentile(samples, np.linspace(0, 100, num_bins + 1))
	else:
		return base_function
		
	bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]
	
	def binned_wrapper(**kwargs):
		values = base_function(**kwargs)
		
		if isinstance(values, (list, np.ndarray)):
			binned_values = np.zeros_like(values)
			for i in range(len(values)):
				closest_bin_idx = np.argmin(np.abs(np.array(bin_means) - values[i]))
				binned_values[i] = bin_means[closest_bin_idx]
			return binned_values
		else:
			closest_bin_idx = np.argmin(np.abs(np.array(bin_means) - values))
			return bin_means[closest_bin_idx]
			
	return binned_wrapper


def exp_levy_dist(alpha=1.37, beta=-1.00, loc=0.92, scale=0.44, size=1, clip=(0, np.inf)):
	"""Exponentiated Lévy stable distribution. Use calibrate_exp_levy_params() for desired mean/std."""
	levy_samples = st.levy_stable.rvs(alpha=alpha, beta=beta, loc=loc, scale=scale, size=size)
	return np.clip(np.exp(levy_samples) + 1e-15, clip[0], clip[1])


def exp_levy_params_from_mean(target_mean, alpha=1.37, beta=-1.00, clip=(0, 10)):
	"""Fast empirical approximation for Lévy params from target mean. For accuracy, use calibrate_exp_levy_params()."""
	loc = np.log(target_mean) - 0.5
	scale = 0.44
	
	return {
		'loc': loc,
		'scale': scale,
		'alpha': alpha,
		'beta': beta,
		'clip': clip
	}


def calibrate_exp_levy_params(target_mean, target_std=None, alpha=1.37, beta=-1.00, 
                              clip=(0, 10), n_samples=50000, tolerance=0.05, verbose=False):
	"""Optimize Lévy params for target mean/std via numerical optimization. Returns dict with 'loc', 'scale', and actual stats."""
	
	def objective(params):
		loc, scale = params
		if scale <= 0:
			return 1e10
		
		try:
			samples = exp_levy_dist(alpha=alpha, beta=beta, loc=loc, scale=scale, 
			                       size=n_samples, clip=clip)
			actual_mean = np.mean(samples)
			actual_std = np.std(samples)
			
			mean_error = ((actual_mean - target_mean) / target_mean) ** 2
			if target_std is not None:
				std_error = ((actual_std - target_std) / target_std) ** 2
				return mean_error + std_error
			else:
				return mean_error
		except:
			return 1e10
	
	if target_mean < 0.5:
		loc_init = -1.5
	elif target_mean < 1.0:
		loc_init = -1.0
	elif target_mean < 2.0:
		loc_init = 0.0
	else:
		loc_init = 0.3
	
	scale_init = 0.35 if target_std and target_std < 0.5 else 0.44
	
	result = minimize(objective, x0=[loc_init, scale_init], 
	                 method='Nelder-Mead',
	                 options={'maxiter': 100, 'xatol': 0.01, 'fatol': 0.001})
	
	loc_opt, scale_opt = result.x
	
	np.random.seed(42)
	final_samples = exp_levy_dist(alpha=alpha, beta=beta, loc=loc_opt, 
	                              scale=scale_opt, size=n_samples, clip=clip)
	final_mean = np.mean(final_samples)
	final_std = np.std(final_samples)
	
	if verbose:
		print(f"Calibration complete:")
		print(f"  Target: mean={target_mean:.3f}, std={target_std if target_std else 'any'}")
		print(f"  Result: mean={final_mean:.3f}, std={final_std:.3f}")
		print(f"  Params: loc={loc_opt:.6f}, scale={scale_opt:.6f}")
	
	return {
		'loc': loc_opt,
		'scale': scale_opt,
		'actual_mean': final_mean,
		'actual_std': final_std,
		'alpha': alpha,
		'beta': beta,
		'clip': clip
	}


def gamma_dist(mean, size=1):
	"""Gamma distribution with fixed shape parameter."""
	shape = 5
	scale = mean / shape
	return np.random.gamma(shape, scale, size) + 1e-15


def P_release_dist(mean, std, size):
	"""Release probability distribution (normal with [0,1] clipping)."""
	val = np.random.normal(mean, std, size)
	s = float(np.clip(val, 0, 1))
	return s
