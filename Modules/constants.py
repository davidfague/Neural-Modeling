from dataclasses import dataclass, field
from Modules.synapse import CS2CP_syn_params, CP2CP_syn_params, FSI_syn_params, LTS_syn_params, PV2PN_syn_params, SOM2PN_syn_params, PN2PN_syn_params

import numpy as np
import scipy.stats as st
@dataclass
class SimulationParameters:
	
	# Name: required argument
	sim_name: str

	# optionally name the morphology. (cab be read in analyses)
	morphology_name: str = 'no_morphology_name_provided'
	sec_type_rules: dict = field(default_factory=lambda: { # controls for precise section typing (tuft, nexus, oblique, trunk, perisomatic, distal_basal)
			"perisomatic_max": 50,
			"prox_apic_min":   50,
			"nexus_min":      500,
			"nexus_max":      950,
			"tuft_min":       950,
			"apic_y_min": 400,
			"use_graph_for_obliques": True
	})


	# optionally name the simulation type (can be read in analyses)
	sim_type: str = 'no_sim_type_provided'

	skeleton_cell_type: str = 'Hay'

	us_allen_cell: bool = False # if true, use the allen cell

	segment_measurement_for_probabilities: str = 'length' # 'length' or 'surface_area' @TODO: remove parameters.use_SA_probs. cannot remove for original pipeline's sake tho.

	# Random state
	numpy_random_state: int = 130
	neuron_random_state: int = 90

	# Environment parameters
	h_celcius: float = 37 # 34
	h_tstop: int = 5000 # Sim runtime (ms)
	h_dt: float = 0.1 # Timestep (ms)
	h_v_init: float = -77.2 # resting voltage all compartments (mV)

	# Current injection
	CI_on: bool = False
	CI_target: str = 'soma'
	h_i_amplitude: float = 0.0#10.0 # (nA)
	h_i_duration: int = 1000 # (ms)
	h_i_delay: int = 10 # (ms)
  
  #record
	# ECP
	record_ecp: bool = False
	record_seg_to_seg: bool = False
 
	all_synapses_off: bool = False
	trunk_exc_synapses: bool = True
	perisomatic_exc_synapses: bool = True
	add_soma_inh_synapses: bool = True
	# num_soma_inh_syns: int = 450 # 150 PCs * ~3 divergence

	# exc gmax distributions
	bin_exc_gmax: bool = False # controls if the exc gmax values should be limited on the values they can take (helps with merging synapses)

	# Density/Number of synapses
	exc_use_density: bool = True # NOTE: setting to false uses "exc_syn_number" instead of "exc_synaptic_density"
	inh_use_density: bool = True # NOTE: setting to false uses "inh_syn_number" instead of "inh_synaptic_density"
	
	# NOTE: we will be matching synapses weights to PSCs from literature and adjusting the synapse densities to match the proper voltage response

	# NOTE: the 'synapse number' is how many you would expect to be on the whole cell,
	# the actual syn numbers on that section type are proportionally scaled by surface area or length, whichever is indicated by 'use_SA_probs'
	# i.e. if trunk is 50% of the cell's length, then N = 0.5 * 26112 will be on the trunk

	# exc_mean_fr_distribution_function: str = 'levy' # 'levy' or 'gamma' # TODO: use string to change. NOTIMPLEMENTED
	#NOTE: also see __post_init__ for mean_fr distriubtions, initW functions, and synapse
	# TODO: merge exc_syn_properties and inh_syn_properties into one dictionary with prop syn_type=[]'exc'|'inh']
	exc_syn_properties: dict = field(default_factory=lambda: { 
		'tuft_local_L23': {
			'sec_type': 'tuft',
			'syn_density': 2.16*0.10*0.66*1.3, # 10% of inputs are local and 2/3 local are L23 # reduce to 20% to prevent constant depolarization
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,3), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 3333333},
			'synapse_type': 'exc', #TODO: synapse_type can be in zip with exc_syn_properties and inh_syn_properties when they are accessed.
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'tuft_local_L5': {
			'sec_type': 'tuft',
			'syn_density': 2.16*0.10*0.33*1.3, #10% are local and 1/3 local are L5 # reduce to 20% to prevent constant depolarization
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,3), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 3333333},
			'synapse_type': 'exc', #TODO: synapse_type can be in zip with exc_syn_properties and inh_syn_properties when they are accessed.
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'tuft_distant': {
			'sec_type': 'tuft',
			'syn_density': 2.16*0.90*1.3, # 90% are distant # reduce to 20% to prevent constant depolarization
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,3), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 3333333},
			'synapse_type': 'exc', #TODO: synapse_type can be in zip with exc_syn_properties and inh_syn_properties when they are accessed.
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'nexus_local_L23':{
			'sec_type': 'nexus',
			'syn_density': 2.16*0.10*0.25*0.25, #  10% are local and 1/4 local are L23 # decrease to 5% to encourage propagation from tuft instead of spontaneous Ca spike
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 444444444},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'nexus_local_L5':{
			'sec_type': 'nexus',
			'syn_density': 2.16*0.10*0.75*0.25, #  10% are local and 3/4 local are L5 # decrease to 5% to encourage propagation from tuft instead of spontaneous Ca spike
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 444444444},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'nexus_distant':{
			'sec_type': 'nexus',
			'syn_density': 2.16*0.90*0.25, # 90% are distant # decrease to 5% to encourage propagation from tuft instead of spontaneous Ca spike.
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 444444444},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'trunk_distant': {
			'sec_type': 'trunk',
			'syn_density': 2.16*0.25*0.15, # 25% are distant # decrease to 30% to encourage activity propagation through instead of spontaneous activity within.
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 111111},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'trunk_local_L5': {
			'sec_type': 'trunk',
			'syn_density': 2.16*0.75*0.8*0.15, # 75% are local and 4/5 local are L5 # decrease to 30% to encourage activity propagation through instead of spontaneous activity within.
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 111111},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'trunk_local_L23': {
			'sec_type': 'trunk',
			'syn_density': 2.16*0.75*0.2*0.15, #  75% are local and 1/5 local are L23 # decrease to 30% to encourage activity propagation through instead of spontaneous activity within
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 111111},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'oblique_distant': {
			'sec_type': 'oblique',
			'syn_density': 2.16*0.35*0.25, # 35% are distant # decrease to 0.3 to reduce NMDA spikes in obliques
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 222222},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'oblique_local_L23': {
			'sec_type': 'oblique',
			'syn_density': 2.16*0.65*0.25*0.25, # 65% are local and 1/4 local are L23 # decrease to 0.3 to reduce NMDA spikes in obliques
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 222222},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'oblique_local_L5': {
			'sec_type': 'oblique',
			'syn_density': 2.16*0.65*0.75*0.25, # 35% are local and 3/4 local are L5 # decrease to 0.3 to reduce NMDA spikes in obliques
			'initial_weight_distribution': {'params': {'mean': 0.42, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 222222},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'distal_basal_local_L5': {
			'sec_type': 'distal_basal',
			'syn_density': 2.16*0.9*0.9*1.75, # 90% are local, 90% of local are L5 # decrease to 0.95 to control soma firing rate.
			'initial_weight_distribution': {'params': {'mean': 0.396, 'std': 1.04, 'clip': (0,5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 555555555},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'distal_basal_local_L23': {
			'sec_type': 'distal_basal',
			'syn_density': 2.16*0.9*0.1*1.75, # 90% are local, 10% of local are L23 # decrease to 0.95 to control soma firing rate.
			'initial_weight_distribution': {'params': {'mean': 0.396, 'std': 1.04, 'clip': (0,5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 555555555},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'distal_basal_distant': {
			'sec_type': 'distal_basal',
			'syn_density': 2.16*0.10*1.75, # 10% are distant # decrease to 0.95 to control soma firing rate.
			'initial_weight_distribution': {'params': {'mean': 0.396, 'std': 1.04, 'clip': (0,5), 'scalar': 1}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.53, 'std': 0.22}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 555555555},
			'synapse_type': 'exc',
			'spike_train_mode': 'pink_noise',
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
	})
		# input rhythmic modulation # default modulation_depth = 0 (currently only implemented for inhibitory)
	# NOTE: [trunk, oblique, tuft] fields can be replaced with ['distal_apic'] if desired (Not recommended without checking CellModel.get_segments_of_type, etc first.)
	# NOTE: gmax WAS clipped to (0,10*mean); no scalar implemented.
	inh_syn_properties: dict = field(default_factory=lambda: {
		'tuft': {
			'sec_type': 'tuft', # section type to place synapses on
			'syn_density': 0.22,
			'initial_weight_distribution': {'params': {'mean': 1.87, 'std': 0.08474, 'clip': [0,5]}},#*0.2*0.66*0.1},#0.08474*0.2*0.66*0.1},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.3, 'std': 0.08}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 11111111},
			'synapse_type': 'inh',
			'spike_train_mode': ['delay','rhythmic'],#'delay',
			'rhythmic_frequency': 16, # frequency of rhythmic modulation (hz)
			'rhythmic_depth': 0.15, # firing rate timecourse amplitude = depth * mean.
			'delay_config': {
				'ref_synapse_type': 'exc',
				'ref_sec_type': 'all',    # which sec_type in exc to delay
				'ref_fg_id': 'all',            # which FG (use integer, or None for all/first)
				'delay_shift': 4           # delay in samples (ms)
			},
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'nexus': {
			'sec_type': 'nexus',
			'syn_density': 0.22*1.,
			'initial_weight_distribution': {'params': {'mean': 1.87, 'std': 0.08474, 'clip': [0,5]}},#0.08474*0.2*0.66*0.1},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.3, 'std': 0.08}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 22222222},
			'synapse_type': 'inh',
			'spike_train_mode': ['delay','rhythmic'],#'delay',
			'rhythmic_frequency': 16, # frequency of rhythmic modulation (hz)
			'rhythmic_depth': 0.15, # firing rate timecourse amplitude = depth * mean.
			'delay_config': {
				'ref_synapse_type': 'exc',
				'ref_sec_type': 'all',    # which sec_type in exc to delay
				'ref_fg_id': 'all',            # which FG (use integer, or None for all/first)
				'delay_shift': 4           # delay in samples (ms)
			},
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'trunk': {
			'sec_type': 'trunk',
			'syn_density': 0.22,
			'initial_weight_distribution': {'params': {'mean': 1.87, 'std': 0.08474, 'clip': [0,5]}},#*0.2*0.66*0.1},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.3, 'std': 0.08}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 88888888},
			'synapse_type': 'inh',
			'spike_train_mode': ['delay','rhythmic'],#'delay',
			'rhythmic_frequency': 16, # frequency of rhythmic modulation (hz)
			'rhythmic_depth': 0.15, # firing rate timecourse amplitude = depth * mean.
			'delay_config': {
				'ref_synapse_type': 'exc',
				'ref_sec_type': 'all',    # which sec_type in exc to delay
				'ref_fg_id': 'all',            # which FG (use integer, or None for all/first)
				'delay_shift': 4           # delay in samples (ms)
			},
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'oblique': {
			'sec_type': 'oblique',
			'syn_density': 0.22, 
			'initial_weight_distribution': {'params': {'mean': 1.87, 'std': 0.08474, 'clip': [0,5]}},#*0.2*0.66*0.1},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.3, 'std': 0.08}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 99999999},
			'synapse_type': 'inh',
			'spike_train_mode':  ['delay','rhythmic'],#'delay',
			'rhythmic_frequency': 16, # frequency of rhythmic modulation (hz)
			'rhythmic_depth': 0.15, # firing rate timecourse amplitude = depth * mean.
			'delay_config': {
				'ref_synapse_type': 'exc',
				'ref_sec_type': 'all',    # which sec_type in exc to delay
				'ref_fg_id': 'all',            # which FG (use integer, or None for all/first)
				'delay_shift': 4           # delay in samples (ms)
			},
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'distal_basal': {
			'sec_type': 'distal_basal',
			'syn_density': 0.22*1.5,
			'initial_weight_distribution': {'params': {'mean': 1.87, 'std': 0.08474, 'clip': [0,5]}},#0.08474*0.916*0.5*.16},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.72, 'std': 0.1}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 111333311},
			'synapse_type': 'inh',
			'spike_train_mode': ['delay', 'rhythmic'],
			'rhythmic_frequency': 16, # frequency of rhythmic modulation (hz)
			'rhythmic_depth': 0.15, # firing rate timecourse amplitude = depth * mean.
			'delay_config': {
				'ref_synapse_type': 'exc',
				'ref_sec_type': 'all',    # which sec_type in exc to delay
				'ref_fg_id': 'all',            # which FG (use integer, or None for all/first)
				'delay_shift': 4           # delay in samples (ms)
			},
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
		'perisomatic': {
			'sec_type':'perisomatic',
			'syn_density': 0.22,
			'initial_weight_distribution': {'params': {'mean': 4.6, 'std':  0.175, 'clip': [0,5]}},
			'release_probability_distribution': {'function': P_release_dist, 'params': {'mean': 0.88, 'std': 0.05}},
			'mean_firing_rate_distribution': {}, # check __post_init__
			'seed': {'synapses': 777777777},
			'synapse_type': 'inh',
			'spike_train_mode': ['delay', 'rhythmic'],
			'rhythmic_frequency': 64, # frequency of rhythmic modulation (hz)
			'rhythmic_depth': 0.15, # firing rate timecourse amplitude = depth * mean.
			'delay_config': {
				'ref_synapse_type': 'exc',
				'ref_sec_type': 'all',    # which sec_type in exc to delay
				'ref_fg_id': 'all',            # which FG (use integer, or None for all/first)
				'delay_shift': 4           # delay in samples (ms)
			},
			'fr_shift': 0 # shift the mean firing rate by this amount.
			},
	})

	# Clustering parameters (example, see modules/clusters_global_l5_fg.py.)
	exc_clustering: dict = field(default_factory=lambda: {
		# 'tuft': {
		# 	'functional_groups': [
		# 		{
		# 			'center': (0, 900, 500),  # 3D coordinates
		# 			'radius': 500.0,  # microns
		# 			'presynaptic_cells': [
		# 				{
		# 					'center': (0, 900, 500),  # relative to functional group center
		# 					'radius': 0.0,  # microns
		# 					'name': 'PC1',
		# 					'max_synapses': 5000  # maximum number of synapses per PC
		# 				},
						# {
						# 	'center': (0, 900, -25),
						# 	'radius': 0.0,
						# 	'name': 'PC2',
						# 	'max_synapses': 10
						# }
					# ]
				# },
				# {
				# 	'center': (0, 0, 0),
				# 	'radius': 50.0,
				# 	'presynaptic_cells': [
				# 		{
				# 			'center': (0, 0, 0),
				# 			'radius': 00.0,
				# 			'name': 'PC3',
				# 			'max_synapses': 10
				# 		}
				# 	]
				# }
		# 	]
		# },
		# 'trunk': {
		# 	'functional_groups': [
		# 		{
		# 			'center': (0, 0, 0),
		# 			'radius': 30.0,
		# 			'presynaptic_cells': [
		# 				{
		# 					'center': (0, 0, 0),
		# 					'radius': 10.0,
		# 					'name': 'PC4',
		# 					'max_synapses': 10
		# 				}
		# 			]
		# 		}
		# 	]
		# }
	})

	inh_clustering: dict = field(default_factory=lambda: {
		# 'perisomatic': {
		# 	'functional_groups': [
		# 		{
		# 			'center': (0, 0, 0),
		# 			'radius': 30.0,
		# 			'presynaptic_cells': [
		# 				{
		# 					'center': (0, 0, 0),
		# 					'radius': 5.0,
		# 					'name': 'PC1',
		# 					'max_synapses': 10
		# 				}
		# 			]
		# 		}
		# 	]
		# }
	})

	# spike train seeding
	precell_spikes_seeds: dict = field(default_factory=lambda: {
		'exc': 5555555,
		'inh': 7777777,
		'soma_inh': 8888888,
	})
 
	# syn_mod
	exc_syn_mod: str = 'pyr2pyr'#'AMPA_NMDA_STP'
	inh_syn_mod: str = 'int2pyr'#'GABA_AB_STP'
 
	synaptic_currents_to_record =['iampa', 'inmda']# listed are for pyr2pyr.	for AMPA_NMDA: ["i_AMPA", "i_NMDA"]

	# Firing rate distributions
	use_levy_dist_for_exc: bool = True
	inh_proximal_mean_fr: float = 16.9 # 9.75#10
	inh_proximal_std_fr: float = 14.3 # 4
	inh_distal_mean_fr: float = 3.9
	inh_distal_std_fr: float = 4.9
	exc_mean_fr: float = 4.43#6.7967 #4.43
	exc_std_fr: float = 4.3#3.4503#2.9

  	# exc FR FR/FR curve
	exc_constant_fr: bool = False # exc synapses will have firing rate of 0 + self.parameters.excFR_increase
	excFR_increase: float = 0.0

	# input rhythmic modulation # default modulation_depth = 0 (currently only implemented for inhibitory)
	rhyth_depth_inh_perisomatic: float = 0.15 # > 100 microns from soma
	rhyth_frequency_inh_perisomatic: float = 64 # (hz)
	rhyth_depth_inh_distal: float = 0.15 # < 100 microns from soma
	rhyth_frequency_inh_distal: float = 16 # (hz)

	# Analyze output
	skip: int = 300

	# Log, plot and save
	save_adj_matrix: bool = True
	save_every_ms: int = 1000
	record_every_time_steps: int = 1 # 10 converts from dt=0.1 ms to saved_dt=1 ms # 1 keeps at 0.1 ms
	path: str = ''

#### Reduction
	do_reduce_cell: bool = True
	reduction: dict = field(default_factory=lambda: {
		'branch_disparity_elec_tolerance': 0.05, 
		'branch_distance_elec_tolerance': 0.05,
		'series_constant_elec_tolerance': 0.05,
		'series_overhang_elec_tolerance': 0.05,
		'branch_mult_tolerance': 1.5,
		'series_mult_tolerance': 1.5,
		'preserve_roots': False,
		'seg_length_um': 5,
		})
	
##################################### legacy reduction parameters ##########################################################
	reduce_cell: bool = False
	expand_cable: bool = False
	reduction_frequency: int = 0 # input frequency used for calculating input transfer impedance

	# degree of branching to reduce at by section type (0 for none, 1 for the first sections and beyond, 2 for the second sections and beyond, etc.) @TODO: replace with dictionary by section type like synapse properties.
	reduce_tufts: int = 0
	reduce_obliques: int = 0
	reduce_apic: int = 0 # cannot do apic with tufts or obliques
	reduce_basals: int = 0
	synapse_mapping: bool = True # True places synapses on complex cell and maps them using transfer impedance. False places synapses onto reduced cell @TODO: update for synapses_file.
	choose_branches: int = 22 # for cable expander, how many branches to turn one cable into
	
	### Segment resolution parameters #@TODO: check that we are only doing one or the other and that they aren;t conflicting.
	optimize_nseg_by_lambda: bool = False # set the segment resolution according to the length constants (electrotonic properties) of the cable sections.
	segs_per_lambda: int = 10 # number of segments per length constant (lambda) of the cable sections. (more means better resolution, but more segments)

	set_nseg_by_length: bool = True # set the segment resolution according to the length  of the cable sections.
	microns_per_segment: int = 5 # desired length of each segment (more means worse resolution, but fewer segments)

	# Whether or not to merge synapses after optimizing nseg by lambda. #TODO: check where merging is happening. (There is probably a second merge in case the segment resolution is lowered and more synapses can be merged for better runtimes.)
	# (synapses should already be merged by the reduce_cell_func, 
	# but could be merged again if optimize_nseg_by_lambda lowers nseg.)
	merge_synapses: bool = False # deletes duplicate point processes (synapses) by moving their spike trains to 1 like synapse (on the same segment, with the same synapse parameters including synapse weight, release probability, etc.).
##################################### end legacy reduction parameters ##########################################################

 ### Additional file specifications @TODO: move to a separate file.
	Hay_biophys: str = "L5PCbiophys3.hoc"
 
 ### stylized (depracated, will reinstate in the future)
	build_stylized: bool = False
	geometry_file: str = "geom_parameters.csv"

### recorders
	record_soma_spikes: bool = True
	record_axon_spikes: bool = False
	record_all_channels: bool = False
	record_all_synapses: bool = False
	record_all_v: bool = True
	record_spike_trains: bool = False
	record_synapse_distributions: bool = False

	spike_threshold: int = -10 # (mV)
	channel_names = []
  
#### replace dendrite with current injection @TODO: replace with dictionary by section type like synapse properties.
	# disable_apic_37: bool = False # disable apical tuft dendrites  and replace with current injection
	# disable_basal_1st: bool = False # disable basal dendrites and replace with current injection
	reduce_soma_gpas: bool = False
	
	num_basal_to_replace_with_CI: int = 0
	basal_AC_stats: list = field(default_factory=lambda: [ # mean, std axial current for each basal dendrite in a full 20 sec complex cell sim
        (0.00693, 0.05926), (-0.0007, 0.05307), (0.01526, 0.09936), 
        (0.00035, 0.0361), (0.00478, 0.17284), (0.01896, 0.07112), 
        (-0.00153, 0.02512), (-0.00151, 0.03715)
    ]) # (mean,std) for each basal dendrite in a full 20 sec complex cell sim
 
	num_tuft_to_replace_with_CI: int = 0
	tuft_AC_stats: list = field(default_factory=lambda: [
     (0.03897, 0.05233), (0.05814, 0.05911)
     ])


	def __post_init__(self): # define parameters that depend on the above parameters. (such as distributions, choices, etc.)
		# syn params choices
		if 'AMPA' in self.exc_syn_mod:
			self.exc_syn_params_choices = {'choices': [{'CS2CP':CS2CP_syn_params}, {'CP2CP':CP2CP_syn_params}], 'probs': [0.9, 0.1]} # first option is CS2CP (90% prob); second is CP2CP (10% prob)
		elif 'pyr2pyr' in self.exc_syn_mod:
			self.exc_syn_params_choices = {'choices': [{'PN2PN':PN2PN_syn_params}], 'probs': 1.0}
		else:
			raise(NotImplementedError(f"desired {self.exc_syn_mod} syn_params not specified"))
		if 'GABA' in self.inh_syn_mod:
			self.inh_syn_params_choices = {'choices':[{'FSI':FSI_syn_params}, {'LTS':LTS_syn_params}], 'probs': 'perisomatic_distance'} # first option is perisomatic; second is not perisomatic
		elif 'int2pyr' in self.inh_syn_mod:
			self.inh_syn_params_choices = {'choices': [{'PV2PN':PV2PN_syn_params}, {'SOM2PN':SOM2PN_syn_params}], 'probs': 'perisomatic_distance'} # first option is perisomatic; second is not perisomatic
		else:
			raise(NotImplementedError(f"desired {self.inh_syn_mod} syn_params not specified"))
		
		# initW distributions
		# exc
		if self.bin_exc_gmax:
			for input_source, syn_props in self.exc_syn_properties.items():
				self.exc_syn_properties[input_source]['initial_weight_distribution']['function'] = binned_log_norm_dist
		else:
			for input_source, syn_props in self.exc_syn_properties.items():
				self.exc_syn_properties[input_source]['initial_weight_distribution']['function'] = log_norm_dist
		# inh
		for input_source in self.inh_syn_properties.keys():
			self.inh_syn_properties[input_source]['initial_weight_distribution']['function'] = norm_dist

		# mean_fr distributions
		# exc
		if self.use_levy_dist_for_exc:
			for input_source, syn_props in self.exc_syn_properties.items():
				if 'L5' in input_source: # different mean FRs for L5 PNs
					levy_params = {'alpha': 1.37, 'beta': -1.00, 'loc': 0.92*0.25*2.5*0.25, #*1.5,
					'scale': 0.44} # baseline activity (1-3 Hz. 2.2 Hz mean mean firing rate)
					# levy_params = {'alpha': 1.37, 'beta': -1.00, 'loc': 0.92*2, 'scale': 0.44}  # task activity (5-20 Hz. 11.5 Hz mean mean firing rate std 7.5 Hz exponential.)
				elif 'L23' in input_source: # different mean FRs for L23 PNs
					levy_params = {'alpha': 1.37, 'beta': -1.00, 'loc': 0.92*0.1*1.5*1.5*0.25, 'scale': 0.44}# baseline activity (0.5-2Hz. ~1.2 Hz mean mean firing rate) (actually mean of 1.8 couldn't get to go lower.)
					# levy_params = {'alpha': 1.37, 'beta': -1.00, 'loc': 0.92*1, 'scale': 0.44}  # task activity (1-10 Hz. ~4.5 Hz mean mean firing rate. 3 Hz std)
				elif 'distant' in input_source: # distant inputs (same as local L5 for now.)
					levy_params = {'alpha': 1.37, 'beta': -1.00, 'loc': 0.92*0.25*0.25, 'scale': 0.44} # baseline activity (1-3 Hz. 2.2 Hz mean mean firing rate)
					# levy_params = {'alpha': 1.37, 'beta': -1.00, 'loc': 0.92*2, 'scale': 0.44}  # task activity (5-20 Hz. 11.5 Hz mean mean firing rate std 7.5 Hz exponential.)
				else: # other input sources
					raise(NotImplementedError(f"desired {input_source} input source not specified for Levy distribution mean firing rate."))
				# assign the parameters to the mean firing rate distribution
				self.exc_syn_properties[input_source]['mean_firing_rate_distribution']['function'] = exp_levy_dist
				self.exc_syn_properties[input_source]['mean_firing_rate_distribution']['params'] = levy_params
		else: # use normal distribution
			for input_source, syn_props in self.exc_syn_properties.items():
				self.exc_syn_properties[input_source]['mean_firing_rate_distribution']['function'] = norm_dist
				self.exc_syn_properties[input_source]['mean_firing_rate_distribution']['params'] = {'mean': self.exc_mean_fr, 'std': self.exc_std_fr}
		# inh
		for input_source, syn_props in self.inh_syn_properties.items():
			if 'perisomatic' in input_source: # proximal
				mean_fr, std_fr = self.inh_proximal_mean_fr, self.inh_proximal_std_fr
			else:
				mean_fr, std_fr = self.inh_distal_mean_fr, self.inh_distal_std_fr
			self.inh_syn_properties[input_source]['mean_firing_rate_distribution']['function'] = st.truncnorm.rvs
			a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
			self.inh_syn_properties[input_source]['mean_firing_rate_distribution']['params'] = {'a': a, 'b': b, 'loc': mean_fr, 'scale': std_fr}

            
class HayParameters(SimulationParameters):
	channel_names = [ # simulation data to record from each segment
		# 'i_pas', 
		# 'ik', 
		'ica', 
		# 'ina', 
		'ihcn_Ih', 
		'gNaTa_t_NaTa_t', 
		# 'ina_NaTa_t', 
		# 'ina_Nap_Et2', 
		# 'ik_SKv3_1', 
		# 'ik_SK_E2', 
		# 'ik_Im', 
		# 'ica_Ca_HVA', 
		# 'ica_Ca_LVAst'
		]

def norm_dist(mean, std, size, clip): # inh
  val = np.random.normal(mean, std, size)
  s = float(np.clip(val, clip[0], clip[1]))
  return s


def log_norm_dist(mean, std, scalar, size, clip):
	val = np.random.lognormal(mean, std, size)
	s = scalar * float(np.clip(val, clip[0], clip[1]))
	return s

def precompute_bin_means(gmax_mean, gmax_std, gmax_scalar, clip, large_sample_size=10000): # should make this work for any function so we can use for norm_dist, too.
    # Generate a large number of log-normal distributed values
    val = np.random.lognormal(gmax_mean, gmax_std, large_sample_size)
    s = gmax_scalar * np.clip(val, clip[0], clip[1])

    # Determine bins and compute the mean for each bin
    num_bins = 10
    bin_edges = np.percentile(s, np.linspace(0, 100, num_bins + 1))
    bin_means = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(num_bins)]

    return bin_means

def binned_log_norm_dist(gmax_mean, gmax_std, gmax_scalar, size, clip, bin_means):
    # Generate log-normal distributed values
    val = np.random.lognormal(gmax_mean, gmax_std, size)
    # Clip the values
    s = gmax_scalar * np.clip(val, clip[0], clip[1])
    # Assign each value to the nearest bin mean
    binned_values = np.zeros_like(s)
    for i in range(size):
        # Find the bin the value belongs to
        bin_index = np.digitize(s[i], bin_means) - 1
        # Assign the value to the bin mean
        binned_values[i] = bin_means[bin_index]
    return binned_values

# Firing rate distribution
def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
	return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, loc = loc, scale = scale, size = size)) + 1e-15

def gamma_dist(mean, size = 1):
	shape = 5
	scale = mean / shape
	return np.random.gamma(shape, scale, size) + 1e-15

# Release probability distribution
def P_release_dist(P_mean, P_std, size):
	val = np.random.normal(P_mean, P_std, size)
	s = float(np.clip(val, 0, 1))
	return s

# Release probability distribution
def P_release_dist(mean, std, size):
	val = np.random.normal(mean, std, size)
	s = float(np.clip(val, 0, 1))
	return s