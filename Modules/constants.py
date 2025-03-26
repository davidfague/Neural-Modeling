from dataclasses import dataclass, field
from Modules.synapse import CS2CP_syn_params, CP2CP_syn_params, FSI_syn_params, LTS_syn_params, PV2PN_syn_params, SOM2PN_syn_params, PN2PN_syn_params

import numpy as np

@dataclass
class SimulationParameters:
	
	# Name: required argument
	sim_name: str

	# optionally name the morphology. (cab be read in analyses)
	morphology_name: str = 'no_morphology_name_provided'

	# optionally name the simulation type (can be read in analyses)
	sim_type: str = 'no_sim_type_provided'

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
	h_i_amplitude: float = 10.0 # (nA)
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

	# inh gmax distributions
	# inh_gmax_dist: float = 5#0.5
	# inh_perisomatic_gmax_dist: tuple = (1.324, 0.373)#(1.324*2, 0.373) # mean, std (normal distribution) #mean*4 is decent too
	# inh_dendritic_gmax_dist: tuple = (1.4035*2,0.08474)#(1.4035*2*2,0.08474/8) # mean, std (normal distribution)
	# inh_dendritic_gmax_dist: dict = field(default_factory=lambda: {
    #     'distal_apic': (1.4035*2, 0.08474*2),
    #     'distal_basal': (1.4035*1.5, 0.08474)
    # 	})
	# soma_gmax_dist: float = 5#0.5
	# apic_inh_gmax_dist: float = 2.8
	# basal_inh_gmax_dist: float = 2.4

	# exc gmax distributions
	bin_exc_gmax: bool = False # controls if the exc gmax values should be limited on the values they can take (helps with merging synapses)
	# exc_gmax_mean_0: float = (np.log(0.45) - 0.5 * np.log((0.35/0.45)**2+1))#0.45#2.3#1.5 # 1.5-1.6 is good
	# exc_gmax_std_0: float = np.sqrt(np.log((0.35/0.45)**2 + 1))#0.35
	# exc_gmax_clip: tuple = (0,5)#(0,5)#(0, 15)
	# exc_scalar_basal: int = 1#1.5 # Scales weight
	# exc_scalar_apical: int = 1#0.5 # Scales weight
	# trunk_exc_gmax_mean: float = exc_gmax_mean_0
	# trunk_exc_gmax_std: float = exc_gmax_std_0
	# oblique_exc_gmax_mean: float = exc_gmax_mean_0
	# oblique_exc_gmax_std: float = exc_gmax_std_0
	# tuft_exc_gmax_mean: float = exc_gmax_mean_0
	# tuft_exc_gmax_std: float = exc_gmax_std_0

	# Density/Number of synapses
	exc_use_density: bool = True # NOTE: setting to false uses "exc_syn_number" instead of "exc_synaptic_density"
	inh_use_density: bool = True # NOTE: setting to false uses "inh_syn_number" instead of "inh_synaptic_density"
	use_SA_probs: bool = False # NOTE: Use surface area instead of lengths for the synapse's segment assignment probabilities (does not yet change the total number calculated using density?)
	
	# Current densities taken from literature on apical main bifurcation, and extrapolated to entire cell.
	# exc_synaptic_density: float = 2.16 # (syn/micron of path length)
	# inh_synaptic_density: float = 0.22 # (syn/micron of path length)
	# exc_syn_number: int = 26112
	# inh_syn_number: int = 3066
	# NOTE: we will be matching synapses weights to PSCs from literature and adjusting the synapse densities to match the proper voltage response

	# NOTE: the 'synapse number' is how many you would expect to be on the whole cell,
	# the actual syn numbers on that section type are proportionally scaled by surface area or length, whichever is indicated by 'use_SA_probs'
	# i.e. if trunk is 50% of the cell's length, then N = 0.5 * 26112 will be on the trunk
	exc_syn_properties: dict = field(default_factory=lambda: { 
		'trunk': {'syn_density': 2.16*0.25, 'syn_number': 26112/8, 
			'gmax_params': {'mean': 0.42*0.5, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}
			},
		'oblique': {'syn_density': 2.16*0.4, 'syn_number': 26112/8,
			'gmax_params': {'mean': 0.42*0.5, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}
			},
		'tuft': {'syn_density': 2.16*0.4, 'syn_number': 26112/8,
			'gmax_params': {'mean': 0.42*0.5, 'std': 0.9675, 'clip': (0,2), 'scalar': 1}
			},
		'nexus':{'syn_density': 2.16*0.05, 'syn_number': 26112/8,
			'gmax_params': {'mean': 0.42*0.5, 'std': 0.9675, 'clip': (0,1.5), 'scalar': 1}
			},
		'distal_basal': {'syn_density': 2.16*1.2, 'syn_number': 26112,
			'gmax_params': {'mean': 0.396, 'std': 1.04, 'clip': (0,1.5), 'scalar': 1}
			},
	})
	# NOTE: [trunk, oblique, tuft] fields can be replaced with 'distal_apic' if desired
	# NOTE: gmax is clipped to (0,10*mean); no scalar implemented.
	inh_syn_properties: dict = field(default_factory=lambda: {
		'perisomatic': {'syn_density': 0.22*1.3, 'syn_number': 3066,
			'gmax_params': {'mean': 4.6*0.75, 'std':  0.175},
			'P_release_params': {'mean': 0.88, 'std': 0.05}
			},
		'trunk': {'syn_density': 0.22, 'syn_number': 3066,
			'gmax_params': {'mean': 1.87, 'std': 0.08474},#*0.2*0.66*0.1},
			'P_release_params': {'mean': 0.3, 'std': 0.08}
			},
		'oblique': {'syn_density': 0.22, 'syn_number': 3066,
			'gmax_params': {'mean': 1.87*4, 'std': 0.08474},#*0.2*0.66*0.1},
			'P_release_params': {'mean': 0.3, 'std': 0.08}
			},
		'tuft': {'syn_density': 0.22*1.5, 'syn_number': 3066,
			'gmax_params': {'mean': 1.87*8, 'std': 0.08474},#*0.2*0.66*0.1},#0.08474*0.2*0.66*0.1},
			'P_release_params': {'mean': 0.3, 'std': 0.08}
			},
		'nexus': {'syn_density': 0.22*1.5, 'syn_number': 3066,
			'gmax_params': {'mean': 1.87*10, 'std': 0.08474},#0.08474*0.2*0.66*0.1},
			'P_release_params': {'mean': 0.3, 'std': 0.08}
			},
		'distal_basal': {'syn_density': 0.22, 'syn_number': 3066,
			'gmax_params': {'mean': 1.87, 'std': 0.08474},#0.08474*0.916*0.5*.16},
			'P_release_params': {'mean': 0.72, 'std': 0.1}
			}
	})
	# Synapse Release probability distributions
	exc_P_release_mean: float = 0.53 # can move to syn_properties
	exc_P_release_std: float = 0.22
	# inh_basal_P_release_mean: float = 0.72
	# inh_basal_P_release_std: float = 0.1
	# inh_apic_P_release_mean: float = 0.3
	# inh_apic_P_release_std: float = 0.08
	# inh_soma_P_release_mean: float = 0.88
	# inh_soma_P_release_std: float = 0.05
	# use_P_release_constants: bool = False # can maybe move this to the __init__ like syn_params
	# exc_P_release_constant: float = 0.6 # from Ben's code
	# inh_P_release_constant: float = 0.25 # from Ben's code
	
 
	# syn_mod
	exc_syn_mod: str = 'pyr2pyr'#'AMPA_NMDA_STP'
	inh_syn_mod: str = 'int2pyr'#'GABA_AB_STP'
 
	synaptic_currents_to_record =['iampa', 'inmda']# listed are for pyr2pyr.	for AMPA_NMDA: ["i_AMPA", "i_NMDA"]

	# Firing rate distributions
	use_levy_dist_for_exc: bool = False
	inh_prox_mean_fr: float = 16.9
	inh_prox_std_fr: float = 14.3
	inh_distal_mean_fr: float = 3.9
	inh_distal_std_fr: float = 4.9
	exc_mean_fr: float = 6.7967 #4.43
	exc_std_fr: float = 4.3#3.4503#2.9

	# actual results from np seed 5000
	# number of EXC pcs: 4137. mean/std number of synapses per pc: 5.32, 3.81
	# number of INH pcs: 433. mean/std number of synapses per pc: 5.17, 3.38
	# number of SOMA pcs: 81. mean/std number of synapses per pc: 7.02, 5.46
	# EXC mean fr distribution: 7.28, 4.25
	# INH mean fr distribution: 6.67, 6.13
	# SOMA mean fr distribution: 18.78, 12.92

  	# exc FR FR/FR curve
	exc_constant_fr: bool = False # exc synapses will have firing rate of 0 + self.parameters.excFR_increase
	excFR_increase: float = 0.0

	# input rhythmic modulation # default modulation_depth = 0 (currently only implemented for inhibitory)
	rhyth_depth_inh_perisomatic: float = 0.15 # > 100 microns from soma
	rhyth_frequency_inh_perisomatic: float = 64 # (hz)
	rhyth_depth_inh_distal: float = 0.15 # < 100 microns from soma
	rhyth_frequency_inh_distal: float = 16 # (hz)
	

	# syn_params
	# exc_syn_params: tuple = if 'AMPA' in exc_syn_mod: (CS2CP_syn_params, CP2CP_syn_params) else: PN2PN_syn_params # 90%, 10%
	# inh_syn_params: tuple = if 'GABA' in inh_syn_mod: (FSI_syn_params, LTS_syn_params) else: (PV2PN_syn_params, SOM2PN_syn_params)# (proximal, distal)

	# kmeans clustering
	exc_n_FuncGroups: int = 50
	exc_n_PreCells_per_FuncGroup: int = 100
	inh_n_FuncGroups: int = 10
	inh_n_PreCells_per_FuncGroup: int = 50
	soma_n_fun_gr: int = 1
	soma_n_pc_per_fg: int = 150
    # "divergence":
    # {
    #     "exc": {"min":2, "max":8},
    #     "peri_inh": {"m":2.8, "s":1.9, "min":1, "max":5},
    #     "basal_inh": {"m":2.7, "s":1.6, "min":1 , "max":5},
    #     "apic_inh": {"m":12, "s":3, "min":6 , "max":18}
    # },

	# Excitatory dend
	exc_functional_group_span: int = 100
	exc_cluster_span: int = 10
	exc_synapses_per_cluster: int = 5

	# Inhibitory dend
	inh_cluster_span: int = 10
	inh_number_of_groups: int = 1
	inh_functional_group_span: int = 100

	# Inhibitory soma
	# Number of presynaptic cells
	soma_number_of_clusters: int = 15
	soma_cluster_span: int = 10
	
	# Number of synapses per presynaptic cell
	soma_synapses_per_cluster: int = 10
	soma_number_of_groups: int = 1
	soma_functional_group_span: int = 100

	# Cell model
	spike_threshold: int = -10 # (mV)
	channel_names = []

	# Post Synaptic Current analysis
	number_of_presynaptic_cells: int = 2651
	PSC_start: int = 5

	# Analyze output
	skip: int = 300

	# Log, plot and save
	save_adj_matrix: bool = False
	save_every_ms: int = 1000
	record_every_time_steps: int = 1 # 10 converts from dt=0.1 ms to saved_dt=1 ms # 1 keeps at 0.1 ms
	path: str = ''

	# Reduction (depracating)
	reduce_cell: bool = False
	expand_cable: bool = False
	reduction_frequency: int = 0
	choose_branches: int = 22
	# Whether or not to optimize the number of segments by lambda after reduction 
	# (may need to add an update to the CellModel class instance's segments list and seg_info list.)
	optimize_nseg_by_lambda: bool = False
	# nseg_per_lambda: int = 10

	set_nseg_by_length: bool = True
	microns_per_segment: int = 5
	# Whether or not to merge synapses after optimizing nseg by lambda. 
	# (synapses should already be merged by the reduce_cell_func, 
	# but could be merged again if optimize_nseg_by_lambda lowers nseg.)
	merge_synapses: bool = False
	# Desired number of segs per length constant
	segs_per_lambda: int = 10
 
	# new mar 2024
	# test_morphology: bool = False
	# reduction_before_synapses: bool = False
	Hay_biophys: str = "L5PCbiophys3.hoc"
	# use_mm: bool = False
 
	# stylized (depracating)
	build_stylized: bool = False
	geometry_file: str = "geom_parameters.csv"
 
	# EPSPs (depracating?)
	only_one_synapse: bool = False
	one_syn_index: int = 0
	simulate_EPSPs: bool = False
  
  	# recorders
	record_soma_spikes: bool = True
	record_axon_spikes: bool = False
	record_all_channels: bool = False
	record_all_synapses: bool = False
	record_all_v: bool = True
	record_spike_trains: bool = False
	record_synapse_distributions: bool = False
 
	# new reduction parameters
	reduce_cell_NRCE: bool = False # depracting NRCE
	# reduce_cell_selective:bool = True
	reduce_tufts: int = 0
	reduce_obliques: int = 0
	reduce_apic: int = 0 # cannot do apic with tufts or oblique
	reduce_basals: int = 0
	synapse_mapping: bool = True # True places synapses on complex cell and maps them using transfer impedance. False places synapses onto reduced cell
  
  
	# CI comp
	# disable_apic_37: bool = False
	# disable_basal_1st: bool = False
	reduce_soma_gpas: bool = False
	
	num_basal_to_replace_with_CI: int = 0
	basal_AC_stats: list = field(default_factory=lambda: [
        (0.00693, 0.05926), (-0.0007, 0.05307), (0.01526, 0.09936), 
        (0.00035, 0.0361), (0.00478, 0.17284), (0.01896, 0.07112), 
        (-0.00153, 0.02512), (-0.00151, 0.03715)
    ]) # (mean,std) for each basal dendrite in a full 20 sec complex cell sim
 
	num_tuft_to_replace_with_CI: int = 0
	tuft_AC_stats: list = field(default_factory=lambda: [
     (0.03897, 0.05233), (0.05814, 0.05911)
     ])

	def __post_init__(self):
		if 'AMPA' in self.exc_syn_mod:
			self.exc_syn_params = (CS2CP_syn_params, CP2CP_syn_params)
		elif 'pyr2pyr' in self.exc_syn_mod:
			self.exc_syn_params = PN2PN_syn_params
		else:
			raise(NotImplementedError(f"desired {self.exc_syn_mod} syn_params not specified"))

		if 'GABA' in self.inh_syn_mod:
			self.inh_syn_params = (FSI_syn_params, LTS_syn_params)
		elif 'int2pyr' in self.inh_syn_mod:
			self.inh_syn_params = (PV2PN_syn_params, SOM2PN_syn_params)
		else:
			raise(NotImplementedError(f"desired {self.inh_syn_mod} syn_params not specified"))
            
class HayParameters(SimulationParameters):
	channel_names = [
		'i_pas', 
		# 'ik', 
		'ica', 
		'ina', 
		'ihcn_Ih', 
		'gNaTa_t_NaTa_t', 
		# 'ina_NaTa_t', 
		# 'ina_Nap_Et2', 
		# 'ik_SKv3_1', 
		# 'ik_SK_E2', 
		'ik_Im', 
		'ica_Ca_HVA', 
		'ica_Ca_LVAst']
