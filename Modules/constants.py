from dataclasses import dataclass

@dataclass
class SimulationParameters:
	
	# Name: required argument
	sim_name: str

	# Random state
	numpy_random_state: int = 130
	neuron_random_state: int = 90

	# Runtime
	parallelize: bool = True

	# Reduction
	reduce_cell: bool = False
	expand_cable: bool = False
	reduction_frequency: int = 0
	choose_branches: int = 22
	# Whether or not to optimize the number of segments by lambda after reduction 
	# (may need to add an update to the CellModel class instance's segments list and seg_info list.)
	optimize_nseg_by_lambda: bool = True
	# Whether or not to merge synapses after optimizing nseg by lambda. 
	# (synapses should already be merged by the reduce_cell_func, 
	# but could be merged again if optimize_nseg_by_lambda lowers nseg.)
	merge_synapses: bool = False
	# Desired number of segs per length constant
	segs_per_lambda: int = 10

	# Morphology parameters used if build_m1
	# SomaL = 28.896601873591436
	# SomaDiam = 14.187950175330796
	# AxonL = 549.528226526987
	# AxonDiam = 1.0198477329563544

	# Neymotin Reduced
	SomaL: float = 48.4123467666
	SomaDiam: float = 28.2149102762
	AxonL: float = 594.292937602 # 549.528226526987
	AxonDiam: float =  1.40966286462
	Axon_L_scale: float = 1 # Used to adjust axon length while maintaing surface area

	# Neuron parameters
	h_celcius: float = 34 # 37
	h_tstop: int = 5000 # Sim runtime (ms)
	h_dt: float = 0.1 # Timestep (ms)

	# Current injection
	CI_on: bool = False
	h_i_amplitude: float = -1.0 # (nA)
	h_i_duration: int = 5000 # (ms)
	h_i_delay: int = 10 # (ms)

	trunk_exc_synapses: bool = True
	perisomatic_exc_synapses: bool = False
	add_soma_inh_synapses: bool = True
	num_soma_inh_syns: int = 150

	# gmax distributions
	exc_gmax_mean_0: float = 0.45
	exc_gmax_std_0: float = 0.345
	exc_gmax_clip: tuple = (0, 0.5)
	inh_gmax_dist: float = 1
	soma_gmax_dist: float = 1
	inh_scalar: int = 1.1
	exc_scalar: int = 1 # Scales weight

	# Synapse density syns/um 
	# Current densities taken from literature on apical main bifurcation, and extrapolated to entire cell.
	exc_synaptic_density: float = 2.16 # (syn/micron of path length)
	inh_synaptic_density: float = 0.22 # (syn/micron of path length)
	use_SA_exc: bool = True # Use surface area instead of lengths for the synapse's segment assignment probabilities

	# Release probability distributions
	exc_P_release_mean: float = 0.53
	exc_P_release_std: float = 0.22
	inh_basal_P_release_mean: float = 0.72
	inh_basal_P_release_std: float = 0.1
	inh_apic_P_release_mean: float = 0.3
	inh_apic_P_release_std: float = 0.08
	inh_soma_P_release_mean: float = 0.88
	inh_soma_P_release_std: float = 0.05

	# syn_mod
	exc_syn_mod: str = 'AMPA_NMDA_STP'
	inh_syn_mod: str = 'GABA_AB_STP'

	# firing rate distributions
	inh_prox_mean_fr: float = 16.9
	inh_prox_std_fr: float = 14.3
	inh_distal_mean_fr: float = 3.9
	inh_distal_std_fr: float = 4.9
	inh_firing_rate_time_shift: int = 4

	# Syn parameters
	#TODO: implement
	# Can also implement area related synapse density, and increase membrane capacitance for exc spines.
	# inh distal apic:
	# inh distal apic:
	LTS_syn_params = {
		'e_GABAA': -90.,
		'Use': 0.3,
		'Dep': 25.,
		'Fac': 100.
		}
	
	# inh perisomatic
	FSI_syn_params = {
		'e_GABAA': -90.,
		'Use': 0.3,
		'Dep': 400.,
		'Fac': 0.
		}
	
	# exc choice of two:
	CS2CP_syn_params = {
		'tau_d_AMPA': 5.2,
		'Use': 0.41,
		'Dep': 532.,
		'Fac': 65.
		}
	
	CP2CP_syn_params = {
		'tau_d_AMPA': 5.2,
		'Use': 0.37,
		'Dep': 31.7,
		'Fac': 519.
		}
	
	exc_syn_params = [CS2CP_syn_params, CP2CP_syn_params] # 90%, 10%
	inh_syn_params = [FSI_syn_params, LTS_syn_params]

	# kmeans clustering
	exc_n_FuncGroups = 24
	exc_n_PreCells_per_FuncGroup = 100
	inh_distributed_n_FuncGroups = 5
	inh_distributed_n_PreCells_per_FuncGroup = 50

	# Excitatory dend
	exc_functional_group_span = 100
	exc_cluster_span = 10
	exc_synapses_per_cluster = 5

	# Inhibitory dend
	inh_cluster_span = 10
	inh_number_of_groups = 1
	inh_functional_group_span = 100

	# Inhibitory soma
	# Number of presynaptic cells
	soma_number_of_clusters = 15
	soma_cluster_span = 10
	
	# Number of synapses per presynaptic cell
	soma_synapses_per_cluster = 10
	soma_number_of_groups = 1
	soma_functional_group_span = 100

	# Cell model
	seg_to_record = 'soma' # Used to set spike recorder
	spike_threshold = -10 # (mV)
	channel_names = ['i_pas', 'i_hd', 'ina', 'ik_kdr','ik_kap','ik_kdmc','ina_nax', 'ica_cal', 'ica_can', 'ica','g_nax']
					
	# Tiesinga
	ties_a_iv = 10
	ties_P = 1
	ties_CV_t = 1
	ties_sigma_iv = 1
	ties_pad_aiv = 0

	# Post Synaptic Current analysis
	number_of_presynaptic_cells = 2651
	PSC_start = 5

	# analyze output
	skip = 300

	# Log, plot and save
	log_every_ms = 1000
	save_every_ms = 1000
	path = ''

class HayParameters(SimulationParameters):
	channel_names = ['i_pas', 'ik', 'ica', 'ina', 'i_h'] + ['gNaTa_t_NaTa_t', 'ina_NaTa_t', 'ina_Nap_Et2', 'ik_SKv3_1', 'ik_SK_E2', 'ik_Im', 'ica_Ca_HVA', 'ica_Ca_LVAst']
