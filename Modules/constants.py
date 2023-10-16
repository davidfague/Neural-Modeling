from dataclasses import dataclass

@dataclass
class SimulationParameters:

	# Name: required argument
	sim_name: str

	# Random state
	numpy_random_state = 130
	neuron_random_state = 90

	# Runtime
	parallelize = True

	# Reduction
	reduce_cell = False
	expand_cable = False
	reduction_frequency = 0
	choose_branches = 22
	# Whether or not to optimize the number of segments by lambda after reduction 
	# (may need to add an update to the CellModel class instance's segments list and seg_info list.)
	optimize_nseg_by_lambda = True
	# Whether or not to merge synapses after optimizing nseg by lambda. 
	# (synapses should already be merged by the reduce_cell_func, 
	# but could be merged again if optimize_nseg_by_lambda lowers nseg.)
	merge_synapses = False
	# Desired number of segs per length constant
	segs_per_lambda = 10

	# Morphology parameters used if build_m1
	# SomaL = 28.896601873591436
	# SomaDiam = 14.187950175330796
	# AxonL = 549.528226526987
	# AxonDiam = 1.0198477329563544

	# Neymotin Reduced
	SomaL = 48.4123467666
	SomaDiam = 28.2149102762
	AxonL = 594.292937602 # 549.528226526987
	AxonDiam =  1.40966286462
	Axon_L_scale = 1 # Used to adjust axon length while maintaing surface area

	# Neuron parameters
	h_celcius = 34 # 37
	h_tstop = 100 # Sim runtime (ms)
	h_dt = 0.1 # Timestep (ms)

	# Current injection
	CI_on = True
	h_i_amplitude: float = -1.0 # (nA)
	h_i_duration = 80 # (ms)
	h_i_delay = 10 # (ms)

	trunk_exc_synapses = True
	perisomatic_exc_synapses = False
	add_soma_inh_synapses = True
	num_soma_inh_syns = 150

	# gmax distributions
	exc_gmax_mean_0 = 0.2
	exc_gmax_std_0 = 0.345
	exc_gmax_clip = (0, 0.65)
	inh_gmax_dist = 1
	soma_gmax_dist = 1
	inh_scalar = 1
	exc_scalar = 1 # Scales weight

	# Synapse density syns/um 
	# Current densities taken from literature on apical main bifurcation, and extrapolated to entire cell.
	exc_synaptic_density = 2.16 # (syn/micron of path length)
	inh_synaptic_density = 0.22 # (syn/micron of path length)
	use_SA_exc = True # Use surface area instead of lengths for the synapse's segment assignment probabilities

	# Release probability distributions
	exc_P_release_mean = 0.53
	exc_P_release_std = 0.22
	inh_basal_P_release_mean = 0.72
	inh_basal_P_release_std = 0.1
	inh_apic_P_release_mean = 0.3
	inh_apic_P_release_std = 0.08
	inh_soma_P_release_mean = 0.88
	inh_soma_P_release_std = 0.05

	# syn_mod
	exc_syn_mod= 'AMPA_NMDA_STP'
	inh_syn_mod = 'GABA_AB_STP'

	# firing rate distributions
	inh_prox_mean_fr = 16.9
	inh_prox_std_fr = 14.3
	inh_distal_mean_fr = 3.9
	inh_distal_std_fr = 4.9
	inh_firing_rate_time_shift = 4

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