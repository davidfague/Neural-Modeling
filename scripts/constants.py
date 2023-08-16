# Random state
numpy_random_states = [123]
neuron_random_states = [87] # Number of calls to MCellRan4()

# Runtime
parallelize = False

# Modfiles
modfiles_folder = "../modfiles"

# Reduction
reduce_cell = False
expand_cable = False
reduction_frequency = 0
choose_branches = 22
optimize_nseg_by_lambda = True # Whether or not to optimize the number of segments by lambda after reduction (may need to add an update to the CellModel class instance's segments list and seg_info list.)
merge_synapses = False # Whether or not to merge synapses after optimizing nseg by lambda. (synapses should already be merged by the reduce_cell_func, but could be merged again if optimize_nseg_by_lambda lowers nseg.)
segs_per_lambda = 10 # Desired number of segs per length constant

# Complex cell
complex_cell_folder = '../complex_cells/L5PC/'
complex_cell_biophys_hoc_name = 'L5PCbiophys3ActiveBasal.hoc'

# Neuron parameters
h_celcius = 37
h_tstop = 2000 #55#2500#20400 # Sim runtime (ms)
h_dt = 0.1 # Timestep (ms)

# Current injection
h_i_amplitudes = [None] #[-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0] # CI amplitudes (nA); to disable external injection, set to [None] (also disables h_i params below)
h_i_duration = 5000 # (ms)
h_i_delay = 400 # (ms)

# gmax distributions
exc_gmax_mean_0 = 0.2
exc_gmax_std_0 = 0.345
exc_gmax_clip = (0,0.65)#(0, 0.7)
inh_gmax_dist = 1#2.25
soma_gmax_dist = 1#2.25
inh_scalar = 2.25 # scales weight
exc_scalar = 0.75 # scales weight

# synapse density syns/um
exc_synaptic_density = 2.16
inh_synaptic_density = 0.22

# release probability distributions
exc_P_release_mean = 0.53
exc_P_release_std = 0.22
inh_basal_P_release_mean = 0.72
inh_basal_P_release_std = 0.1
inh_apic_P_release_mean = 0.3
inh_apic_P_release_std = 0.08
inh_soma_P_release_mean = 0.88
inh_soma_P_release_std = 0.05

# syn_mod
exc_syn_mod= 'AMPA_NMDA_STP' #'pyr2pyr'
inh_syn_mod = 'GABA_AB_STP' # 'int2pyr'

# firing rate distributions
inh_prox_mean_fr = 16.9
inh_prox_std_fr = 14.3
inh_distal_mean_fr = 3.9
inh_distal_std_fr = 4.9
inh_firing_rate_time_shift = 4

# syn parameters # not yet implemented. Can also implement area related synapse density, and increase membrane capacitance for exc spines.
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
exc_syn_params=[CS2CP_syn_params,CP2CP_syn_params]

# kmeans clustering
exc_n_FuncGroups = 24
exc_n_PreCells_per_FuncGroup = 100
inh_distributed_n_FuncGroups = 5
inh_distributed_n_PreCells_per_FuncGroup = 50

# Excitatory dend
exc_functional_group_span = 100
exc_cluster_span = 10
exc_synapses_per_cluster = 5
trunk_exc_synapses = False # on/off switch
perisomatic_exc_synapses = False

# Inhibitory dend
inh_cluster_span = 10
inh_number_of_groups = 1
inh_functional_group_span = 100

# Inhibitory soma
soma_number_of_clusters = 15 # Number of presynaptic cells
soma_cluster_span = 10
soma_synapses_per_cluster = 10 # Number of synapses per presynaptic cell
soma_number_of_groups = 1
soma_functional_group_span = 100

# Cell model
spike_threshold = 10
channel_names = ['gNaTa_t_NaTa_t', 'ina_NaTa_t', 'gNap_Et2_Nap_Et2', 'ina_Nap_Et2',
                 'ik_K_Pst', 'ik_K_Tst', 'ik_SK_E2', 'ik_SKv3_1', 'ica_Ca_HVA', 
                 'ica_Ca_LVAst', 'ihcn_Ih', 'i_pas']

# Tiesinga
ties_a_iv = 10
ties_P = 1
ties_CV_t = 1
ties_sigma_iv = 1
ties_pad_aiv = 0

# Post Synaptic Current analysis
number_of_presynaptic_cells = 6524
PSC_start = 5

# Log, plot and save
save_dir = "output"
log_every_ms = 500
save_every_ms = 500
