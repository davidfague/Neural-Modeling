# Random state
numpy_random_states = [123]
neuron_random_states = [87] # Number of calls to MCellRan4()

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
h_tstop = 500 #55#2500#20400 # Sim runtime (ms)
h_dt = 0.1 # Timestep (ms)

# Current injection
h_i_amplitudes = [None] #[-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0] # CI amplitudes (nA); to disable external injection, set to [None] (also disables h_i params below)
h_i_duration = 5000 # (ms)
h_i_delay = 400 # (ms)

# Excitatory dend
exc_gmax_mean_0 = 0.2
exc_gmax_std_0 = 0.345
exc_gmax_clip = (0, 0.7)
exc_synaptic_density = 2.12
exc_functional_group_span = 100
exc_cluster_span = 10
exc_synapses_per_cluster = 5

# Inhibitory dend
inh_gmax_dist = 2.25
inh_synaptic_density = 0.22
inh_cluster_span = 10
inh_prox_mean_fr = 16.9
inh_prox_std_fr = 14.3
inh_distal_mean_fr = 3.9
inh_distal_std_fr = 4.9
inh_number_of_groups = 1
inh_functional_group_span = 100
inh_firing_rate_time_shift = 4

# Inhibitory soma
soma_gmax_dist = 2.25
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
