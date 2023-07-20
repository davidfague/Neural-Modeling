# Random state
numpy_random_states = [123]
neuron_random_states = [1] # Number of calls to MCellRan4()

# Modfiles
modfiles_folder = "../modfiles"

# Reduction
# reduce_cell = False
# expand_cable = False
# choose_branches = 22

# Complex cell
complex_cell_folder = '../complex_cells/L5PC/'
complex_cell_biophys_hoc_name = 'L5PCbiophys3ActiveBasal.hoc'

# Neuron parameters
h_celcius = 37
h_tstop = 1000 # Sim runtime (ms)
h_dt = 0.1 # Timestep (ms)

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

# Log, plot and save
save_dir = "output"
log_every_ms = 500
save_every_ms = 200
