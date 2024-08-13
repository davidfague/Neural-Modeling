from dataclasses import dataclass, field
from synapse import CS2CP_syn_params, CP2CP_syn_params, FSI_syn_params, LTS_syn_params

@dataclass
class SimulationParameters:
	
	# Name: required argument
	sim_name: str

	# Random state
	numpy_random_state: int = 130
	neuron_random_state: int = 90

	# Environment parameters
	h_celcius: float = 37 # 34
	h_tstop: int = 5000 # Sim runtime (ms)
	h_dt: float = 0.1 # Timestep (ms)

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
	num_soma_inh_syns: int = 150

	# gmax distributions
	exc_gmax_mean_0: float = 2.3#1.5 # 1.5-1.6 is good
	exc_gmax_std_0: float = 0.2
	exc_gmax_clip: tuple = (0, 15)
	inh_gmax_dist: float = 0.5
	soma_gmax_dist: float = 0.5
	exc_scalar: int = 1 # Scales weight

	# Density/Number of synapses
	# Current densities taken from literature on apical main bifurcation, and extrapolated to entire cell.
	exc_synaptic_density: float = 2.16 # (syn/micron of path length)
	inh_synaptic_density: float = 0.22 # (syn/micron of path length)
	exc_use_density: bool = True # setting to false uses "exc_syn_number" instead of "exc_synaptic_density"
	inh_use_density: bool = True # setting to false uses "inh_syn_number" instead of "inh_synaptic_density"
	exc_syn_number: int = 700
	inh_syn_number: int = 150 
	use_SA_probs: bool = True # Use surface area instead of lengths for the synapse's segment assignment probabilities (does not yet change the total number calculated using density?)

	# Synapse Release probability distributions
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

	# Firing rate distributions
	exc_mean_fr: float = 4.43
	exc_std_fr: float = 2.9
	inh_prox_mean_fr: float = 16.9
	inh_prox_std_fr: float = 14.3
	inh_distal_mean_fr: float = 3.9
	inh_distal_std_fr: float = 4.9

	# syn_params
	exc_syn_params: tuple = (CS2CP_syn_params, CP2CP_syn_params) # 90%, 10%
	inh_syn_params: tuple = (FSI_syn_params, LTS_syn_params)

	# kmeans clustering
	exc_n_FuncGroups: int = 24
	exc_n_PreCells_per_FuncGroup: int = 15
	inh_distributed_n_FuncGroups: int = 24
	inh_distributed_n_PreCells_per_FuncGroup: int = 15

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
	save_every_ms: int = 1000
	path: str = ''

	# Reduction (depracating)
	reduce_cell: bool = False
	expand_cable: bool = False
	reduction_frequency: int = 0
	choose_branches: int = 22
	# Whether or not to optimize the number of segments by lambda after reduction 
	# (may need to add an update to the CellModel class instance's segments list and seg_info list.)
	optimize_nseg_by_lambda: bool = False
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
 
	# new reduction parameters
	reduce_cell_NRCE: bool = False # depracting NRCE
	# reduce_cell_selective:bool = True
	reduce_tufts: bool = False
	reduce_apic: bool = False # cannot do apic with tufts or oblique
	reduce_basals: int = 0 #bool = False
	# reduce_2nd_basals: bool = False
	reduce_obliques: bool = False
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

class HayParameters(SimulationParameters):
	channel_names = [
		'i_pas', 
		'ik', 
		'ica', 
		'ina', 
		'ihcn_Ih', 
		'gNaTa_t_NaTa_t', 
		'ina_NaTa_t', 
		'ina_Nap_Et2', 
		'ik_SKv3_1', 
		'ik_SK_E2', 
		'ik_Im', 
		'ica_Ca_HVA', 
		'ica_Ca_LVAst']
