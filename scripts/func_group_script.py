import sys
sys.path.append("../")

from Modules.synapse_generator import SynapseGenerator
from Modules.cell_model import CellModel
from Modules.spike_generator import SpikeGenerator
from Modules.complex_cell import build_L5_cell
from Modules.functional_group import generate_excitatory_functional_groups, generate_inhibitory_functional_groups
from Modules.cell_utils import get_segments_and_len_per_segment
from Modules.plotting_utils import plot_simulation_results

from neuron import h

from cell_inference.config import params
from cell_inference.utils.currents.ecp import EcpMod
from cell_inference.utils.currents.recorder import Recorder
from cell_inference.utils.plotting.plot_results import plot_lfp_heatmap, plot_lfp_traces

import numpy as np
from functools import partial
import scipy.stats as st
import time, datetime
import os

import func_group_sim_constants as constants

def main(numpy_random_state, neuron_random_state):

    # Random seed
    random_state = np.random.RandomState(numpy_random_state)
    neuron_r = h.Random()

    for _ in range(neuron_random_state):
        neuron_r.MCellRan4()

    # Time vector for generating inputs
    t = np.arange(0, constants.h_tstop, 1)

    # Build cell
    complex_cell = build_L5_cell(constants.complex_cell_folder, constants.complex_cell_biophys_hoc_name)

    h.celsius = constants.h_celcius
    h.v_init = complex_cell.soma[0].e_pas

    # Sim runtime
    h.tstop = constants.h_tstop

    # Timestep (ms)
    h.dt = constants.h_dt
    h.steps_per_ms = 1 / h.dt

    # Measure time
    runtime_start_time = time.time()

    # Get segments and lengths
    all_segments, all_len_per_segment, all_SA_per_segment,\
    all_segments_center, soma_segments, soma_len_per_segment,\
    soma_SA_per_segment, soma_segments_center, no_soma_segments,\
    no_soma_len_per_segment, no_soma_SA_per_segment, no_soma_segments_center =\
    get_segments_and_len_per_segment(complex_cell)

    # ---- Excitatory

    # Excitatory gmax distribution
    exc_gmax_mean_0 = constants.exc_gmax_mean_0
    exc_gmax_std_0 = constants.exc_gmax_std_0

    gmax_mean = np.log(exc_gmax_mean_0) - 0.5 * np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1)
    gmax_std = np.sqrt(np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1))

    # gmax distribution
    def log_norm_dist(gmax_mean, gmax_std, size):
        val = np.random.lognormal(gmax_mean, gmax_std, size)
        s = float(np.clip(val, constants.exc_gmax_clip[0], constants.exc_gmax_clip[1]))
        return s

    gmax_exc_dist = partial(log_norm_dist, gmax_mean, gmax_std, size = 1)

    # Excitatory firing rate distribution
    def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
        return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, 
                                         loc = loc, scale = scale, size = size)) + 1e-15
    
    spike_generator = SpikeGenerator()
    synapse_generator = SynapseGenerator()

    exc_number_of_groups = int(sum(all_len_per_segment) / constants.exc_functional_group_span)

    # Number of presynaptic cells
    cells_per_group = int(constants.exc_functional_group_span * constants.exc_synaptic_density / constants.exc_synapses_per_cluster)

    # Distribution of mean firing rates
    mean_fr_dist = partial(exp_levy_dist, alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1)

    # New list to change probabilty of exc functional group nearing soma
    adjusted_no_soma_len_per_segment = []
    for i, seg in enumerate(no_soma_segments):
        if h.distance(seg, complex_cell.soma[0](0.5)) < 75:
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i] / 10)
        else:
            adjusted_no_soma_len_per_segment.append(no_soma_len_per_segment[i])

    exc_functional_groups = generate_excitatory_functional_groups(all_segments = no_soma_segments,
                                                                  all_len_per_segment = no_soma_len_per_segment,
                                                                  all_segments_centers = no_soma_segments_center,
                                                                  number_of_groups = exc_number_of_groups,
                                                                  cells_per_group = cells_per_group,
                                                                  synapses_per_cluster = constants.exc_synapses_per_cluster,
                                                                  functional_group_span = constants.exc_functional_group_span,
                                                                  cluster_span = constants.exc_cluster_span,
                                                                  gmax_dist = gmax_exc_dist,
                                                                  mean_fr_dist = mean_fr_dist,
                                                                  spike_generator = spike_generator,
                                                                  synapse_generator = synapse_generator,
                                                                  t = t, random_state = random_state,
                                                                  neuron_r = neuron_r,
                                                                  record = True, syn_mod = 'AMPA_NMDA')
    
    exc_spikes = spike_generator.spike_trains

    # ---- Inhibitory

    inh_number_of_clusters = int(sum(all_len_per_segment) / constants.inh_cluster_span)
    inh_synapses_per_cluster = int(constants.inh_cluster_span * constants.inh_synaptic_density)

    # Proximal inh mean_fr distribution
    mean_fr, std_fr = constants.inh_prox_mean_fr, constants.inh_prox_std_fr
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    # Distal inh mean_fr distribution
    mean_fr, std_fr = constants.inh_distal_mean_fr, constants.inh_distal_std_fr
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    inhibitory_functional_groups = generate_inhibitory_functional_groups(cell = complex_cell,
                                                                         all_segments = all_segments,
                                                                         all_len_per_segment = all_len_per_segment,
                                                                         all_segments_centers = all_segments_center,
                                                                         number_of_groups = 1,
                                                                         cells_per_group = inh_number_of_clusters,
                                                                         synapses_per_cluster = inh_synapses_per_cluster,
                                                                         functional_group_span = constants.inh_functional_group_span,
                                                                         cluster_span = constants.inh_cluster_span,
                                                                         gmax_dist = constants.inh_gmax_dist,
                                                                         proximal_inh_dist = proximal_inh_dist,
                                                                         distal_inh_dist = distal_inh_dist,
                                                                         spike_generator = spike_generator,
                                                                         synapse_generator = synapse_generator,
                                                                         t = t, f_group_name_prefix = "diffuse_inh_",
                                                                         random_state = random_state, neuron_r = neuron_r,
                                                                         spike_trains_to_delay = exc_spikes, 
                                                                         fr_time_shift = constants.inh_firing_rate_time_shift,
                                                                         record = True, syn_mod = 'GABA_AB')

    # ---- Soma

    soma_inhibitory_functional_groups = generate_inhibitory_functional_groups(cell = complex_cell,
                                                                              all_segments = soma_segments,
                                                                              all_len_per_segment = soma_SA_per_segment,
                                                                              all_segments_centers = soma_segments_center,
                                                                              number_of_groups = constants.soma_number_of_groups,
                                                                              cells_per_group = constants.soma_number_of_clusters,
                                                                              synapses_per_cluster = constants.soma_synapses_per_cluster,
                                                                              functional_group_span = constants.soma_functional_group_span,
                                                                              cluster_span = constants.soma_cluster_span,
                                                                              gmax_dist = constants.soma_gmax_dist,
                                                                              proximal_inh_dist = proximal_inh_dist,
                                                                              distal_inh_dist = distal_inh_dist,
                                                                              spike_generator = spike_generator,
                                                                              synapse_generator = synapse_generator,
                                                                              t = t, f_group_name_prefix = "soma_inh_",
                                                                              random_state = random_state, neuron_r = neuron_r,
                                                                              spike_trains_to_delay = exc_spikes, 
                                                                              fr_time_shift = constants.inh_firing_rate_time_shift,
                                                                              record = True, syn_mod = 'GABA_AB')
    # ---- Set up a cell model

    # Get all synapses
    all_syns = []
    for synapse_list in synapse_generator.synapses: # synapse_generator.synapses is a list of synapse lists
        for synapse in synapse_list:
            all_syns.append(synapse)

    cell = CellModel(hoc_model = complex_cell, synapses = all_syns,
                    netcons = spike_generator.netcons, spike_trains = spike_generator.spike_trains,
                    spike_threshold = constants.spike_threshold, random_state = random_state,
                    var_names = constants.channel_names)
    
    # Reduce cell and store new things: can update CellModel module to do this stuff

    def find_distal_sections(cell, region=str):
        '''
        Finds all terminal sections then gathers terminal apical sections that are greater than 800 microns from the soma in path length
        '''
        # find distal tuft sections:
        parent_sections=[]
        for sec in cell.all: # find non-terminal sections
            if sec.parentseg() is not None:
                if sec.parentseg().sec not in parent_sections:
                    parent_sections.append(sec.parentseg().sec)
        terminal_sections=[]
        for sec in getattr(cell,region):  # check if the section is a terminal section and if it is apical tuft
            # print(h.distance(sec(0.5)))
            if region=='apic':
                if (sec not in parent_sections) and (h.distance(cell.soma[0](0.5),sec(0.5)) > 800):
                    terminal_sections.append(sec)
            else:
                if (sec not in parent_sections):
                    terminal_sections.append(sec)

                # print(sec, 'is a terminal section of the tuft'

        return terminal_sections

    # ---- Prepare simulation
    basals = find_distal_sections(cell, 'dend')
    tufts = find_distal_sections(cell, 'apic')

    # Find segments of interest
    soma_seg_index = cell.segments.index(cell.soma[0](0.5))
    axon_seg_index = cell.segments.index(cell.axon[-1](0.9))
    basal_seg_index = cell.segments.index(basals[0](0.5))
    trunk_seg_index = cell.segments.index(cell.apic[0](0.999))

    # Find tuft and nexus
    tuft_seg_index = cell.segments.index(tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
    nexus_seg_index = cell.segments.index(cell.apic[36](0.961538))

    # Compute electrotonic distances from nexus
    cell.recompute_segment_elec_distance(segment = cell.segments[nexus_seg_index], seg_name = "nexus")

    # Record time points
    t_vec = h.Vector(round(h.tstop / h.dt) + 1).record(h._ref_t)

    # Record membrane voltage of all segments
    V_rec = Recorder(cell.segments)

    elec_pos = params.ELECTRODE_POSITION
    ecp = EcpMod(cell, elec_pos, min_distance=params.MIN_DISTANCE)  # create an ECP object for extracellular potential

    # ---- Run simulation
    sim_duration = h.tstop / 1000 # Convert from ms to s
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          f"Running Simulation, duration: {sim_duration} sec.")

    sim_start_time = time.time()
    h.run()
    sim_end_time = time.time()

    elapsedtime = sim_end_time - sim_start_time
    print(f'Simulation time: {round(elapsedtime)} sec.')
    total_runtime = sim_end_time - runtime_start_time
    print(f'Total runtime: {round(total_runtime)} sec.')

    # Time array (ms)
    t = t_vec.as_numpy().copy()  

    # Soma membrane potential
    Vm = V_rec.as_numpy()

    loc_param = [0., 0., 45., 0., 1., 0.]

    # LFP array
    lfp = ecp.calc_ecp(move_cell=loc_param).T  # unit: mV


    # Save data
    random_seed_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_seeds_" +\
                       str(numpy_random_state) + "_" + str(neuron_random_state)
    save_folder = os.path.join(constants.save_dir, random_seed_name)
    os.mkdir(save_folder)

    plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, nexus_seg_index, trunk_seg_index,
                            loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, vlim = [-0.023,0.023],
                            show = False, save_dir = save_folder)
    
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": Saving simulation data.")

    save_start_time = time.time()
    cell.generate_recorder_data()
    
    cell.write_data(save_folder)
    save_end_time = time.time()

    elapsedtime = save_end_time - save_start_time
    print(f'Save time: {round(elapsedtime)} sec.')
    total_runtime = save_end_time - runtime_start_time
    print(f'Total runtime incl. saving: {round(total_runtime)} sec.')

if __name__ == "__main__":

    # Sanity checks
    if os.path.exists('x86_64'):
        raise FileExistsError("Delete x86_64 folder.")
    
    if not os.path.exists(constants.save_dir):
        raise FileNotFoundError("No save folder with the given name.")

    # Compile and load modfiles
    os.system(f"nrnivmodl {constants.modfiles_folder}")
    h.load_file('stdrun.hoc')
    h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')

    for np_state in constants.numpy_random_states:
        for neuron_state in constants.neuron_random_states:
            print(f"Running for seeds ({np_state}, {neuron_state})...")
            main(np_state, neuron_state)

    
