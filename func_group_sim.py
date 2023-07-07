from Modules.synapse_generator import SynapseGenerator
from Modules.reduction import Reductor
from Modules.cell_model import CellModel
from Modules.spike_generator import SpikeGenerator
from Modules.complex_cell import build_L5_cell
from Modules.functional_group import generate_excitatory_functional_groups, generate_inhibitory_functional_groups
from Modules.cell_utils import get_segments_and_len_per_segment

import numpy as np
from functools import partial
import scipy.stats as st
import time
import datetime
import os

from neuron import h

import ipywidgets as widgets
from ipywidgets import interactive_output, HBox, VBox, Layout
import matplotlib.pyplot as plt
from Modules.plotting_utils import plot_morphology, plot_simulation_results, plot_LFP_Vm_currents

from cell_inference.config import params, paths
from cell_inference.cells.activecell_axon import ReducedOrderL5Cell, ReducedOrderL5CellPassiveDendrite
from cell_inference.utils.currents.ecp import EcpMod
from cell_inference.utils.currents.recorder import Recorder
from cell_inference.utils.plotting.plot_results import plot_lfp_heatmap, plot_lfp_traces
from cell_inference.utils.plotting.plot_variable_with_morphology import plot_variable_with_morphology
from cell_inference.utils.metrics.measure_segment_distance import measure_segment_distance

if __name__ == "__main__":

    # Random seed
    random_state = np.random.RandomState(123)
    neuron_r = h.Random()
    neuron_r.MCellRan4()

    # Global vars
    reduce_cell = False
    expand_cable = False
    choose_branches = 22

    complex_cell_folder = 'complex_cells/L5PC/'

    # Simulation params
    h_celcius = 37
    h_tstop = 100 # Sim runtime
    h_dt = 0.1 # Timestep (ms)

    # Time vector for generating inputs
    t = np.arange(0, h_tstop, 1)

    # Compile and load modfiles
    os.system("nrnivmodl modfiles")
    h.load_file('stdrun.hoc')
    h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')

    # Build cell
    complex_cell = build_L5_cell(complex_cell_folder,'L5PCbiophys3ActiveBasal.hoc')

    h.celsius = h_celcius
    h.v_init = complex_cell.soma[0].e_pas

    # Sim runtime
    h.tstop = h_tstop

    # Timestep (ms)
    h.dt = h_dt
    h.steps_per_ms = 1 / h.dt

    runtime_start_time = time.time()

    all_segments, all_len_per_segment, all_SA_per_segment,\
    all_segments_center, soma_segments, soma_len_per_segment,\
    soma_SA_per_segment, soma_segments_center, no_soma_segments,\
    no_soma_len_per_segment, no_soma_SA_per_segment, no_soma_segments_center = get_segments_and_len_per_segment(complex_cell)

    # Excitatory gmax distribution
    exc_gmax_mean_0 = 0.2
    exc_gmax_std_0 = 0.345

    gmax_mean = np.log(exc_gmax_mean_0) - 0.5 * np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1)
    gmax_std = np.sqrt(np.log((exc_gmax_std_0 / exc_gmax_mean_0) ** 2 + 1))

    # gmax distribution
    def log_norm_dist(gmax_mean, gmax_std, size):
        val = np.random.lognormal(gmax_mean, gmax_std, size)
        s = float(np.clip(val, 0, 0.7))
        return s

    gmax_exc_dist = partial(log_norm_dist, gmax_mean, gmax_std, size = 1)

    # Excitatory firing rate distribution
    def exp_levy_dist(alpha = 1.37, beta = -1.00, loc = 0.92, scale = 0.44, size = 1):
        return np.exp(st.levy_stable.rvs(alpha = alpha, beta = beta, loc = loc, scale = scale, size = size)) + 1e-15
    
    spike_generator = SpikeGenerator()
    synapse_generator = SynapseGenerator()

    synaptic_density = 2.12
    functional_group_span = 100
    cluster_span = 10
    synapses_per_cluster = 5

    number_of_groups = int(sum(all_len_per_segment) / functional_group_span)

    # Number of presynaptic cells
    cells_per_group = int(functional_group_span*synaptic_density / synapses_per_cluster)

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
                                                                  number_of_groups = number_of_groups,
                                                                  cells_per_group = cells_per_group,
                                                                  synapses_per_cluster = synapses_per_cluster,
                                                                  functional_group_span = functional_group_span,
                                                                  cluster_span = cluster_span,
                                                                  gmax_dist = gmax_exc_dist,
                                                                  mean_fr_dist = mean_fr_dist,
                                                                  spike_generator = spike_generator,
                                                                  synapse_generator = synapse_generator,
                                                                  t = t, random_state = random_state,
                                                                  neuron_r = neuron_r,
                                                                  record = True, syn_mod = 'AMPA_NMDA')
    
    exc_spikes = spike_generator.spike_trains

    synaptic_density = 0.22
    cluster_span = 10

    synapses = synaptic_density * sum(all_len_per_segment)
    number_of_clusters = int(sum(all_len_per_segment) / cluster_span)
    synapses_per_cluster = int(cluster_span * synaptic_density) # 12 # synapses/number_of_clusters
    # synapses_per_node = 5
    # number_of_clusters=int(synapses / (cluster_span * synaptic_density)) # number of presynaptic cells

    # Proximal inh mean_fr distribution
    mean_fr, std_fr = 16.9, 14.3
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    proximal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    # Distal inh mean_fr distribution
    mean_fr, std_fr = 3.9, 4.9
    a, b = (0 - mean_fr) / std_fr, (100 - mean_fr) / std_fr
    distal_inh_dist = partial(st.truncnorm.rvs, a = a, b = b, loc = mean_fr, scale = std_fr)

    gmax_inh_dist = 2.25 #1.65 # need inh gmax distribution}
    number_of_groups = 1

    inhibitory_functional_groups = generate_inhibitory_functional_groups(cell = complex_cell,
                                                                        all_segments = all_segments,
                                                                        all_len_per_segment = all_len_per_segment,
                                                                        all_segments_centers = all_segments_center,
                                                                        number_of_groups = 1,
                                                                        cells_per_group = number_of_clusters,
                                                                        synapses_per_cluster = synapses_per_cluster,
                                                                        functional_group_span = functional_group_span,
                                                                        cluster_span = cluster_span,
                                                                        gmax_dist = gmax_inh_dist,
                                                                        proximal_inh_dist = proximal_inh_dist,
                                                                        distal_inh_dist = distal_inh_dist,
                                                                        spike_generator = spike_generator,
                                                                        synapse_generator = synapse_generator,
                                                                        t = t, f_group_name_prefix = "diffuse_inh_",
                                                                        random_state = random_state, neuron_r = neuron_r,
                                                                        spike_trains_to_delay = exc_spikes, fr_time_shift = 4,
                                                                        record = True, syn_mod = 'GABA_AB')

    # 150 soma inh synapses
    number_of_clusters = 15 # Number of presynaptic cells
    cluster_span = 10
    synapses_per_cluster = 10 # number of synapses per presynaptic cell

    soma_inhibitory_functional_groups = generate_inhibitory_functional_groups(cell = complex_cell,
                                                                            all_segments = soma_segments,
                                                                            all_len_per_segment = soma_SA_per_segment,
                                                                            all_segments_centers = soma_segments_center,
                                                                            number_of_groups = 1,
                                                                            cells_per_group = number_of_clusters,
                                                                            synapses_per_cluster = synapses_per_cluster,
                                                                            functional_group_span = functional_group_span,
                                                                            cluster_span = cluster_span,
                                                                            gmax_dist = gmax_inh_dist,
                                                                            proximal_inh_dist = proximal_inh_dist,
                                                                            distal_inh_dist = distal_inh_dist,
                                                                            spike_generator = spike_generator,
                                                                            synapse_generator = synapse_generator,
                                                                            t = t, f_group_name_prefix = "soma_inh_",
                                                                            random_state = random_state, neuron_r = neuron_r,
                                                                            spike_trains_to_delay = exc_spikes, fr_time_shift = 4,
                                                                            record = True, syn_mod = 'GABA_AB')
    
    all_syns = []
    for synapse_list in synapse_generator.synapses: # synapse_generator.synapses is a list of synapse lists
        for synapse in synapse_list:
            all_syns.append(synapse)

    cell = CellModel(hoc_model = complex_cell, synapses = all_syns,
                    netcons = spike_generator.netcons, spike_trains = spike_generator.spike_trains,
                    spike_threshold = 10, random_state = random_state,
                    var_names = ['gNaTa_t_NaTa_t', 'ina_NaTa_t', 'gNap_Et2_Nap_Et2', 'ina_Nap_Et2',
                                'ik_K_Pst', 'ik_K_Tst', 'ik_SK_E2', 'ik_SKv3_1',
                                'ica_Ca_HVA', 'ica_Ca_LVAst', 'ihcn_Ih', 'i_pas'])
    
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

    #reduce cell
    if reduce_cell==True:
        #reduce complex dendritic trees to cables
        reduced_cell, synapses_list, netcons_list, txt_nr = subtree_reductor(complex_cell, synapses_list, netcons_list, reduction_frequency=0,return_seg_to_seg=True)
        print("synapses_list after NR reduction:", synapses_list)
        if expand_cable==True:
            #expand cable to idealized dendritic trees
            sections_to_expand = [reduced_cell.hoc_model.apic[0]]
            furcations_x=[0.289004]
            nbranches=[choose_branches]
            reduced_dendritic_cell, synapses_list, netcons_list, txt_ce = cable_expander(reduced_cell, sections_to_expand, furcations_x, nbranches,
                                                                                                                                                                synapses_list, netcons_list, reduction_frequency=0,return_seg_to_seg=True)
            #remove basal dend 3d coordinates because they seem off
            for sec in reduced_dendritic_cell.dend:
                sec.pt3dclear()

            cell = CellModel(reduced_dendritic_cell,synapses_list=synapses_list,netcons_list=netcons_list,spike_threshold = 10) #NR model with apical cable converted to tree
            tufts=find_distal_sections(cell, 'apic')
            basals=find_distal_sections(cell, 'dend')
            cell._nbranch=len(tufts)

            print(cell._nbranch, "terminal tuft branches in reduced_dendritic_cell")
        else:
            #workaround since cell.all was not attribute (can update cell_model class to include this list formation)
            reduced_cell.all =[]
            for sec in [reduced_cell.soma]:
                reduced_cell.all.append(sec)
            for sec in [reduced_cell.apic]:
                reduced_cell.all.append(sec)
            for sec in reduced_cell.dend:
                reduced_cell.all.append(sec)
            for sec in reduced_cell.axon:
                reduced_cell.all.append(sec)
            #make apic sec a list (can update cell_model class to include this) (cell_model class expects cell.apic to be iterable)
            reduced_cell.apic=[reduced_cell.apic]
            #use cell_model python class
            cell=CellModel(reduced_cell,synapses_list=synapses_list,netcons_list=netcons_list,spike_threshold = 10) # neuron_reduce model
            tufts=cell.apic # Vm plot may return middle segment of trunk since it is the distal apical section.
            basals=find_distal_sections(cell, 'dend')
            cell._nbranch=len(tufts)
            print(cell._nbranch, "terminal tuft branches in reduced_cell")
    else:
        basals=find_distal_sections(cell, 'dend')
        tufts=find_distal_sections(cell, 'apic')
        # cell = cell_model(complex_cell,synapses_list=synapses_list,netcons_list=netcons_list,spike_threshold = 10) # original cell
        # original cell should already be defined
        pass

    print(len(cell.all))
    print(len(cell.segments))

    # find segments of interest
    soma_seg_index = cell.segments.index(cell.soma[0](0.5))
    axon_seg_index = cell.segments.index(cell.axon[-1](0.9))
    basal_seg_index = cell.segments.index(basals[0](0.5))
    trunk_seg_index = cell.segments.index(cell.apic[0](0.999))
    # find tuft and nexus
    if (reduce_cell == True) and (expand_cable == True): # Dendritic reduced model
        tuft_seg_index = tuft_seg_index=cell.segments.index(tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
        nexus_seg_index = cell.segments.index(cell.apic[0](0.99))
    elif (reduce_cell == True) and (expand_cable == False): # NR model
        tuft_seg_index = cell.segments.index(tufts[0](0.9)) # tufts[0] will be the cable that is both trunk and tuft in this case, so we have to specify near end of cable
        nexus_seg_index = cell.segments.index(cell.apic[0](0.289004))
    else: # Complex cell
        tuft_seg_index=cell.segments.index(tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
        nexus_seg_index=cell.segments.index(cell.apic[36](0.961538))
    # compute electrotonic distances from nexus
    cell.recompute_segment_elec_distance(segment = cell.segments[nexus_seg_index], seg_name = "nexus")

    # Record time points
    t_vec = h.Vector(round(h.tstop / h.dt) + 1).record(h._ref_t)

    # Record membrane voltage of all segments
    V_rec = Recorder(cell.segments)

    elec_pos = params.ELECTRODE_POSITION
    ecp = EcpMod(cell, elec_pos, min_distance=params.MIN_DISTANCE)  # create an ECP object for extracellular potential

    # coordinates (x, y, z) of electrodes
    # print(params.ELECTRODE_POSITION.shape)
    # print(params.ELECTRODE_POSITION)

    # Run simulation
    sim_duration = h.tstop / 1000 #convert from ms to s
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Running Simulation for ",sim_duration,"sec")

    sim_start_time = time.time()
    h.run()
    sim_end_time = time.time()

    elapsedtime = sim_end_time - sim_start_time
    print('It took',round(elapsedtime),'sec to run a',sim_duration,'sec simulation.')
    total_runtime = sim_end_time - runtime_start_time
    print('The total runtime was',round(total_runtime),'sec')

    # Get results
    t = t_vec.as_numpy().copy()  # time array (ms)
    # Soma membrane potential
    Vm = V_rec.as_numpy()

    loc_param = [0., 0., 45., 0., 1., 0.]

    soma_seg_index = cell.segments.index(cell.soma[0](0.5))
    axon_seg_index = cell.segments.index(cell.axon[-1](0.9))
    basal_seg_index = cell.segments.index(basals[0](0.5))
    trunk_seg_index = cell.segments.index(cell.apic[0](0.999))

    # Choose a tuft to plot voltage
    if (reduce_cell == True) and (expand_cable == True): # Dendritic reduced model
        tuft_seg_index = tuft_seg_index=cell.segments.index(tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
        nexus_seg_index = cell.segments.index(cell.apic[0](0.99))
    elif (reduce_cell == True) and (expand_cable == False): # NR model
        tuft_seg_index = cell.segments.index(tufts[0](0.9)) # tufts[0] will be the cable that is both trunk and tuft in this case, so we have to specify near end of cable
        nexus_seg_index = cell.segments.index(cell.apic[0](0.289004))
    else: # Complex cell
        tuft_seg_index=cell.segments.index(tufts[0](0.5)) # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
        nexus_seg_index=cell.segments.index(cell.apic[36](0.961538))

    print("Plotting",cell.segments[soma_seg_index],"Voltage| y coordinate of this seg:", cell.seg_coords['pc'][soma_seg_index][1])
    print("Plotting",cell.segments[tuft_seg_index],"Voltage| y coordinate of this seg:", cell.seg_coords['pc'][tuft_seg_index][1])
    print("Plotting",cell.segments[nexus_seg_index],"Voltage| y coordinate of this seg:", cell.seg_coords['pc'][nexus_seg_index][1])
    print("Plotting",cell.segments[basal_seg_index],"Voltage| y coordinate of this seg:", cell.seg_coords['pc'][basal_seg_index][1])
    print("Plotting",cell.segments[axon_seg_index],"Voltage| y coordinate of this seg:", cell.seg_coords['pc'][axon_seg_index][1])
    print("Plotting",cell.segments[trunk_seg_index],"Voltage| y coordinate of this seg:", cell.seg_coords['pc'][trunk_seg_index][1])

    # lfp array
    lfp = ecp.calc_ecp(move_cell=loc_param).T  # unit: mV

    # plot membrane voltage at given segments and plot lfp from extracellular probe
    plot_simulation_results(t, Vm, soma_seg_index, axon_seg_index, basal_seg_index, tuft_seg_index, nexus_seg_index, trunk_seg_index,
                            loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, vlim = [-0.023,0.023])
    

    
    # Save data
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"Saving Simulation data")

    save_start_time = time.time()
    data_dict=cell.get_recorder_data() #also creates directory # update to remove folder first
    save_end_time = time.time()

    elapsedtime = save_end_time - save_start_time
    print('It took',round(elapsedtime),'sec to save simulation')
    total_runtime = save_end_time - runtime_start_time
    print('The total runtime was',round(total_runtime),'sec')

    cell.write_seg_info_to_csv()

    
