from cell_builder import SkeletonCell, CellBuilder
from constants import SimulationParameters
from logger import Logger

from neuron import h

import os, datetime
import pickle
import pandas as pd

import numpy as np
import time

class Simulation:

    def __init__(self, cell_type: SkeletonCell, title = None):
        self.cell_type = cell_type
        if title:
          self.title = title
          self.path = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}-{title}"#-%S')}-{title}" # had to remove seconds because with mpi the timing can be slightly off.
        else:
          self.title = None
          self.path = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}-{cell_type}"#-%S')}-{cell_type}"

        self.logger = Logger(None)
        self.pool = []

    def run_single_simulation(self, parameters: SimulationParameters):
        
        parameters.path = os.path.join(parameters.sim_name)

        # Create a folder to save to
        os.mkdir(parameters.path)

        # Build the cell
        cell_builder = CellBuilder(self.cell_type, parameters, self.logger)
        cell, _ = cell_builder.build_cell()
        
        adj_matrix = cell.compute_directed_adjacency_matrix()
        np.savetxt(os.path.join(parameters.path, "adj_matrix.txt"), adj_matrix)

        # Classify segments by morphology, save coordinates
        segments, seg_data = cell.get_segments(["all"]) # (segments is returned here to preserve NEURON references)
        seg_sections = []
        seg_idx = []
        seg_coords = []
        seg_half_seg_RAs = []
        seg = []
        seg_Ls = []
        sec_Ls = []
        sec_Ds = []
        seg_distance = []
        psegs=[]
        
        for i,entry in enumerate(seg_data):
            # if parameters.build_stylized: #@DEPRACATED
            #     sec_name = entry.section.split(".")[-1]
            # else:
            sec_name = entry.section.split(".")[-1] # name[idx]
            #print(f"sec_name: {sec_name}")
            seg_sections.append(sec_name.split("[")[0])
            seg_idx.append(sec_name.split("[")[1].split("]")[0])
            seg_coords.append(entry.coords)
            seg_half_seg_RAs.append(entry.seg_half_seg_RA)
            seg.append(entry.seg)
            seg_Ls.append(entry.L)
            psegs.append(entry.pseg)
            sec_Ls.append(segments[i].sec.L)
            sec_Ds.append(segments[i].sec.diam)
            seg_distance.append(h.distance(segments[0], segments[i]))
            
            
        seg_sections = pd.DataFrame({
            "section": seg_sections, 
            "idx_in_section_type": seg_idx,
            "seg_half_seg_RA": seg_half_seg_RAs,
            "L": seg_Ls,
            "seg":seg,
            "pseg":psegs,
            "Section_L":sec_Ls,
            "Section_diam":sec_Ds,
            "Distance":seg_distance
            })

        seg_coords = pd.concat(seg_coords)

        seg_data = pd.concat((seg_sections.reset_index(drop = True), seg_coords.reset_index(drop = True)), axis = 1)
        seg_data.to_csv(os.path.join(parameters.path, "segment_data.csv"))

        # Compute electrotonic distances from soma
        elec_distances_soma = cell.compute_electrotonic_distance(from_segment = cell.soma[0](0.5))
        elec_distances_soma.to_csv(os.path.join(parameters.path, "elec_distance_soma.csv"))
        
        if not parameters.reduce_apic:
            nexus_seg_index = cell.find_nexus_seg()
        else:
            nexus_seg_index = segments.index(cell.apic[0](0.4)) #New.apic[109](0.386364),New.apic[109](0.431818), chooses seg 0.386. may need to check
                
        if parameters.reduce_soma_gpas:
            cell.soma[0](0.5).g_pas = 10 * cell.soma[0](0.5).g_pas

        # Compute electrotonic distances from nexus
        elec_distances_nexus = cell.compute_electrotonic_distance(from_segment = segments[nexus_seg_index])
        elec_distances_nexus.to_csv(os.path.join(parameters.path, "elec_distance_nexus.csv"))

        # Save constants
        with open(os.path.join(parameters.path, "parameters.pickle"), "wb") as file:
           pickle.dump(parameters, file)

        if parameters.simulate_EPSPs:
          self.simulate_EPSPs(cell, parameters)
        else:
          self.set_all_recorders(cell, parameters)
          self.simulate(cell, parameters)

    def set_all_recorders(self, cell, parameters: SimulationParameters):
        # Set recorders
        if parameters.record_all_channels:
            for var_name in parameters.channel_names:
                cell.add_segment_recorders(var_name = var_name)
        if parameters.record_all_v:
            cell.add_segment_recorders(var_name = "v")
       
        if parameters.record_ecp == True:
    			# Create an ECP object for extracellular potential
    			#elec_pos = params.ELECTRODE_POSITION
    			#ecp = EcpMod(cell, elec_pos, min_distance = params.MIN_DISTANCE)
    			#     # Reason: (NEURON: Impedance calculation with extracellular not implemented)
            self.logger.log_warining("Recording ECP adds the extracellular channel to all segments after computing electrotonic distance.\
                                      This channel is therefore not accounted for in impedence calculation, but it might affect the simulation.")
            h.cvode.use_fast_imem(1)
    			#for sec in cell.all: sec.insert('extracellular') # may not be needed
            cell.add_segment_recorders(var_name = "i_membrane_")
    		
        if (parameters.record_all_synapses):
            for var_name in ["i_AMPA", "i_NMDA"]:
                cell.add_synapse_recorders(var_name = var_name)
        if parameters.record_soma_spikes:
            cell.add_spike_recorder(sec = cell.soma[0], var_name = "soma_spikes", spike_threshold = parameters.spike_threshold)
        if parameters.record_axon_spikes:
            cell.add_spike_recorder(sec = cell.axon[0], var_name = "axon_spikes", spike_threshold = parameters.spike_threshold)

    def set_neuron_parameters(self, parameters):
        h.celsius = parameters.h_celcius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        h.v_init = parameters.h_v_init
        h.finitialize(h.v_init)         
      
    def simulate(self, cell, parameters: SimulationParameters, log=True, record_runtime=True, path=None):
            #@MARK TODO: Can change this to cells a list of cells to record. 
            # This is because if you build cells in a notebook and then simulate them 1 at a time then all cells will get solved for each cell.
            if path is None:
                path = parameters.path
                
            if not os.path.exists(path):
                os.mkdir(path)
                
            with open(os.path.join(parameters.path, "parameters.pickle"), "wb") as file:
                pickle.dump(parameters, file)
            # In time stamps, i.e., ms / dt
            time_step = 0
    
            h.celsius = parameters.h_celcius
            h.tstop = parameters.h_tstop
            h.dt = parameters.h_dt
            h.steps_per_ms = 1 / h.dt
            # if is_indexable(cell.soma):
            #     h.v_init = cell.soma[0].e_pas
            # else:
            #     h.v_init = cell.soma.e_pas
            h.v_init = parameters.h_v_init
            h.finitialize(h.v_init)
    
            if log: self.logger.log("Starting simulation.")
    
            if record_runtime: start_time = time.time()
            while h.t <= h.tstop + 1:
    
                if (time_step > 0) and (time_step % (parameters.save_every_ms / parameters.h_dt) == 0):
                    self.logger.log(f"Saving data at step: {time_step}")
    
                    # Save data
                    cell.write_recorder_data(
                        os.path.join(path, f"saved_at_step_{time_step}"), 
                        parameters.record_every_time_steps)
                    if log: self.logger.log("Finished writing data")
    
                    # Reinitialize recording vectors
                    for recorder_or_list in cell.recorders: recorder_or_list.clear()
                    if log: self.logger.log("Finished clearing recorders")
    
                try:
                    h.fadvance()
                except Exception as e:
                    self.logger.log(f"Error advancing simulation at time_step {time_step}: {e}")
                    break  # Exit the loop on error
                
                time_step += 1
    
            if record_runtime: 
              end_time = time.time()
              simulation_runtime = end_time - start_time
            if log: self.logger.log(f"Finish simulation in {simulation_runtime:.3f} seconds")
            # Record the simulation runtime to a file
            if record_runtime:
              runtime_file_path = os.path.join(path, "simulation_runtime.txt")
              with open(runtime_file_path, "w") as runtime_file:
                  runtime_file.write(f"{simulation_runtime:.3f} seconds")
            #except Exception as e:
            #  self.logger.log(f"Unexpected error in run_single_simulation: {e}")
            
    # def simulate_EPSPs(self, cell, parameters: SimulationParameters):
    #         results = []
    
    #         # Temporarily disable all synapses to ensure a controlled start
    #         for synapse in cell.synapses:
    #             for netcon in synapse.netcons:
    #                 netcon.active(False)
    
    #         # Iterate over each synapse to simulate EPSPs one by one
    #         for idx, synapse in enumerate(cell.synapses):
    #             # Reactivate the current synapse with a spike at 50
    #             spike_train = [50]
    #             vec = h.Vector(spike_train)
    #             stim = h.VecStim()
    #             stim.play(vec)
    #             synapse.netcons.append(h.NetCon(stim, synapse.h_syn, 1, 0, 1))
    #             path = os.path.join(parameters.path, f"synapse_{idx}")
               
    #             # Record data
    #             synapse_type = synapse.syn_mod
    #             segment_name = str(synapse.h_syn.get_segment())
    #             cell.add_segment_recorders(var_name='v', segment_to_record=cell.soma[0](0.5))
    #             cell.add_segment_recorders(var_name='v', segment_to_record=synapse.h_syn.get_segment())
    #             distance = h.distance(synapse.h_syn.get_segment(), cell.soma[0](0.5))
    #             results.append({"Synapse Index": idx, "Synapse Type": synapse_type,"Segment Name": segment_name, "Distance from Soma": distance})
            
    #             # Simulate
    #             self.simulate(cell, parameters, path=parameters.path+str(idx))
        
    #             cell.recorders = []

    #             # Deactivate the synapse after simulation
    #             for netcon in synapse.netcons:
    #                 netcon.active(False)

    #         # Convert results to a DataFrame
    #         df_results = pd.DataFrame(results)
    
    #         # Output DataFrame to a CSV file or other storage as needed
    #         df_results.to_csv(os.path.join(path, 'EPSP_simulation_results.csv'), index=False)
    
    #         return df_results
    
    

def is_indexable(obj: object):
    """
    Check if the object is indexable.
    """
    try:
        _ = obj[0]
        return True
    except:
        return False
    

