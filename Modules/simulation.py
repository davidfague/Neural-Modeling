from cell_builder import SkeletonCell, CellBuilder
from constants import SimulationParameters
from logger import Logger

# from cell_inference.config import params
# from cell_inference.utils.currents.ecp import EcpMod

from neuron import h

import os, datetime
import pickle, h5py
import pandas as pd
import numpy as np

from multiprocessing import Pool, cpu_count

# https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-clas
def unwrap_self_run_single_simulation(args):
    return Simulation.run_single_simulation(args[0], args[1])

class Simulation:

    def __init__(self, cell_type: SkeletonCell, title = None):
        self.cell_type = cell_type
        if title:
          self.path = f"{title}-{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
        else:
          self.path = f"{cell_type}-{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"

        self.logger = Logger(None)
        self.pool = []

    def submit_job(self, parameters: SimulationParameters):
       parameters.path = os.path.join(self.path, parameters.sim_name)
       self.pool.append(parameters)

    def run(self):
        self.logger.log(f"Total number of jobs: {len(self.pool)}")
        self.logger.log(f"Total number of proccessors: {cpu_count()}")

        # Create the simulation parent folder
        os.mkdir(self.path)

        # Compile the modfiles and suppress output
        self.logger.log(f"Compiling modfiles.")
        os.system(f"nrnivmodl {self.cell_type.value['modfiles']} > /dev/null 2>&1")

        h.load_file('stdrun.hoc')
        h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')
        
        pool = Pool(processes = len(self.pool))
        pool.map(unwrap_self_run_single_simulation, zip([self] * len(self.pool), self.pool))
        pool.close()
        pool.join()

        # Delete the compiled modfiles
        os.system("rm -r x86_64")

    def run_single_simulation(self, parameters: SimulationParameters):
        
        # Create a folder to save to
        os.mkdir(parameters.path)

        # Build the cell
        cell_builder = CellBuilder(self.cell_type, parameters, self.logger)
        cell, _ = cell_builder.build_cell()

        # Save the (parent, child) adjacency matrix
        adj_matrix = cell.compute_directed_adjacency_matrix()
        np.savetxt(os.path.join(parameters.path, "adj_matrix.txt"), adj_matrix)

        # Classify segments by morphology, save coordinates
        segments, seg_data = cell.get_segments(["all"]) # (segments is returned here to preserve NEURON references)
        seg_sections = []
        seg_idx = []
        seg_coords = []
        seg_half_seg_RAs = []
        seg_Ls = []
        for entry in seg_data:
            sec_name = entry.section.split(".")[1] # name[idx]
            seg_sections.append(sec_name.split("[")[0])
            seg_idx.append(sec_name.split("[")[1].split("]")[0])
            seg_coords.append(entry.coords)
            seg_half_seg_RAs.append(entry.seg_half_seg_RA)
            seg_Ls.append(entry.L)
            
        seg_sections = pd.DataFrame({
            "section": seg_sections, 
            "idx_in_section": seg_idx, 
            "seg_half_seg_RA": seg_half_seg_RAs,
            "L": seg_Ls
            })
        
        seg_coords = pd.concat(seg_coords)
        seg_data = pd.concat((seg_sections.reset_index(drop = True), seg_coords.reset_index(drop = True)), axis = 1)
        seg_data.to_csv(os.path.join(parameters.path, "segment_data.csv"))

        # Compute electrotonic distances from soma
        elec_distances_soma = cell.compute_electrotonic_distance(from_segment = cell.soma[0](0.5))
        elec_distances_soma.to_csv(os.path.join(parameters.path, "elec_distance_soma.csv"))

        # Compute electrotonic distances from nexus
        elec_distances_nexus = cell.compute_electrotonic_distance(from_segment = cell.apic[36](0.961538))
        elec_distances_nexus.to_csv(os.path.join(parameters.path, "elec_distance_nexus.csv"))

        # Create an ECP object for extracellular potential
        #elec_pos = params.ELECTRODE_POSITION
        #ecp = EcpMod(cell, elec_pos, min_distance = params.MIN_DISTANCE)
        # Set ECP
        # if parameters.record_ecp == True:
        #     # Reason: (NEURON: Impedance calculation with extracellular not implemented)
        #     self.logger.log_warining("Recording ECP adds the extracellular channel to all segments after computing electrotonic distance.\
        #                              This channel is therefore not accounted for in impedence calculation, but it might affect the simulation.")
        #     h.cvode.use_fast_imem(1)
        #     for sec in cell.all: sec.insert('extracellular')
        #     cell.add_segment_recorders(var_name = "i_membrane_")

        # Save constants
        with open(os.path.join(parameters.path, "parameters.pickle"), "wb") as file:
           pickle.dump(parameters, file)

        # In time stamps, i.e., ms / dt
        time_step = 0

        h.celsius = parameters.h_celcius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        if is_indexable(cell.soma):
            h.v_init = cell.soma[0].e_pas
        else:
            h.v_init = cell.soma.e_pas

        h.finitialize(h.v_init)

        self.logger.log("Starting simulation.")
        while h.t <= h.tstop + 1:

            if (time_step > 0) & (time_step % (parameters.save_every_ms / parameters.h_dt) == 0):
                # Log progress
                self.logger.log_step(time_step)

                # Save data
                cell.write_recorder_data(
                    os.path.join(parameters.path, f"saved_at_step_{time_step}"), 
                    int(1 / parameters.h_dt))

                # Save lfp
                #loc_param = [0., 0., 45., 0., 1., 0.]
                #lfp = ecp.calc_ecp(move_cell = loc_param).T  # Unit: mV

                #with h5py.File(os.path.join(parameters.path, f"saved_at_step_{time_step}", "lfp.h5"), 'w') as file:
                #    file.create_dataset("report/biophysical/data", data = lfp)

                # Save net membrane current
                #with h5py.File(os.path.join(parameters.path, f"saved_at_step_{time_step}", "i_membrane_report.h5"), 'w') as file:
                #    file.create_dataset("report/biophysical/data", data = ecp.im_rec.as_numpy())

                # Reinitialize recording vectors
                for recorder_or_list in cell.recorders: recorder_or_list.clear()

                #for vec in ecp.im_rec.vectors: vec.resize(0)

            h.fadvance()
            time_step += 1
    
def is_indexable(obj: object):
    """
    Check if the object is indexable.
    """
    try:
        _ = obj[0]
        return True
    except:
        return False


        
        
