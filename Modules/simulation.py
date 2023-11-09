from Modules.cell_builder import SkeletonCell, CellBuilder
from Modules.constants import SimulationParameters
from Modules.logger import Logger
from Modules.recorder import Recorder

from cell_inference.config import params
from cell_inference.utils.currents.ecp import EcpMod

from neuron import h

import os, datetime
import pickle, h5py

from multiprocessing import Pool, cpu_count

# https://stackoverflow.com/questions/31729008/python-multiprocessing-seems-near-impossible-to-do-within-classes-using-any-clas
def unwrap_self_run_single_simulation(args):
    return Simulation.run_single_simulation(args[0], args[1])

class Simulation:

    def __init__(self, cell_type: SkeletonCell, title=None):
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
        cell = cell_builder.build_cell()

        # Construct segment indexes
        seg_indexes = self.construct_seg_indexes(cell, parameters)

        # Compute electrotonic distances from nexus
        cell.recompute_segment_elec_distance(segment = cell.segments[seg_indexes["nexus"]], seg_name = "nexus")

        # Create an ECP object for extracellular potential
        elec_pos = params.ELECTRODE_POSITION
        ecp = EcpMod(cell, elec_pos, min_distance = params.MIN_DISTANCE)

        # Save segment indexes for plotting
        with open(os.path.join(parameters.path, "seg_indexes.pickle"), "wb") as file: 
            pickle.dump(seg_indexes, file)

        # Save segment info
        cell.write_seg_info_to_csv(path = parameters.path, seg_info = cell_builder.detailed_seg_info, title_prefix = 'detailed_')

        # Save constants
        with open(os.path.join(parameters.path, "parameters.pickle"), "wb") as file:
            pickle.dump(parameters, file)

        # In time stamps, i.e., ms / dt
        time_step = 0

        h.celsius = parameters.h_celcius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        if self.is_indexable(cell.soma):
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
                cell.generate_recorder_data(parameters.vector_length)
                cell.write_recorder_data(os.path.join(parameters.path, f"saved_at_step_{time_step}"))

                # Save lfp
                loc_param = [0., 0., 45., 0., 1., 0.]
                lfp = ecp.calc_ecp(move_cell = loc_param).T  # Unit: mV

                with h5py.File(os.path.join(parameters.path, f"saved_at_step_{time_step}", "lfp.h5"), 'w') as file:
                    file.create_dataset("report/biophysical/data", data = lfp)

                # Save net membrane current
                with h5py.File(os.path.join(parameters.path, f"saved_at_step_{time_step}", "i_membrane_report.h5"), 'w') as file:
                    file.create_dataset("report/biophysical/data", data = ecp.im_rec.as_numpy())

                # Reinitialize vectors: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=2579
                for recorder in cell.recorders:
                    recorder.vec.resize(0)

                for syn in cell.synapses:
                    for vec in syn.rec_vec: vec.resize(0)

                for vec in ecp.im_rec.vectors: vec.resize(0)

            h.fadvance()
            time_step += 1

    def construct_seg_indexes(self, cell, parameters):
        _, _, segments = cell.get_segments()
        soma_seg_index = segments.index(cell.soma[0](0.5))
        axon_seg_index = segments.index(cell.axon[-1](0.9))
        basal_seg_index = segments.index(cell.basals[0](0.5))
        trunk_seg_index = segments.index(cell.apic[0](0.999))

        # Find tuft and nexus
        # Dendritic reduced model
        if (parameters.reduce_cell == True) and (parameters.expand_cable == True):
            # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
            tuft_seg_index = tuft_seg_index = segments.index(cell.tufts[0](0.5))
            nexus_seg_index = segments.index(cell.apic[0](0.99))
            # NR model
        elif (parameters.reduce_cell == True) and (parameters.expand_cable == False):
            # tufts[0] will be the cable that is both trunk and tuft in this case, so we have to specify near end of cable
            tuft_seg_index = segments.index(cell.tufts[0](0.9))
            nexus_seg_index = segments.index(cell.apic[0](0.289004))
        else: # Complex cell
            # Otherwise tufts[0] will be truly tuft section and the segment in the middle of section is fine
            tuft_seg_index = segments.index(cell.tufts[0](0.5))
            if self.cell_type == SkeletonCell.NeymotinDetailed:
                nexus_seg_index = segments.index(cell.apic[24](0.99)) # May need to adjust
            else:
                nexus_seg_index = segments.index(cell.apic[36](0.961538))

        seg_indexes = {
            "soma": soma_seg_index,
            "axon": axon_seg_index,
            "basal": basal_seg_index,
            "trunk": trunk_seg_index,
            "tuft": tuft_seg_index,
            "nexus": nexus_seg_index
        }
        return seg_indexes
        
    def is_indexable(self, obj: object):
        """
        Check if the object is indexable.
        """
        try:
            _ = obj[0]
            return True
        except:
            return False


        
        
