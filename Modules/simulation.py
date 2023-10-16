from cell_builder import SkeletonCell, CellBuilder
from constants import SimulationParameters
from logger import Logger

from neuron import h
import os, datetime
from dataclasses import dataclass

from multiprocessing import Process, cpu_count

class Simulation:

    def __init__(self, cell_type: SkeletonCell):
        self.cell_type = cell_type
        self.path = f"{cell_type}-{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"

        self.logger = Logger(None)
        self.pool = []

    def submit_job(self, parameters: SimulationParameters):
       parameters.path = os.path.join(self.path, parameters.sim_name)
       self.pool.append(Process(target = self.run_single_simulation, args = [parameters]))

    def run(self):
        self.logger.log(f"Total number of jobs: {len(self.pool)}")
        self.logger.log(f"Total number of proccessors: {cpu_count()}")

        for p in self.pool: p.start()
        for p in self.pool: p.join()
        for p in self.pool: p.terminate()


    def run_single_simulation(self, parameters: SimulationParameters):

        os.mkdir(parameters.path)
        os.system(f"cd {parameters.path} > nrnivmodl {MODFILES}") # ADD MODFILES HERE!

        cell = CellBuilder(self.cell_type, parameters, self.logger)

        h.celsius = parameters.h_celcius
        h.tstop = parameters.h_tstop
        h.dt = parameters.h_dt
        h.steps_per_ms = 1 / h.dt
        if self.is_indexable(cell.soma):
            h.v_init = cell.soma[0].e_pas
        else:
            h.v_init = cell.soma.e_pas

        h.finitialize(h.v_init)
        while h.t <= h.tstop + 1:

            if time_step % (constants.log_every_ms / constants.h_dt) == 0:
                logger.log(f"Running simulation step {time_step}")
                logger.log_memory()

            if (time_step > 0) & (time_step % (constants.save_every_ms / constants.h_dt) == 0):
                # Save data
                cell.generate_recorder_data(constants.save_every_ms)
                cell.write_data(os.path.join(save_folder, f"saved_at_step_{time_step}"))

                # Save lfp
                loc_param = [0., 0., 45., 0., 1., 0.]
                lfp = ecp.calc_ecp(move_cell = loc_param).T  # Unit: mV

                with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "lfp.h5"), 'w') as file:
                    file.create_dataset("report/biophysical/data", data = lfp)
                # save net membrane current
                with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "i_membrane_report.h5"), 'w') as file:
                    file.create_dataset("report/biophysical/data", data = ecp.im_rec.as_numpy())

                # Save time
                with h5py.File(os.path.join(save_folder, f"saved_at_step_{time_step}", "t.h5"), 'w') as file:
                    file.create_dataset("report/biophysical/data", data = t_vec.as_numpy())

                logger.log(f"Saved at time step {time_step}")

                time_steps_saved_at.append(time_step)

                # Reinitialize vectors: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=2579
                t_vec.resize(0)
                for vec in V_rec.vectors: vec.resize(0)
                for vec in cell.Vm.vectors: vec.resize(0)
                for recorder in cell.recorders.items():
                    for vec in recorder[1].vectors: vec.resize(0)
                cell.spikes.resize(0)

                for inj in cell.injection: inj.rec_vec.resize(0)

                for syn in all_syns:
                    for vec in syn.rec_vec: vec.resize(0)
                
                for vec in ecp.im_rec.vectors: vec.resize(0)

            h.fadvance()
            time_step += 1
        
    def is_indexable(self, obj: object):
        """
        Check if the object is indexable.
        """
        try:
            _ = obj[0]
            return True
        except:
            return False


        
        