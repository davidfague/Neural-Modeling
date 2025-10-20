import sys
sys.path.append('..')
sys.path.append('../Modules')

import os
from multiprocessing import Pool

from Modules.synapses_file import PreSimSynapseGenerator #TODO: rename synapses_file to something else
from Modules.simulation_slurm import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.logger import Logger
import Modules.analysis as analysis

def _run(sim_dir):

    #--- ...
    parameters = analysis.DataReader.load_parameters(sim_dir)

    #--- Read simulation folder to build cell synapses
    pssg = PreSimSynapseGenerator(sim_dir)

    # TODO: Implement run on all simulations within sims_dir
    cell = pssg.build_synapses_onto_cell_obj()

    #--- Simulate

    #TODO: clean up implementation. Probably creating extra folder. # create_dir is a quick fix to prevent creating a new sim_dir
    # (based on skeleton_cell) since we are using a sim_dir that already exists.
    sim = Simulation(getattr(SkeletonCell, parameters.skeleton_cell_type), create_dir = False) 
    
    #TODO: Prevent Simulation from creating a new sim_dir if one already exists. (path using skeleton_cell)
    #TODO: move builder_runtime.txt and replace_runtime.txt to sim_dir and to runtimes.csv
    sim.path = os.path.split(sim_dir)[0] #simulator.sims_dir
    sim.logger = Logger(sim_dir)

    sim.run_single_simulation(parameters=parameters, cell=cell)

if __name__ == "__main__":

    N_PROCESSES = 6  # max number of parallel processes

    sims_dir = '/home/drfrbc/Neural-Modeling/simulations/2025-10-15-15-13-1xTuftInh_2xTuftExc_2.5xNexusInh_0.25xNexusExc_ReduceExcFRto0.25x_0.75xBasalExc_2xperisomaticInh' # folder containing simulation folders

    print(f'Running simulations in {sims_dir}')

    sim_dirs = [
        os.path.join(sims_dir, sim_dir)
        for sim_dir in os.listdir(sims_dir)
        if os.path.isdir(os.path.join(sims_dir, sim_dir))
    ]

    if len(sim_dirs) < N_PROCESSES:
        N_PROCESSES = len(sim_dirs)
    pool = Pool(processes = N_PROCESSES)

    print(f'Running simulations: {sim_dirs}')

    pool.map_async(_run, sim_dirs)
    pool.close()
    pool.join()

    print(f'Finished all simulations')