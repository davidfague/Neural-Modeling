from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation_slurm import Simulation
from Modules.cell_builder import SkeletonCell

from neuron import h
import pickle

if __name__ == "__main__":

    print(rank)

    try:
        h.load_file('stdrun.hoc')
        h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')
    except:
        # Already loaded
        pass 

    sim = Simulation(SkeletonCell.Hay)

    # Load parameters for this pid
    with open(f"params/{rank}.pickle", "rb") as file:
        parameters = pickle.load(file)

    sim.run_single_simulation(parameters)

