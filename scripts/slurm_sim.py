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

# Use a command-line argument for sim_title if provided, otherwise use a default value.
if len(sys.argv) > 1:
    sim_title = sys.argv[1]
else:
    sim_title = 'sim'  # Provide a default if no argument is given.

if __name__ == "__main__":

    print(f"Rank {rank} running simulation with title: {sim_title}")

    try:
        h.load_file('stdrun.hoc')
        h.nrn_load_dll('./x86_64/.libs/libnrnmech.so')
    except:
        # Already loaded
        pass 

    # sim = Simulation(SkeletonCell.Allen, sim_title)
    sim = Simulation(SkeletonCell.Hay, sim_title)

    # Load parameters for this pid
    with open(f"params/{rank}.pickle", "rb") as file:
        parameters = pickle.load(file)

    sim.run_single_simulation(parameters)

