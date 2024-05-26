import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters
from time import time

if __name__ == "__main__":
    
    start = time()
    sim = Simulation(SkeletonCell.Hay)
    seed = 123
    sim.submit_job(HayParameters(
        f"{seed}", 
        h_tstop = 1000, 
        reduce_cell = False, 
        expand_cable = False, 
        numpy_random_state = seed, 
        neuron_random_state = seed))
    sim.run()
    print(time() - start)

