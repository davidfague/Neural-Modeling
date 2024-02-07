import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay)
    for i in range(4):
        sim.submit_job(
            HayParameters(
                f"sim_{i}", 
                h_tstop = 1000, 
                reduce_cell = False, 
                expand_cable = False,
                numpy_random_state = np.random.randint(10, 999),
                neuron_random_state = np.random.randint(10, 999),
                exc_mean_fr = float(np.random.randint(30, 50) / 10),
                inh_prox_mean_fr = float(np.random.randint(100, 200) / 10),
                inh_distal_mean_fr = float(np.random.randint(30, 50) / 10)
                ))
    sim.run(batch_size = 2)

