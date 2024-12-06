import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    sim = Simulation(SkeletonCell.Hay)
    exc_gmax_mean = 2.3#4.6 for CE reduced before synapses
    exc_gmax_clip = (0, 15)#30 for CE reduced before synapses
    
    random_states= [[1111,1111],[2222,2222],[3333,3333],[4444,4444],[5555,5555]]

    sim.submit_job(HayParameters(f"Detailed_EPSPs", h_tstop= 100, reduce_cell = False, expand_cable = False, all_synapses_off=False, record_ecp=False, save_every_ms=100, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=2.3, exc_gmax_clip=(0,15), numpy_random_state=1111, neuron_random_state=1111, use_mm=False, simulate_EPSPs=True))
    
    sim.run()