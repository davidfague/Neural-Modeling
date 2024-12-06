import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    sim = Simulation(SkeletonCell.Hay)
    
    sim.submit_job(HayParameters(f"Testing_FINISH_error", h_tstop= 250, reduce_cell = True, expand_cable = True, all_synapses_off=False, record_ecp=False,
    CI_on=False, h_i_amplitude=-1.0, h_i_duration=200, h_i_delay=50, save_every_ms=250, test_morphology = True, exc_n_FuncGroups=642, exc_n_PreCells_per_FuncGroup=1, inh_distributed_n_FuncGroups=642, inh_distributed_n_PreCells_per_FuncGroup=1))
    
    sim.run()