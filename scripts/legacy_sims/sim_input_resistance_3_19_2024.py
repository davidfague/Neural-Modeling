import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    sim = Simulation(SkeletonCell.Hay)
    
    sim.submit_job(HayParameters(f"Detailed_Rinput", h_tstop= 250, reduce_cell = False, expand_cable = False, all_synapses_off=True, record_ecp=False,
    CI_on=True, h_i_amplitude=-1.0, h_i_duration=200, h_i_delay=50, save_every_ms=250, use_mm=True, CI_target='nexus'))
    
    sim.submit_job(HayParameters(f"Expanded_Rinput", h_tstop= 250, reduce_cell = True, expand_cable = True, all_synapses_off=True, record_ecp=False,
    CI_on=True, h_i_amplitude=-1.0, h_i_duration=200, h_i_delay=50, save_every_ms=250, reduction_before_synapses=True, use_mm=True, CI_target='nexus'))

    sim.run()