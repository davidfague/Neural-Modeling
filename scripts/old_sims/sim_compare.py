import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    # Runs all 3 models
    sim = Simulation(SkeletonCell.Hay)
    sim.submit_job(HayParameters(f"detailed", h_tstop= 5000, reduce_cell = False, expand_cable = False, record_ecp=True, record_seg_to_seg=False))
    sim.submit_job(HayParameters(f"neuron_reduce", h_tstop= 5000, reduce_cell = True, expand_cable = False, record_ecp=True, record_seg_to_seg=True))
    sim.submit_job(HayParameters(f"cable_expander", h_tstop= 5000, reduce_cell = True, expand_cable = True, record_ecp=True, record_seg_to_seg=True))
    sim.run()