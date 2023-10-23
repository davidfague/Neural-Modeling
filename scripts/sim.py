import sys
sys.path.append("../")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import SimulationParameters, HayParameters

import numpy as np

if __name__ == "__main__":

    sim = Simulation(SkeletonCell.Hay)
    sim.submit_job(HayParameters(sim_name = f"ben1"))
    # etc, submit all the variations

    sim.run()
