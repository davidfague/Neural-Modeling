import sys
sys.path.append("../")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import SimulationParameters, HayParameters

import numpy as np

if __name__ == "__main__":

    sim = Simulation(SkeletonCell.Hay)

    for ci in [-1, 0, 1]:
        sim.submit_job(HayParameters(sim_name = f"{ci}", h_i_amplitude = ci))

    sim.run()