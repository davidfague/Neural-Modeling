import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay)
    for i in range(3):
        sim.submit_job(HayParameters(f"sim_{i}", h_tstop = 1000, reduce_cell = False, expand_cable = False))
    sim.run()

