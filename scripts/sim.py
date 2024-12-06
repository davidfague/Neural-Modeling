import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay)
    #sim.submit_job(HayParameters("Expanded_LFP_try1", h_tstop= 5000, reduce_cell = True, expand_cable = True))
    sim.submit_job(HayParameters(f"selective", h_tstop= 1000, reduce_cell = True, reduce_cell_selective = True, record_ecp=True, reduce_tufts=True))
    sim.run()

