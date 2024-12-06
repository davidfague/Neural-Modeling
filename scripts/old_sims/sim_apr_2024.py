import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay)
    # WIP
    #sim.submit_job(HayParameters("Expanded_LFP_try1", h_tstop= 5000, reduce_cell = True, expand_cable = True))
    #sim.submit_job(HayParameters(f"testing_section_deletion_3_2024", h_tstop = 1000, all_synapses_off = True, reduce_cell = False, expand_cable = False, record_ecp=False, record_seg_to_seg = False, delete_section = True))
    sim.run()

