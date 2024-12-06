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
    Ca_percents = ['00',20,40,60,80]
    #sim.submit_job(HayParameters(f"Baseline_Complex_Testing_mm", h_tstop= 1000, reduce_cell = False, expand_cable = False, all_synapses_off=True, record_ecp=True, save_every_ms=1000, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=exc_gmax_mean, exc_gmax_clip=exc_gmax_clip, numpy_random_state=1111, neuron_random_state=1111, use_mm=True))

    sim.submit_job(HayParameters(f"Baseline_Reduced", h_tstop= 10000, reduce_cell = True, expand_cable = True, all_synapses_off=False, record_ecp=True, save_every_ms=1000, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=4.6, exc_gmax_clip=(0,30), numpy_random_state=1111, neuron_random_state=1111, use_mm=True))
    
#    for Ca_percent in Ca_percents:
#        sim.submit_job(HayParameters(f"Ca_{Ca_percent}_Complex", h_tstop= 30000, reduce_cell = False, expand_cable = False, all_synapses_off=False, record_ecp=False, save_every_ms=1000, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=exc_gmax_mean, exc_gmax_clip=exc_gmax_clip, numpy_random_state=1111, neuron_random_state=1111, Hay_biophys=f"L5PCbiophys3_{Ca_percent}_percent_Ca.hoc"))
#
#        sim.submit_job(HayParameters(f"Ca_{Ca_percent}_Reduced", h_tstop= 30000, reduce_cell = True, expand_cable = True, all_synapses_off=False, record_ecp=True, save_every_ms=1000, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=4.6, exc_gmax_clip=(0,30), numpy_random_state=1111, neuron_random_state=1111, Hay_biophys=f"L5PCbiophys3_{Ca_percent}_percent_Ca.hoc"))
    
    sim.run()