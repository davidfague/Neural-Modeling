import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    sim = Simulation(SkeletonCell.Hay)
    exc_gmax_original_mean = 2.3
    exc_gmax_means = [exc_gmax_original_mean*1,exc_gmax_original_mean*2, exc_gmax_original_mean*3, exc_gmax_original_mean*4, exc_gmax_original_mean*5, exc_gmax_original_mean*6]
    
    exc_gmax_clip = (0, 30)
    
    # Round the entries to one decimal place
    rounded_exc_gmax_means = [round(entry, 1) for entry in exc_gmax_means]
    
    random_states= [[1111,1111],[2222,2222],[3333,3333],[4444,4444],[5555,5555]]
    
    for exc_gmax_mean in rounded_exc_gmax_means:
        for [numpy_random_state, neuron_random_state] in random_states:
            #sim.submit_job(HayParameters(f"Tuning_{neuron_random_state}_CE_firing_rate_by_exc_gmax_mean_{exc_gmax_mean}", h_tstop= 2000, reduce_cell = True, expand_cable = True, all_synapses_off=False, record_ecp=False, CI_on=False, h_i_amplitude=-1.0, h_i_duration=200, h_i_delay=50, save_every_ms=1000, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=exc_gmax_mean, exc_gmax_clip=exc_gmax_clip, numpy_random_state=numpy_random_state, neuron_random_state=neuron_random_state))
            
            sim.submit_job(HayParameters(f"Tuning_{neuron_random_state}_CE_firing_rate_by_exc_gmax_mean_{exc_gmax_mean}", h_tstop= 2000, reduce_cell = True, expand_cable = True, all_synapses_off=False, record_ecp=False, save_every_ms=1000, test_morphology = False, reduction_before_synapses=True,exc_gmax_mean_0=exc_gmax_mean, exc_gmax_clip=(0,30), numpy_random_state=numpy_random_state, neuron_random_state=neuron_random_state, use_mm=True))
    
    sim.run()