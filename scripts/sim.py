import sys
sys.path.append("../")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import SimulationParameters, HayParameters

import numpy as np

in_vitro = False

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay) # Detailed (all 3 if in-vivo)
    sim.submit_job(HayParameters("try1", h_tstop = 1000, reduce_cell = False, expand_cable = False))
    
    # if in_vitro:
    #   for amp in [-2.0,-1.0,-0.5,0.0,0.25,0.5,0.6,0.75,1.0,1.5,2.0]:
    #     sim.submit_job(HayParameters(sim_name=f"FI_HAY_DR_{int(amp*1000)}", reduce_cell=True, expand_cable=True, 
    #       CI_on=True, h_i_amplitude=amp, h_i_duration=10000, h_i_delay=1000, h_tstop=11000))
    #     sim.submit_job(HayParameters(sim_name=f"FI_HAY_NR_{int(amp*1000)}", reduce_cell=True, expand_cable=False, 
    #       CI_on=True, h_i_amplitude=amp, h_i_duration=10000, h_i_delay=1000, h_tstop=11000))
    #     sim.submit_job(HayParameters(sim_name=f"FI_HAY_DET_{int(amp*1000)}", reduce_cell=False, expand_cable=False, 
    #       CI_on=True, h_i_amplitude=amp, h_i_duration=10000, h_i_delay=1000, h_tstop=11000))

    # else:
    #   sim.submit_job(HayParameters(sim_name=f"HAY_DET_sim", reduce_cell=False, expand_cable=False))
    #   sim.submit_job(HayParameters(sim_name=f"HAY_NR_sim", reduce_cell=True, expand_cable=False))
    #   sim.submit_job(HayParameters(sim_name=f"HAY_DR_sim", reduce_cell=True, expand_cable=True))
      
    sim.run()
    
#    for inh_gmax in [1,1.5,2,2.5,3]:
#      sim.submit_job(HayParameters(
#      sim_name=f"HAY_sim_{inh_gmax}_inh_gmax",
#      inh_gmax_dist=inh_gmax,
#      soma_gmax_dist=inh_gmax,
#      h_tstop=50000
#      ))
    
#    # Looping over the range for exc_gmax_mean
#    for exc_gmax_mean in np.arange(0.15, 0.46, 0.15):  # Assuming a step of 0.05, modify as needed
#        for clip_upper_bound in np.arange(0.5, 1.1, 0.25):  # Assuming a step of 0.1, modify as needed
#            exc_gmax_clip = (0, clip_upper_bound)
#            
#            # Now you can use these values in your simulation parameters
#            sim.submit_job(HayParameters(
#                sim_name=f"HAY_sim_mean_{exc_gmax_mean}_clip_{clip_upper_bound}",
#                exc_gmax_mean_0=exc_gmax_mean,
#                exc_gmax_clip=exc_gmax_clip
#            ))

