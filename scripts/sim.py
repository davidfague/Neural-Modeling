import sys
sys.path.append("../")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import SimulationParameters, HayParameters

import numpy as np

if __name__ == "__main__":
    sim = Simulation(SkeletonCell.Hay)
    
    # Looping over the range for exc_gmax_mean
    for exc_gmax_mean in np.arange(0.15, 0.46, 0.05):  # Assuming a step of 0.05, modify as needed
        for clip_upper_bound in np.arange(0.1, 1.1, 0.1):  # Assuming a step of 0.1, modify as needed
            exc_gmax_clip = (0, clip_upper_bound)
            
            # Now you can use these values in your simulation parameters
            sim.submit_job(HayParameters(
                sim_name=f"HAY_sim_mean_{exc_gmax_mean}_clip_{clip_upper_bound}",
                exc_gmax_mean_0=exc_gmax_mean,
                exc_gmax_clip=exc_gmax_clip
            ))

    sim.run()
