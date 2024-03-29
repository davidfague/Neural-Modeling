import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.simulation import Simulation
from Modules.cell_builder import SkeletonCell
from Modules.constants import HayParameters

import numpy as np

if __name__ == "__main__":
    
    sim = Simulation(SkeletonCell.Hay)
    for amp in np.arange(-2, 2.1, 0.2):  # 2.1 ensures that 2.0 is included in the range
        rounded_amp = round(amp, 1)  # Round the amplitude to 1 decimal place
        sim.submit_job(HayParameters(f"Detailed_FI_amp{rounded_amp}", h_tstop= 2000, reduce_cell = False, expand_cable = False, all_synapses_off=True, record_ecp=False,
        CI_on=True, h_i_amplitude=rounded_amp, h_i_duration=1950, h_i_delay=50))
        sim.submit_job(HayParameters(f"Reduced_FI_amp{rounded_amp}", h_tstop= 2000, reduce_cell = True, expand_cable = False, all_synapses_off=True, record_ecp=False,
        CI_on=True, h_i_amplitude=rounded_amp, h_i_duration=1950, h_i_delay=50))
        sim.submit_job(HayParameters(f"Expanded_FI_amp{rounded_amp}", h_tstop= 2000, reduce_cell = True, expand_cable = True, all_synapses_off=True, record_ecp=False,
        CI_on=True, h_i_amplitude=rounded_amp, h_i_duration=1950, h_i_delay=50, use_mm=True))
    sim.run()