import sys
sys.path.append("../")

import numpy as np
import h5py, pickle, os
import func_group_sim_constants as constants

from Modules.plotting_utils import plot_simulation_results
from cell_inference.config import params
from cell_inference.utils.plotting.plot_results import plot_lfp_heatmap, plot_lfp_traces

# Output folder should store folders 'saved_at_step_xxxx'
output_folder = "output/2023-07-25_14-10-01_seeds_123_1L5PCtemplate[0]_642nseg_108nbranch_28918NCs_28918nsyn"
steps = [10000]
step_size = 10000

# Check shape of each saved file
analyze = True

# Plot voltage and lfp traces
plot_voltage_lfp = True

def main():
    if analyze:
        for step in steps:
            dirname = os.path.join(output_folder, f"saved_at_step_{step}")
            for filename in os.listdir(dirname):
                if filename.endswith(".h5"):
                    with h5py.File(os.path.join(dirname, filename)) as file:
                        data = np.array(file["report"]["biophysical"]["data"])
                        print(f"{os.path.join(f'saved_at_step_{step}', filename)}: {data.shape}")
    
    if plot_voltage_lfp:
        t = np.arange(int(constants.h_tstop / constants.h_dt))
        Vm = []
        lfp = []
        for step in steps:
            dirname = os.path.join(output_folder, f"saved_at_step_{step}")
            with h5py.File(os.path.join(dirname, "Vm_report.h5")) as file:
                Vm.append(np.array(file["report"]["biophysical"]["data"])[:, :step_size])
            with h5py.File(os.path.join(dirname, "lfp.h5")) as file:
                lfp.append(np.array(file["report"]["biophysical"]["data"])[:step_size, :])
        Vm = np.hstack(Vm)
        lfp = np.vstack(lfp)
        print(lfp.shape)

        with open(os.path.join(output_folder, "seg_indexes.pickle"), "rb") as file:
            seg_indexes = pickle.load(file)
        
        loc_param = [0., 0., 45., 0., 1., 0.]
        elec_pos = params.ELECTRODE_POSITION
        plot_simulation_results(t, Vm, seg_indexes['soma'], seg_indexes['axon'], seg_indexes['basal'], 
                                seg_indexes['tuft'], seg_indexes['nexus'], seg_indexes['trunk'], 
                                loc_param, lfp, elec_pos, plot_lfp_heatmap, plot_lfp_traces, vlim = [-0.023,0.023],
                                show = False, save_dir = output_folder)


if __name__ == "__main__":
    main()