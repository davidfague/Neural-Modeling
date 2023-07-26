import sys
sys.path.append("../")

import numpy as np
import h5py, os
import constants
import matplotlib.pyplot as plt

# Output folder should store folders 2023...
output_folder = "output"
skip = 300 # (ms)

def main():
    step_size = int(constants.save_every_ms / constants.h_dt) # Timestamps
    steps = range(step_size, int(constants.h_tstop / constants.h_dt) + 1, step_size) # Timestamps
    
    firing_rates = []

    for ampl in constants.h_i_amplitudes:
        spikes = []
        for ampl_dir in os.listdir(output_folder):
            if ampl_dir.endswith(f"_{int(ampl * 1000)}"):
                for step in steps:
                    dirname = os.path.join(output_folder, ampl_dir, f"saved_at_step_{step}")
                    with h5py.File(os.path.join(dirname, "spikes_report.h5")) as file:
                        spikes.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
        spikes = np.hstack(spikes)
        firing_rate = len(spikes[spikes > skip]) / (constants.h_tstop / 1000)
        firing_rates.append(firing_rate)

    # Save FI curve
    plt.figure(figsize = (7, 8))
    plt.plot(constants.h_i_amplitudes, firing_rates)
    plt.xlabel("Amplitude (nA)")
    plt.ylabel("Hz")
    plt.savefig(os.path.join(output_folder, f"FI.png"))

    # Save firing rates
    with open(os.path.join(output_folder, "firing_rates.csv"), "a") as file:
        for i in range(len(constants.h_i_amplitudes)):
            file.writelines(f"{constants.h_i_amplitudes[i]},{firing_rates[i]}\n")

if __name__ == "__main__":
    main()