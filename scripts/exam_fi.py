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
        Vm = []
        t = []
        print("amplitude: ", ampl)
        print(f"_{int(ampl * 1000)}")
        for ampl_dir in os.listdir(output_folder): # list folders in directory
            if ampl_dir.endswith(f"_{int(ampl * 1000)}"): # go over all amplitudes
                print(ampl, ampl_dir)
                for step in steps:
                    dirname = os.path.join(output_folder, ampl_dir, f"saved_at_step_{step}")
                    with h5py.File(os.path.join(dirname, "Vm_report.h5")) as file:
                        Vm.append(np.array(file["report"]["biophysical"]["data"])[:, :step_size])
                    with h5py.File(os.path.join(dirname, "t.h5")) as file:
                        t.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
                    with h5py.File(os.path.join(dirname, "spikes_report.h5")) as file:
                        spikes.append(np.array(file["report"]["biophysical"]["data"])[:step_size])
        t = np.hstack(t) # (ms)
        Vm = np.hstack(Vm)
        spikes = np.hstack(spikes)
        print("spikes:", spikes)
        plt.figure(figsize = (7,8))
        plt.plot(t, Vm[0])
        for spike in spikes:
          plt.scatter(spike, 30, color = 'black', marker='*')
        plt.xlabel("Time (ms)")
        plt.ylabel("Vm (mV)")
        plt.title(str(ampl))
        plt.savefig(str(ampl)+".png")
        plt.close()   
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