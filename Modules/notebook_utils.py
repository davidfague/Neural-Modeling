import shutil
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
from Modules.cell_model import find_nexus_seg
import Modules.analysis as analysis


def plot_FI(cells: dict, sim: object, parameters: object, savename: str = "1000SynapsesFI.png"):
    # Adjust the parameters and initialize the cell structure here
    root_path = parameters.path
    FI_paths = []
    parameters.h_tstop = 2000
    parameters.h_i_duration = 1950
    parameters.h_i_delay = 50
    amps = np.arange(-2, 2.1, 0.5)

    # Prepare subplots: one for the somatic injection, another for the nexus injection
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Create a CSV file to store the results
    csv_file_path = "FI_Curve_Data.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cell Name", "Use Nexus", "Amplitude (nA)", "Firing Rate (Hz)"])

        for use_nexus, ax in zip([False, True], axes):
            for cell_name, cell in cells.items():
                firing_rates = []

                if use_nexus:  # move current injection to the nexus
                    nexus_segment = find_nexus_seg(cell, cell.compute_directed_adjacency_matrix())
                    segments, _ = cell.get_segments(['all'])
                    cell.current_injection.loc(segments[nexus_segment])

                for amp in amps:
                    parameters.path = root_path + f"/{cell_name}_{use_nexus}_1000synapsesFI_{amp:.1f}"
                    # os.mkdir(parameters.path)
                    # with open(os.path.join(parameters.path, "parameters.pickle"), "wb") as file:
                    #     pickle.dump(parameters, file)
                    FI_paths.append(parameters.path)
                    cell.current_injection.amp = amp

                    # Perform simulation
                    print(f"simulating {cell_name} {amp:.1f}")
                    sim.simulate(cell, parameters)

                    # Read the voltage and spike data
                    v = analysis.DataReader.read_data(parameters.path, "v", parameters=parameters)
                    soma_spikes = analysis.DataReader.read_data(parameters.path, "soma_spikes", parameters=parameters)

                    # Calculate the firing rate
                    firing_rate = len(soma_spikes[0]) / (parameters.h_tstop / 1000)
                    firing_rates.append(firing_rate)
                    
                    # Write the results to the CSV file
                    writer.writerow([cell_name, use_nexus, amp, firing_rate])

                label = cell_name
                ax.plot(amps, firing_rates, label=label)

            ax.set_title("Somatic F/I with Nexus Current Injection" if use_nexus else "Somatic F/I with Somatic Current Injection")
            ax.set_xlabel('Amplitude (nA)')
            ax.set_ylabel('Firing Rate (Hz)')
            ax.set_xlim(0,2)
            ax.legend()

    # Display the plots
    plt.tight_layout()

    # Save the figure to the current directory
    plt.savefig(savename)  # You can specify other formats like 'pdf' by changing the file extension

    plt.show()

    # Clean up temporary directories
    for FI_path in FI_paths:
        shutil.rmtree(FI_path)

    # Reset for future simulations
    parameters.path = root_path
    for cell_name,cell in cells.items():
        cell.current_injection.amp = 0