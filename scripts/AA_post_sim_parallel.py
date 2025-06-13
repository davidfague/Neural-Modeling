# count dspikes in simulations, plot stas etc in parallel

# NOTE: THIS SCRIPT ASSUMES YOU ARE IN A SUBFOLDER OF Neural-Modeling (the repo). [hence the ../]. 
# If you get an error that the scripts cannot be found then just cd into Neural-Modeling/scripts.

import sys
sys.path.append('..')
sys.path.append('../Modules')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Modules.analysis as analysis

from multiprocessing import Pool
import subprocess

def _run(sim_dir):
    # load the parameters from the simulation directory
    parameters = analysis.DataReader.load_parameters(sim_dir)

    # plot voltage
    def plot_voltage(sim_dir, seg_id):
        print(f"Plotting voltage for {sim_dir}:")
        v = analysis.DataReader.read_data(sim_dir, 'v')
        parameters = analysis.DataReader.load_parameters(sim_dir)
        t = np.arange(0,parameters.h_tstop +parameters.h_dt, parameters.h_dt)
        plt.plot(t, v[seg_id,:])
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'Voltage at seg {seg_id}')
        # plt.show()
        plt.savefig(os.path.join(sim_dir, 'soma voltage'))
    plot_voltage(sim_dir, seg_id=0)

    soma_spikes = analysis.DataReader.read_data(sim_dir, 'soma_spikes')
    # !python ../scripts/plot_ac.py -d {sim_dir}

    # find dendritic spikes
    if parameters.record_all_channels and parameters.record_all_synapses:
        print("Checking dendritic spikes")
        subprocess.run(["python", "../scripts/find_events_ben.py", "-d", sim_dir]) # !python "../scripts/find_events_ben.py" -d {sim_dir}
    else:
        print("Not checking dendritic spikes")


    # # plot voltages where d spikes were detected
    # if parameters.record_all_channels and parameters.record_all_synapses:
    #     !python ../scripts/plot_vm_for_dend_spikes.py -d {sim_dir} -s
    # else:
    #     print("Not checking dendritic spikes")

    # plot d spike properties
    if parameters.record_all_channels and parameters.record_all_synapses:
        # !python ../Modules/event_histograms.py -d {sim_dir}
        subprocess.run(["python", "../Modules/event_histograms.py", "-d", sim_dir])
        pd.read_csv(os.path.join(sim_dir, 'dSpike_table.csv')).head(10) # print first 10 rows of the dendritic spikes csv file
    else:
        print("Not checking dendritic spikes")

    # STA
    # !python ../scripts/plot_sta.py -d {sim_dir} -s
    subprocess.run(["python", "../scripts/plot_sta.py", "-d", sim_dir, "-s"])
    

if __name__ == "__main__":

    pool = Pool(processes = 6)

    sims_dir = '/home/drfrbc/Neural-Modeling/simulations/2025-06-13-12-26-pink_and_delay_no_clustering_fi' # folder containing simulation folders

    sim_dirs = [os.path.join(sims_dir, sim_dir) for sim_dir in os.listdir(sims_dir)] # every directory in sims_dir

    print(sim_dirs)

    pool.map_async(_run, sim_dirs)
    pool.close()
    pool.join()

    print(f'Finished all simulations')