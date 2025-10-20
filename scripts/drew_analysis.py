import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')
sys.path.append('../Modules')
from Modules import drew_functions
import matplotlib.pyplot as plt
sys.path.append('/home/drfrbc/InhibOnDendComp_david')
from src.sta_files import sta_files
from src.load_caspks_csv import load_caspks_csv
from src.load_spike_h5 import load_spike_h5
from Modules import analysis

def add_elec_quantiles(sim_dir):
    ds_fpaths = [os.path.join(sim_dir, 'ca.csv'),
             os.path.join(sim_dir, 'nmda.csv'),
             os.path.join(sim_dir, 'na.csv')]

    new_ds_list = [drew_functions.join_nexus_atten(curr_ds_fpath) for curr_ds_fpath in ds_fpaths]

    # save new dataframes
    for orig_ds_fpath, new_ds_df in zip(ds_fpaths, new_ds_list):
        new_ds_fpath = orig_ds_fpath[:-4] + '_withNexusQ.csv'
        new_ds_df.to_csv(new_ds_fpath, index=False)



def dspike_analysis(sim_dir):
    samps_per_ms = 10
    sim_win = [0, try_to_load_max_timestep(sim_dir)]  # [0, 500000]#[0, 2000000] # beginning and start points of simulation in samples
    sta_win = [-100, 100]  # multiply by step to get window in milliseconds
    sta_step = 1  # binning step for each point in the STA
    step = 2 * samps_per_ms  # number of simulation steps for creating the dendritic
                             # event occurrence series
    
    figures_folder = os.path.join(sim_dir, 'figures_drew')
    if not os.path.exists(figures_folder):
        os.mkdir(figures_folder)
    sim_dict = {'SimName': 'output_allpoisson',
                'RootDir': sim_dir,
                'CaFile': os.path.join(sim_dir, 'ca_withNexusQ.csv'), #'output_allpoisson_ca_nex.csv',
                'NaFile': os.path.join(sim_dir, 'na_withNexusQ.csv'),#'output_allpoisson_na_nex.csv',
                'NMDAFile': os.path.join(sim_dir, 'nmda_withNexusQ.csv'),#'output_allpoisson_nmda_nex.csv',
                'APFile': os.path.join(sim_dir, 'spikes.h5') if os.path.exists(os.path.join(sim_dir, 'spikes.h5')) else sim_dir
                # 'APFile': "/home/drfrbc/InhibOnDendComp_david/data_ben/spikes.h5"
                }
    
    # process the simulation
    t_list = np.arange(sta_win[0], sta_win[1], sta_step) * (step / samps_per_ms)
    eqt_list = np.arange(9,-1,-1)
    t_inds = np.where(np.abs(t_list)<=50)[0]
    eqt_inds = np.arange(0,10)#np.where(eqt_list>0)[0]
    drew_functions.set_sta_axes(t_list, eqt_list, t_inds, eqt_inds)

    pois_sta = process_dspikes_relative_to_APs(sim_dict, step, sim_win, sta_step, sta_win)
    plot_dspikes_relative_to_APs(pois_sta, figures_folder)
    plot_Ca_relative_to_APs(pois_sta, figures_folder)

    pois_ca_sta = process_dspikes_relative_to_Ca_spikes(sim_dict, step, sim_win, sta_step, sta_win)
    plot_dspikes_relative_to_Ca_spikes(pois_ca_sta, figures_folder)


    pois_canmda_sta = process_NMDA_CA_coordination(sim_dict, step, sim_win, sta_step, sta_win, samps_per_ms)
    plot_NMDA_CA_coordination(pois_canmda_sta, pois_sta, figures_folder)


    
def plot_dspikes_relative_to_APs(pois_sta, figures_folder):
    fig,ax = plt.subplots(2,2)
    drew_functions.plot_sta_im(pois_sta['nmda_a'], ax[0,0])
    ax[0,0].set_title('NMDA Apical')
    ax[0,0].get_images()[0].set_clim(-500,500)
    drew_functions.plot_sta_im(pois_sta['nmda_b'], ax[1,0])
    ax[1,0].set_title('NMDA Basal')
    ax[1,0].get_images()[0].set_clim(-150,150)
    drew_functions.plot_sta_im(pois_sta['na_a'], ax[0,1])
    ax[0,1].set_title('Na Apical')
    ax[0,1].get_images()[0].set_clim(-400,400)
    drew_functions.plot_sta_im(pois_sta['na_b'], ax[1,1])
    ax[1,1].set_title('Na Basal')
    ax[1,1].get_images()[0].set_clim(-100,100)
    fig.supxlabel('Time from action potential (ms)')
    fig.supylabel('Electrotonic quantile 90->0')
    fig.suptitle('Coordination of dendritic events and action potentials')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, 'DendSpikesCC.png'))

def plot_Ca_relative_to_APs(pois_sta, figures_folder):
    fig,ax = plt.subplots()
    drew_functions.plot_sta_stair(pois_sta['ca_a'],ax)
    ax.set_title('Ca Apical')
    fig.supxlabel('Time from action potential (ms)')
    fig.supylabel('Percent change')
    fig.suptitle('Coordination of dendritic events and action potentials')
    fig.tight_layout()
    # fig.savefig('../figures/CaSpikesCC.pdf')
    fig.savefig(os.path.join(figures_folder, 'CaSpikesCC.png'))

def try_to_load_max_timestep(sim_dir):
    try:
        parameters = analysis.DataReader.load_parameters(sim_dir)
        n_timesteps = int(parameters.h_tstop / parameters.h_dt)
        return n_timesteps
    except Exception as e:
        # raise ValueError(f"Could not load parameters from {sim_dir}: {e}") # likely not our simulation
        n_timesteps = 1500000  # default to 1500000 (150 seconds at dt=0.1 ms)
        return n_timesteps

def process_dspikes_relative_to_APs(sim_dict, step, sim_win, sta_step, sta_win):
    # process the simulation
    pois_sta = {}
    pois_sta['ca_sta'] = sta_files(sim_dict['CaFile'], sim_dict['APFile'], 
                                step, sim_win,sta_step,sta_win, agg_colname='Elec_distanceQ')
    pois_sta['nmda_sta'] = sta_files(sim_dict['NMDAFile'], sim_dict['APFile'], 
                                    step, sim_win,sta_step,sta_win, agg_colname='Elec_distanceQ')
    pois_sta['na_sta'] = sta_files(sim_dict['NaFile'], sim_dict['APFile'], 
                                step, sim_win,sta_step,sta_win, agg_colname='Elec_distanceQ')

    pois_sta['ca_a'] = pois_sta['ca_sta'].loc['apic','sta']
    pois_sta['nmda_a'] = pois_sta['nmda_sta'].loc['apic','sta']
    pois_sta['nmda_b'] = pois_sta['nmda_sta'].loc['dend','sta']
    pois_sta['na_a'] = pois_sta['na_sta'].loc['apic','sta']
    pois_sta['na_b'] = pois_sta['na_sta'].loc['dend','sta']
    return pois_sta

def process_dspikes_relative_to_Ca_spikes(sim_dict, step, sim_win, sta_step, sta_win):
    # process the simulation
    pois_ca_sta = {}
    pois_ca_sta['nmda_sta'] = sta_files(sim_dict['NMDAFile'], sim_dict['CaFile'], 
                                    step, sim_win,sta_step,sta_win, 
                                    agg_colname='Elec_distance_nexusQ', ca_spk=True)
    pois_ca_sta['na_sta'] = sta_files(sim_dict['NaFile'], sim_dict['CaFile'], 
                                step, sim_win,sta_step,sta_win, 
                                agg_colname='Elec_distance_nexusQ', ca_spk=True)

    pois_ca_sta['nmda_a'] = pois_ca_sta['nmda_sta'].loc['apic','sta']
    pois_ca_sta['na_a'] = pois_ca_sta['na_sta'].loc['apic','sta']
    return pois_ca_sta

def plot_dspikes_relative_to_Ca_spikes(pois_ca_sta, figures_folder):
    fig,ax = plt.subplots(2,1)
    drew_functions.plot_sta_im(pois_ca_sta['nmda_a'], ax[0])
    ax[0].set_title('NMDA Apical')
    drew_functions.plot_sta_im(pois_ca_sta['na_a'], ax[1])
    ax[1].set_title('Na Apical')

    fig.supxlabel('Time from Ca spike (ms)')
    fig.supylabel('Electrotonic quantile 90->0 from nexus')
    fig.suptitle('Coordination of dendritic events and Ca spikes')
    fig.tight_layout()
    # fig.savefig('../figures/DendSpikesCaCC.pdf')
    fig.savefig(os.path.join(figures_folder,'DendSpikesCaCC.png'))

def process_NMDA_CA_coordination(sim_dict, step, sim_win, sta_step, sta_win, samps_per_ms):
    # get spkikes that are 20 ms after the start of a Ca spike
    caspk_t = load_caspks_csv(sim_dict['CaFile'], BEN=False)
    spk_t = load_spike_h5(sim_dict['APFile'])

    lag_win = samps_per_ms * 20
    spk_ca_t = [curr_spk for curr_spk in spk_t if np.any(((curr_spk-caspk_t)<lag_win)&((curr_spk-caspk_t)>=0))]


    pois_canmda_sta = {}
    pois_canmda_sta['nmda_sta'] = sta_files(sim_dict['NMDAFile'], spk_ca_t, 
                                    step, sim_win,sta_step,sta_win, 
                                    agg_colname='Elec_distanceQ')
    pois_canmda_sta['na_sta'] = sta_files(sim_dict['NaFile'], spk_ca_t, 
                                    step, sim_win,sta_step,sta_win, 
                                    agg_colname='Elec_distanceQ')

    pois_canmda_sta['nmda_a'] = pois_canmda_sta['nmda_sta'].loc['apic','sta']
    pois_canmda_sta['nmda_b'] = pois_canmda_sta['nmda_sta'].loc['dend','sta']
    pois_canmda_sta['na_a'] = pois_canmda_sta['na_sta'].loc['apic','sta']
    pois_canmda_sta['na_b'] = pois_canmda_sta['na_sta'].loc['dend','sta']
    return pois_canmda_sta

def plot_NMDA_CA_coordination(pois_canmda_sta, pois_sta, figures_folder):
    # numpy restrict values to set range
    def prc_chg(chg_map,base_map):
        # set denom values less than 10 to nan
        
        #base_map[abs(base_map)<10] = np.nan
        #out_map = np.clip(chg_map/base_map,-5,5)
        out_map = np.clip((chg_map-base_map)/np.abs(base_map), -5, 5)*100

        return out_map

    fig,ax = plt.subplots(2,1)
    drew_functions.plot_sta_im(prc_chg(pois_canmda_sta['nmda_a'],pois_sta['nmda_a']), ax[0])
    ax[0].set_title('NMDA Apical')
    drew_functions.plot_sta_im(prc_chg(pois_canmda_sta['nmda_b'],pois_sta['nmda_b']), ax[1])
    ax[1].set_title('NMDA Basal')
    fig.supxlabel('Time from action potential (ms)')
    fig.supylabel('Electrotonic quantile 90->0 from nexus')
    fig.suptitle('Percent change in NMDA/AP coordination during Ca spikes')
    fig.tight_layout()
    # fig.savefig('../figures/NMDAAPCC_modbyCa.pdf')
    fig.savefig(os.path.join(figures_folder,'NMDAAPCC_modbyCa.png'))

if __name__ == "__main__":
    # specifiy the simulation directories to analyze
    # sims_dir = "/home/drfrbc/Neural-Modeling/simulations/2025-10-16-07-49-IncreaseNexusMaxYTo900" 
    # sims_dir = "/home/drfrbc/Neural-Modeling/scripts/L5BaselineResults"
    
    if '-d' in sys.argv: # specify single sim directory
        d_index = sys.argv.index('-d') + 1
        sim_dir = sys.argv[d_index]
        sim_dirs = [sim_dir]
    elif '-f' in sys.argv: # specify folder of sim directories
        f_index = sys.argv.index('-f') + 1
        sims_dir = sys.argv[f_index]
        sim_dirs = [
            os.path.join(sims_dir, d)
            for d in sorted(os.listdir(sims_dir))
            if os.path.isdir(os.path.join(sims_dir, d))
        ]
    else: # specify manually in code
        sim_dirs = ["/home/drfrbc/Neural-Modeling/simulations/2025-10-17-13-58-SetNexusExc0.15from0.25_DecreaseNexusInh_resetTrunkOblique/allinh_rhythmic_depth_0.00_Np5000"]

    for sim_dir in sim_dirs:
        add_elec_quantiles(sim_dir)
        dspike_analysis(sim_dir)