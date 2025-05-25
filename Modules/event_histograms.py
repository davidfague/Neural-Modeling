# Note that this script does some additional event filtering.
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../Modules/")
import analysis
import os
import matplotlib
from matplotlib.colors import Normalize
import pandas as pd

def load_data(sim_directory, ben=False):
    # load data from simulation
    if ben:
        dur_to_use = 150
        soma_id_to_use = 2
        # %cd ../scripts/L5BaselineResults/ % just replaced this with the full path to the data. path could be wrong by chance.
        # v = np.array(h5py.File('/scripts/L5BaselineResults/v_report.h5','r')['report']['biophysical']['data'])
        # hva = np.array(h5py.File('/scripts/L5BaselineResults/Ca_HVA.ica_report.h5','r')['report']['biophysical']['data'])
        # lva = np.array(h5py.File('/scripts/L5BaselineResults/Ca_LVAst.ica_report.h5','r')['report']['biophysical']['data'])
        # ih = np.array(h5py.File('/scripts/L5BaselineResults/Ih.ihcn_report.h5','r')['report']['biophysical']['data'])
        # nmda = np.array(h5py.File('/scripts/L5BaselineResults/inmda_report.h5','r')['report']['biophysical']['data'])
        na = np.array(h5py.File('/scripts/L5BaselineResults/NaTa_t.gNaTa_t_report.h5','r')['report']['biophysical']['data'])
        spks = h5py.File('/scripts/L5BaselineResults/spikes.h5','r')
        spktimes = spks['spikes']['biophysical']['timestamps'][:]
        spkinds = np.sort((spktimes*10).astype(int))

        na_df = pd.read_csv('/scripts/L5BaselineResults/na.csv')
        ca_df = pd.read_csv('/scripts/L5BaselineResults/ca.csv')
        nmda_df = pd.read_csv('/scripts/L5BaselineResults/nmda.csv')
    else:
        soma_id_to_use = 0
        # sim_directory = '2024-10-12-00-28-34-ZiaoSynapses_final_detailed_refactored150sec/Complex_InhGmaxApic204_InhGmaxDend7.0_SomaGmax6.0_ExcGmax-1.0351_Np1000' # ziao synapses
        # sim_directory = '2024-10-11-14-32-54-BenSynapses_final_detailed150sec/Complex_InhGmaxApic7.1_InhGmaxDend0.0016_SomaGmax0.0025_ExcGmax-1.0351_Np1000/' # ben synapses refactored detailed 150 sec
        #'2024-08-29-12-19-13-CheckdSpikes_AfterTuningSynapses_AfterUpdateExcRates/Complex_InhGmaxApic204_InhGmaxDend7.0_SomaGmax6.0_ExcGmax-1.0351_Np1000'
        #'2024-08-09-09-19-21-2500Segs_TripleNa/Complex_Np5'
        # sim_directory = '2024-07-24-15-59-37-STA/Complex_Np5'33
    #'2024-07-12-12-17-52-STA/Complex_Np5'
        # %cd ../scripts/
        dur_to_use = analysis.DataReader.load_parameters(sim_directory).h_tstop / 1000
        # nmda = analysis.DataReader.read_data(sim_directory, "i_NMDA").T
        # nmda = analysis.DataReader.read_data(sim_directory, "inmda").T
        na = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t").T
        spks = analysis.DataReader.read_data(sim_directory, "soma_spikes")
        # v = analysis.DataReader.read_data(sim_directory, "v").T
        # hva = analysis.DataReader.read_data(sim_directory, "ica_Ca_HVA").T
        # lva = analysis.DataReader.read_data(sim_directory, "ica_Ca_LVAst").T
        # ih = analysis.DataReader.read_data(sim_directory, "ihcn_Ih").T
        # nmda = analysis.DataReader.read_data(sim_directory, "i_NMDA").T
        # na = analysis.DataReader.read_data(sim_directory, "na")
        spktimes = spks[0][:]
        spkinds = np.sort((spktimes*10).astype(int))
        # print(f"spkinds: {spkinds}")

        import os
        na_df = pd.read_csv(os.path.join(sim_directory,'na.csv'))
        ca_df = pd.read_csv(os.path.join(sim_directory,'ca.csv'))
        nmda_df = pd.read_csv(os.path.join(sim_directory,'nmda.csv'))
        # return v, hva, lva, ih, nmda, na, spktimes, spkinds, na_df, ca_df, nmda_df, dur_to_use, soma_id_to_use
        return na, spkinds, na_df, ca_df, nmda_df, dur_to_use, soma_id_to_use 

def get_segs(sim_directory, ben=False):
    # get segs from csv
    if ben:
        # segs = pd.read_csv('/Volumes/TOSHIBA EXT/L5NeuronSimulation_new/L5NeuronSimulation/MorphAnalysis/Segments.csv')
        segs = pd.read_csv('Segments.csv')
        segs_degrees = pd.read_csv('SegmentsDegrees.csv').groupby(['Type','Sec ID'])['Degrees'].max().reset_index()
        segs['segmentID'] = segs.index
        segs = segs.set_index(['Type','Sec ID']).join(segs_degrees.set_index(['Type','Sec ID'])).reset_index()

        segs['Sec ID'] = segs['Sec ID'].astype(int)
        segs['X'] = segs['X'].astype(float)
        segs['Elec_distanceQ'] = 'None'

        segs.loc[segs.Type=='dend','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='dend','Elec_distance'], 10, labels=False)
        segs.loc[segs.Type=='apic','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='apic','Elec_distance'], 10, labels=False)
    else:
        segs = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
        # make same
        segs['Sec ID'] = segs['idx_in_section_type']
        segs['Type'] = segs['section']
        segs['Coord X'] = segs['pc_0']
        segs['Coord Y'] = segs['pc_1']
        segs['Coord Z'] = segs['pc_2']
        elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{'soma'}.csv"))
        segs['Elec_distance'] = elec_dist['25_active']
        elec_dist = pd.read_csv(os.path.join(sim_directory, f"elec_distance_{'nexus'}.csv"))
        segs['Elec_distance_nexus'] = elec_dist['25_active']
        Xs = []
        for seg in segs['seg']:
            Xs.append(seg.split('(')[-1].split(')')[0])
        segs['X'] = Xs

        # continue
        segs['segmentID'] = segs.index

        segs['Sec ID'] = segs['Sec ID'].astype(int)
        segs['X'] = segs['X'].astype(float)
        segs['Elec_distanceQ'] = 'None'

        segs.loc[segs.Type=='dend','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='dend','Elec_distance'], 10, labels=False)
        segs.loc[segs.Type=='apic','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='apic','Elec_distance'], 10, labels=False)
    return segs

def post_process_na_df(na_df, na):
    # adds duration, peak_value, duration_low, duration_high to na_df
    max_retries = 1000  # Number of retries
    for _ in range(max_retries):
        try:
            for i in np.random.choice(na_df[(na_df.na_lower_bound > 20) & (na_df.na_lower_bound < 1400000)].index, 10000):
                seg = na_df.loc[i, 'segmentID']
                if not pd.isnull(na_df.loc[i, 'na_lower_bound']):
                    spkt = int(na_df.loc[i, 'na_lower_bound'])
                    # Ensure trace slicing does not go out of bounds
                    trace_start = max(0, spkt - 10)
                    trace_end = min(na.shape[0], spkt + 10)
                    trace = na[trace_start:trace_end, seg]
                    
                    if len(trace) == 20:  # Ensure the trace is the expected length
                        peak_value = np.max(trace)
                        half_peak = peak_value / 2
                        duration = np.arange(trace_start - (spkt - 10), trace_end - (spkt - 10))[trace > half_peak] + trace_start - (spkt - 10)
                        na_df.loc[i, 'duration_low'] = duration[0]
                        na_df.loc[i, 'duration_high'] = duration[-1]
                        na_df.loc[i, 'peak_value'] = peak_value
                    else:
                        raise ValueError(f"Trace length is {len(trace)} not 20, retrying...")
                else:
                    raise ValueError("Invalid na_lower_bound, retrying...")
            break  # Exit the retry loop if no errors occur
        except Exception as e:
            print(f"Retry {_ + 1}/{max_retries} failed: {e}")
    else:
        print("Maximum retries reached. Unable to process.")
    na_df['duration'] = (na_df['duration_high'] - na_df['duration_low'] + 1)/10
    return na_df

def post_process_nmda_df(nmda_df):
    nmda_df['duration'] = (nmda_df['nmda_upper_bound'] - nmda_df['nmda_lower_bound'])/10 #@davidfague need to know where this 10 comes from. presumably the time step?
    nmda_df['log_duration'] = np.log(nmda_df['duration'])
    nmda_df['log_mag'] = np.log(np.abs(nmda_df['mag']))
    return nmda_df

def post_process_ca_df(ca_df, spkinds):
    # print(f"ca_df: {ca_df}")
    # print(f"spkinds: {spkinds}")
    # ca_df['dist_from_soma_spike'] = ca_df['ca_lower_bound'].apply(lambda x: np.min(np.abs(x-spkinds))) # could be no soma spikes.
    ca_df['dist_from_soma_spike'] = ca_df['ca_lower_bound'].apply(
        lambda x: np.min(np.abs(x - spkinds)) if spkinds.size > 0 else np.nan
    )
    ca_df['duration'] = (ca_df['ca_upper_bound'] - ca_df['ca_lower_bound'])/10
    ca_df['mag_dur'] = ca_df['mag']/ca_df['duration']
    ca_df = ca_df[(ca_df.mag<-0.1)&
                            (ca_df.duration<250)&
                            (ca_df.duration>26)&
                            (ca_df.dist_from_soma_spike>50)&
                            (ca_df.mag_dur<-0.006)]
    return ca_df

def plot_na_prop_scatter(sim_directory, na_df):
    # plot scatter of duration vs peak value
    h = sns.jointplot(data=na_df[~pd.isnull(na_df.duration_low)], x="duration", y="peak_value",color='red')
    h.set_axis_labels('duration (ms)', 'peak value (mS/cm^2)', fontsize=16)
    h.ax_marg_y.set_ylim(0, 0.02)
    plt.title('Na')
    plt.savefig(f'{sim_directory}/spike_properties/na_prop_scatter.svg')

def plot_na_prop_heatmap(sim_directory, na_df):
    # plot heatmap of duration vs peak value
    na_df_bin = na_df[~pd.isnull(na_df.duration_low)].reset_index(drop=True)

    na_df_bin['duration_bin'] = pd.cut(na_df_bin['duration'], bins = np.arange(0.15,1.05,0.1), labels=False)
    na_df_bin['mag_bin'] = pd.cut(na_df_bin['peak_value'], bins = 2*np.logspace(-3,-2,num=15), labels=False)

    na_df_gb = na_df_bin.groupby(['duration_bin','mag_bin'])['duration'].count().reset_index()

    na_df_imhist = np.zeros((15,15))
    for i in np.arange(0,15):
        for j in np.arange(0,15):
            try:
                na_df_imhist[i,j] = na_df_gb[(na_df_gb.duration_bin==j) & (na_df_gb.mag_bin==i)]['duration']
            except:
                na_df_imhist[i,j] = 0

    plt.figure(figsize=(6,6))      
    plt.imshow(100 * na_df_imhist / na_df_imhist.sum(), origin = 'lower')
    plt.xlabel('duration (ms)', fontsize = 16)
    plt.ylabel('magnitude (mS/cm^2)', fontsize = 16)
    plt.xticks(ticks = [0,4,8,12], labels = [0.15, 0.55, 0.95, 1.35])
    plt.title('Na')
    #plt.yticks(ticks = [0,4,8,12], labels = [2e-3, 0.00386, 0.00746, 0.0144])
    plt.colorbar(label='% of events')
    plt.savefig(f'{sim_directory}/spike_properties/na_prop_heatmap.svg')
    #plt.xlim(0,10)
    # plt.show()

def plot_na_spk_locations(sim_directory, na_df, segs, dur_to_use, soma_id_to_use):
    seg_na_df = na_df.groupby('segmentID')['na_lower_bound'].count().reset_index().rename(columns={'na_lower_bound':'num_na_spikes'})
    segs_na_df = segs.set_index('segmentID').join(seg_na_df.set_index('segmentID'))
    segs_na_df.loc[segs_na_df.num_na_spikes>1000,'num_na_spikes'] = 1000
    color_field = 'num_na_spikes'

    fig, ax = plt.subplots(figsize=(2, 5))
    cmap = matplotlib.colormaps.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin = 0, vmax = 5)

    for i in segs_na_df[segs_na_df.Type=='apic']['Sec ID'].unique():
        section = segs_na_df[(segs_na_df.Type=='apic')&(segs_na_df['Sec ID']==i)]
        for j in section.index.tolist()[:-1]:
            # print(type(section.loc[j:j+1, 'Coord X']), section.loc[j:j+1, 'Coord X'])
            # print(type(section.loc[j:j+1, 'Coord Y']), section.loc[j:j+1, 'Coord Y'])
            ax.plot(section.loc[j:j+1,'Coord X'].values,
                    section.loc[j:j+1,'Coord Y'].values,
                color=cmap(norm(section.loc[j:j+1,color_field].mean()/dur_to_use)),
                linewidth = 1*section.loc[j:j+1,'Section_diam'].mean())
        
    for i in segs_na_df[segs_na_df.Type=='dend']['Sec ID'].unique():
        section = segs_na_df[(segs_na_df.Type=='dend')&(segs_na_df['Sec ID']==i)]
        for j in section.index.tolist()[:-1]:
            # print(type(section.loc[j:j+1, 'Coord X']), section.loc[j:j+1, 'Coord X'])
            # print(type(section.loc[j:j+1, 'Coord Y']), section.loc[j:j+1, 'Coord Y'])
            ax.plot(section.loc[j:j+1,'Coord X'],
                    section.loc[j:j+1,'Coord Y'],
                color=cmap(norm(section.loc[j:j+1,color_field].mean()/dur_to_use)),
                linewidth = 1*section.loc[j:j+1,'Section_diam'].mean())
            
    ax.scatter(segs_na_df[(segs_na_df.Type=='soma')&(segs_na_df['Sec ID']==0)].loc[soma_id_to_use,'Coord X'],
            segs_na_df[(segs_na_df.Type=='soma')&(segs_na_df['Sec ID']==0)].loc[soma_id_to_use,'Coord Y'],color='k',s=100)
    ax.vlines(110,400,500)
    ax.text(0,450,'100 um')
    ax.hlines(400,110,210)
    ax.text(110,350,'100 um')
    # Create a dummy mappable for the colorbar
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a dummy mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the mappable
    cbar = plt.colorbar(sm, ax=ax)
    #cbar.ax.set_ylabel('log(elec_distance)', rotation=270)

    #ax2.ax.set_title('log(elec_distance)',rotation=270)
    plt.box(False)
    plt.title('Na')
    plt.savefig(f'{sim_directory}/spike_properties/na_spk_locations.svg')
    return segs_na_df

def plot_nmda_prop_scatter(sim_directory, nmda_df):
    # plot scatter of duration vs magnitude
    h = sns.jointplot(data=nmda_df[(nmda_df.mag<-0.1)&
                                (nmda_df.duration<250)&
                                (nmda_df.duration>26)], x="duration", y="mag",alpha=0.02, color='blue')
    h.set_axis_labels('duration (ms)', 'magnitude (nA ms)', fontsize=16)
    plt.title('NMDA')
    plt.savefig(f'{sim_directory}/spike_properties/nmda_prop_scatter.svg')

def plot_nmda_prop_heatmap(sim_directory, nmda_df):
    nmda_df_bin = nmda_df[(nmda_df.mag<-0.1)&(nmda_df.duration<250)&(nmda_df.duration>26)].copy()

    nmda_df_bin['duration_bin'] = pd.cut(nmda_df_bin['duration'], bins = 2*np.logspace(1.1,1.8,num=15), labels=False)
    nmda_df_bin['mag_bin'] = pd.cut(-nmda_df_bin['mag'], bins = np.logspace(-1,0.5,num=15), labels=False)

    nmda_df_gb = nmda_df_bin.groupby(['duration_bin','mag_bin'])['duration'].count().reset_index()

    nmda_df_imhist = np.zeros((15,15))
    for i in np.arange(0,15):
        for j in np.arange(0,15):
            try:
                nmda_df_imhist[i,j] = nmda_df_gb[(nmda_df_gb.duration_bin==j) & (nmda_df_gb.mag_bin==i)]['duration']
            except:
                nmda_df_imhist[i,j] = 0
    plt.figure(figsize=(6,6))   
    plt.imshow(100 * nmda_df_imhist / nmda_df_imhist.sum(), origin = 'lower')
    plt.xlabel('duration (ms)', fontsize = 16)
    plt.ylabel('magnitude (nA ms)', fontsize = 16)
    plt.xticks(ticks = [0,5,10,14], labels = [25, 45, 80, 126])
    plt.yticks(ticks = [0,4,8,12], labels = [0.10, 0.27, 0.72, 1.93])
    plt.colorbar(label='% of events')
    plt.title('NMDA')
    plt.savefig(f'{sim_directory}/spike_properties/nmda_prop_heatmap.svg')
    # plt.show()

def plot_nmda_spk_locations(sim_directory, nmda_df, segs, dur_to_use, soma_id_to_use):
    seg_nmda_df = nmda_df.groupby('segmentID')['nmda_lower_bound'].count().reset_index().rename(columns={'nmda_lower_bound':'num_nmda_spikes'})
    segs_nmda_df = segs.set_index('segmentID').join(seg_nmda_df.set_index('segmentID'))
    color_field = 'num_nmda_spikes'

    fig, ax = plt.subplots(figsize=(2, 5))
    cmap = matplotlib.colormaps.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin = 0, vmax = 10)

    for i in segs_nmda_df[segs_nmda_df.Type=='apic']['Sec ID'].unique():
        section = segs_nmda_df[(segs_nmda_df.Type=='apic')&(segs_nmda_df['Sec ID']==i)]
        for j in section.index.tolist()[:-1]:
            ax.plot(section.loc[j:j+1,'Coord X'],
                    section.loc[j:j+1,'Coord Y'],
                color=cmap(norm(section.loc[j:j+1,color_field].mean()/dur_to_use)),
                linewidth = 1*section.loc[j:j+1,'Section_diam'].mean())
        
    for i in segs_nmda_df[segs_nmda_df.Type=='dend']['Sec ID'].unique():
        section = segs_nmda_df[(segs_nmda_df.Type=='dend')&(segs_nmda_df['Sec ID']==i)]
        for j in section.index.tolist()[:-1]:
            ax.plot(section.loc[j:j+1,'Coord X'],
                    section.loc[j:j+1,'Coord Y'],
                color=cmap(norm(section.loc[j:j+1,color_field].mean()/dur_to_use)),
                linewidth = 1*section.loc[j:j+1,'Section_diam'].mean())
            
    ax.scatter(segs_nmda_df[(segs_nmda_df.Type == 'soma') & (segs_nmda_df['Sec ID'] == 0)].loc[soma_id_to_use, 'Coord X'],
            segs_nmda_df[(segs_nmda_df.Type == 'soma') & (segs_nmda_df['Sec ID'] == 0)].loc[soma_id_to_use, 'Coord Y'],
            color='k', s=100)
    ax.vlines(110, 400, 500)
    ax.text(0, 450, '100 um')
    ax.hlines(400, 110, 210)
    ax.text(110, 350, '100 um')
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a dummy mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the mappable
    cbar = plt.colorbar(sm, ax=ax)
    plt.box(False)
    plt.title('NMDA')
    plt.savefig(f'{sim_directory}/spike_properties/nmda_spk_locations.svg')
    return segs_nmda_df

def plot_ca_prop_heatmap(sim_directory, ca_df):
    ca_df_bin = ca_df[(ca_df.mag<-0.1)&
                           (ca_df.duration<250)&
                           (ca_df.duration>26)&
                           (ca_df.dist_from_soma_spike>50)&
                           (ca_df.mag_dur<-0.006)].copy()

    ca_df_bin['duration_bin'] = pd.cut(ca_df_bin['duration'], bins = 2*np.logspace(1.1,1.5,num=15), labels=False)
    ca_df_bin['mag_bin'] = pd.cut(-ca_df_bin['mag'], bins = np.linspace(0.1,1.4,num=15), labels=False)

    ca_df_gb = ca_df_bin.groupby(['duration_bin','mag_bin'])['duration'].count().reset_index()

    ca_df_imhist = np.zeros((15,15))
    for i in np.arange(0,15):
        for j in np.arange(0,15):
            try:
                ca_df_imhist[i,j] = ca_df_gb[(ca_df_gb.duration_bin==j) & (ca_df_gb.mag_bin==i)]['duration']
            except:
                ca_df_imhist[i,j] = 0

    plt.figure(figsize=(6,6))   
    plt.imshow(100 * ca_df_imhist / ca_df_imhist.sum(), origin = 'lower')
    plt.xlabel('duration (ms)', fontsize = 16)
    plt.ylabel('magnitude (nA ms)', fontsize = 16)
    plt.xticks(ticks = [0,5,10,14], labels = [25, 35, 49, 63])
    plt.yticks(ticks = [0,4,8,12], labels = [0.10, 0.47, 0.84, 1.21])
    plt.colorbar(label='% of events')
    plt.title('Ca')
    plt.savefig(f'{sim_directory}/spike_properties/ca_prop_heatmap.svg')
    plt.show()

def plot_ca_prop_scatter(sim_directory, ca_df):
    h = sns.jointplot(data=ca_df, x="duration", y="mag",alpha=0.1, color='black')
    h.set_axis_labels('duration (ms)', 'magnitude (nA ms)', fontsize=16)
    plt.title('Ca')
    plt.savefig(f'{sim_directory}/spike_properties/ca_prop_scatter.svg')

def plot_ca_spk_locations(sim_directory, ca_df, segs, dur_to_use, ben):
    ca_df = ca_df[(ca_df.mag<-0.1)&
                           (ca_df.duration<250)&
                           (ca_df.duration>26)&
                           (ca_df.dist_from_soma_spike>50)]

    seg_ca_df = ca_df.groupby('segmentID')['ca_lower_bound'].count().reset_index().rename(columns={'ca_lower_bound':'num_ca_spikes'})
    segs_ca_df = segs.set_index('segmentID').join(seg_ca_df.set_index('segmentID')).reset_index()
    segs_ca_df_plot = segs_ca_df.copy()
    segs_ca_df_plot[(segs_ca_df_plot.Type=='soma')&(segs_ca_df_plot['Sec ID']==0)]
    color_field = 'num_ca_spikes'
    if ben: soma_id_to_use_ca = 2523 
    else: soma_id_to_use_ca =  0

    fig, ax = plt.subplots(figsize=(2,5))
    cmap = matplotlib.colormaps.get_cmap('jet')
    norm = matplotlib.colors.Normalize(vmin = 0, vmax = 2)

    for i in segs_ca_df_plot[segs_ca_df_plot.Type=='apic']['Sec ID'].unique():
        section = segs_ca_df_plot[(segs_ca_df_plot.Type=='apic')&(segs_ca_df_plot['Sec ID']==i)]
        for j in section.index.tolist()[:-1]:
            ax.plot(section.loc[j:j+1,'Coord X'],
                    section.loc[j:j+1,'Coord Y'],
                color=cmap(norm(section.loc[j:j+1,color_field].mean()/dur_to_use)),
                linewidth = 1*section.loc[j:j+1,'Section_diam'].mean())
        
    for i in segs_ca_df_plot[segs_ca_df_plot.Type=='dend']['Sec ID'].unique():
        section = segs_ca_df_plot[(segs_ca_df_plot.Type=='dend')&(segs_ca_df_plot['Sec ID']==i)]
        for j in section.index.tolist()[:-1]:
            ax.plot(section.loc[j:j+1,'Coord X'],
                    section.loc[j:j+1,'Coord Y'],
                color=cmap(norm(section.loc[j:j+1,color_field].mean()/dur_to_use)),
                linewidth = 1*section.loc[j:j+1,'Section_diam'].mean())
            
    ax.scatter(segs_ca_df_plot[(segs_ca_df_plot.Type=='soma')&(segs_ca_df_plot['Sec ID']==0)].loc[soma_id_to_use_ca,'Coord X'],
            segs_ca_df_plot[(segs_ca_df_plot.Type=='soma')&(segs_ca_df_plot['Sec ID']==0)].loc[soma_id_to_use_ca,'Coord Y'],color='k',s=100)
    ax.vlines(110,400,500)
    ax.text(0,450,'100 um')
    ax.hlines(400,110,210)
    ax.text(110,350,'100 um')
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a dummy mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the mappable
    cbar = plt.colorbar(sm, ax=ax)
    #cbar.ax.set_ylabel('log(elec_distance)', rotation=270)

    #ax2.ax.set_title('log(elec_distance)',rotation=270)
    plt.box(False)
    plt.title('Ca')
    plt.savefig(f'{sim_directory}/spike_properties/ca_spk_locations.svg')
    return segs_ca_df

def generate_spk_table(sim_directory, dur_to_use, segs_nmda_df, segs_na_df, segs_ca_df):
    # Initialize an empty list to collect rows before converting them into a DataFrame
    rows = []

    # Define the spike types and their corresponding DataFrames and columns
    spike_types = {
        'num_nmda_spikes': ('Total_NMDA_Spikes', segs_nmda_df),
        'num_na_spikes': ('Total_NA_Spikes', segs_na_df),
        'num_ca_spikes': ('Total_CA_Spikes', segs_ca_df)
    }

    # Calculate the total number of spikes for each segment type and spike type
    for seg_type in ['apic', 'dend']:
        row = {'Segment_Type': seg_type}
        for spike_field, (column_name, df) in spike_types.items():
            total_spikes = df[df.Type == seg_type][spike_field].sum()
            row[column_name] = round(total_spikes / dur_to_use, 1)
        rows.append(row)

    # Convert the list of rows into a DataFrame
    spike_table = pd.DataFrame(rows)

    spike_table.to_csv(os.path.join(sim_directory, 'dSpike_table.csv'))

    # # Display the table
    # print(spike_table)

def generate_figs_for_simulation(sim_directory):
        if not os.path.exists(f'{sim_directory}/spike_properties'):
            os.makedirs(f'{sim_directory}/spike_properties')
        # v, hva, lva, ih, nmda, na, spktimes, spkinds, na_df, ca_df, nmda_df, dur_to_use, soma_id_to_use = load_data(sim_directory)
        na, spkinds, na_df, ca_df, nmda_df, dur_to_use, soma_id_to_use = load_data(sim_directory)
        segs = get_segs(sim_directory)

        # na
        na_df = post_process_na_df(na_df, na)
        plot_na_prop_scatter(sim_directory, na_df)
        plot_na_prop_heatmap(sim_directory, na_df)
        segs_na_df = plot_na_spk_locations(sim_directory, na_df, segs, dur_to_use, soma_id_to_use)

        # nmda
        nmda_df = post_process_nmda_df(nmda_df)
        plot_nmda_prop_scatter(sim_directory, nmda_df)
        plot_nmda_prop_heatmap(sim_directory, nmda_df)
        segs_nmda_df = plot_nmda_spk_locations(sim_directory, nmda_df, segs, dur_to_use, soma_id_to_use)

        # ca
        ca_df = post_process_ca_df(ca_df, spkinds)
        plot_ca_prop_heatmap(sim_directory, ca_df)
        plot_ca_prop_scatter(sim_directory, ca_df)
        segs_ca_df = plot_ca_spk_locations(sim_directory, ca_df, segs, dur_to_use, soma_id_to_use)

        generate_spk_table(sim_directory, dur_to_use, segs_nmda_df, segs_na_df, segs_ca_df)

if __name__ == '__main__':
    # get sim_directories from command line
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
        generate_figs_for_simulation(sim_directory)
    elif "-f" in sys.argv:
        simulations_directory = sys.argv[sys.argv.index("-f") + 1]
        print(f"simulations_directory: {simulations_directory}")
        for sim_directory in os.listdir(simulations_directory):
            full_path_sim = os.path.join(simulations_directory, sim_directory)
            print(f"sim_directory: {sim_directory}")
            generate_figs_for_simulation(full_path_sim)
    else:
        raise RuntimeError
