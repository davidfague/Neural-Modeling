import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as ss
import os
import sys

# Add the parent directory of "Modules" to the system path
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../Modules"))
import Modules.analysis as analysis

def voltage_criterion(data=None, v_thresh=-40, time_thresh=260):
    threshold_crossings = np.diff(data > v_thresh, prepend=False)
    upward_crossings = np.argwhere(threshold_crossings)[::2,0]
    downward_crossings = np.argwhere(threshold_crossings)[1::2,0]
    # If length of threshold_crossings is not even
    if np.mod(np.argwhere(threshold_crossings).reshape(-1,).shape[0],2)!=0:
        legit_up_crossings = upward_crossings[:-1][np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2]>time_thresh]
        legit_down_crossings = downward_crossings[np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2]>time_thresh]
    else:
        legit_up_crossings = upward_crossings[np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2]>time_thresh]
        legit_down_crossings = downward_crossings[np.diff(np.argwhere(threshold_crossings).reshape(-1,))[::2]>time_thresh]
    return upward_crossings, legit_up_crossings, legit_down_crossings

# Input: upward and downward crossings
# Output: bounds of NMDA spikes meeting current criterion
def current_criterion(legit_uc_iso=[], legit_dc_iso=[], p=1, control_inmda=np.array([1])):
    bounds = []
    sum_current = []
    

    for ind1 in np.arange(0,len(legit_uc_iso)):
        e1 = control_inmda[legit_uc_iso[ind1], p] #current @ up_crossing[ind1]
        #all the indices where current crosses 130% of e1
        x30 = np.argwhere(np.diff(control_inmda[legit_uc_iso[ind1]:legit_dc_iso[ind1], p] < 1.3*e1, prepend=False))
        #all the indices where current crosses 115% of e1
        x15 = np.argwhere(np.diff(control_inmda[legit_uc_iso[ind1]:legit_dc_iso[ind1], p] < 1.15*e1, prepend=False))
        
        if len(x30)>0:
        
            x15_copy = x15
            x30_copy = x30
            
            try:
                i = x30[0][0]
            except:
                import pdb; pdb.set_trace()
                
            n = 0
            
            
            while n==0:
                if len(np.sort(x15[x15>i]))!=0:
                    b1 = i
                    b2 = np.sort(x15[x15>i])[0]
                    bounds.append([legit_uc_iso[ind1]+b1,legit_uc_iso[ind1]+b2])
                    sum_current.append(np.sum(control_inmda[legit_uc_iso[ind1]+b1:legit_uc_iso[ind1]+b2,p])/10)
                else:
                    b1 = i
                    b2 = (legit_dc_iso[ind1]-legit_uc_iso[ind1])
                    bounds.append([legit_uc_iso[ind1]+b1,legit_uc_iso[ind1]+b2])
                    sum_current.append(np.sum(control_inmda[legit_uc_iso[ind1]+b1:legit_uc_iso[ind1]+b2,p])/10)
                    n=1
                
                x30_copy = x30_copy[x30_copy>legit_uc_iso[ind1]+b2]
                #import pdb; pdb.set_trace()
                
                if len(x30_copy)!=0:
                    i = x30_copy[x30_copy>b2][0]
                else:
                    n=1
    return bounds, sum_current

def load_data(sim_directory, ben):
    # load simulated data
    if ben:
        base_path = os.path.abspath("../scripts/L5BaselineResults/")
        v = np.array(h5py.File(os.path.join(base_path, 'v_report.h5'), 'r')['report']['biophysical']['data'])
        hva = np.array(h5py.File(os.path.join(base_path, 'Ca_HVA.ica_report.h5'), 'r')['report']['biophysical']['data'])
        lva = np.array(h5py.File(os.path.join(base_path, 'Ca_LVAst.ica_report.h5'), 'r')['report']['biophysical']['data'])
        ih = np.array(h5py.File(os.path.join(base_path, 'Ih.ihcn_report.h5'), 'r')['report']['biophysical']['data'])
        nmda = np.array(h5py.File(os.path.join(base_path, 'inmda_report.h5'), 'r')['report']['biophysical']['data'])
        na = np.array(h5py.File(os.path.join(base_path, 'NaTa_t.gNaTa_t_report.h5'), 'r')['report']['biophysical']['data'])
        spks = h5py.File(os.path.join(base_path, 'spikes.h5'), 'r')
        spktimes = spks['spikes']['biophysical']['timestamps'][:]
        spkinds = np.sort((spktimes*10).astype(int))


    else:
        # sim_directory = 2024-10-11-14-32-54-BenSynapses_final_detailed150secComplex_InhGmaxApic7.1_InhGmaxDend0.0016_SomaGmax0.0025_ExcGmax-1.0351_Np1000/
        #'2024-10-10-15-46-14-BenSynapses_final_detailed/Complex_InhGmaxApic7.1_InhGmaxDend0.0016_SomaGmax0.0025_ExcGmax-1.0351_Np1000'
        #'2024-08-29-12-19-13-CheckdSpikes_AfterTuningSynapses_AfterUpdateExcRates/Complex_InhGmaxApic204_InhGmaxDend7.0_SomaGmax6.0_ExcGmax-1.0351_Np1000'
        #'2024-08-13-23-31-53-TuningSynapses_150%Na/Complex_InhGmax3.0_SomaGmax0.2_Np10'
    #'2024-08-02-08-31-54-STA/Complex_Np5'
        # sim_directory = '2024-07-24-17-33-39-STA/Complex_Np5'
    #'2024-07-24-15-59-37-STA/Complex_Np5'
    #'2024-07-12-12-17-52-STA/Complex_Np5'
        # os.chdir("../scripts/")
        base_path = os.path.abspath("../scripts/")
        sys.path.append(base_path)
        sys.path.append(os.path.join(base_path, "Modules/"))
        sim_directory = os.path.join(base_path, sim_directory)
        na = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t").T
        spks = analysis.DataReader.read_data(sim_directory, "soma_spikes")
        v = analysis.DataReader.read_data(sim_directory, "v").T
        hva = analysis.DataReader.read_data(sim_directory, "ica_Ca_HVA").T
        lva = analysis.DataReader.read_data(sim_directory, "ica_Ca_LVAst").T
        ih = analysis.DataReader.read_data(sim_directory, "ihcn_Ih").T
        parameters = analysis.DataReader.load_parameters(sim_directory)
        if parameters.exc_syn_mod == 'pyr2pyr': # two types with different variable name
            nmda = analysis.DataReader.read_data(sim_directory, "inmda").T
        else:
            nmda = analysis.DataReader.read_data(sim_directory, "i_NMDA").T
        # na = analysis.DataReader.read_data(sim_directory, "na")
        spktimes = spks[0][:]
        spkinds = np.sort((spktimes*10).astype(int))

    # load segment data
    if ben:
        # segs = pd.read_csv('DetailedSegmentsAxialR.csv')
        segs = pd.read_csv('Segments.csv')
        segs['segmentID'] = segs.index

        segs['Sec ID'] = segs['Sec ID'].astype(int)
        segs['X'] = segs['X'].astype(float)
        segs['Elec_distanceQ'] = 'None'

        segs.loc[segs.Type=='dend','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='dend','Elec_distance'], 10, labels=False)
        segs.loc[segs.Type=='apic','Elec_distanceQ'] = pd.qcut(segs.loc[segs.Type=='apic','Elec_distance'], 10, labels=False)
    else:
        # segs = pd.read_csv('DetailedSegmentsAxialR.csv')
        segs = pd.read_csv(os.path.join(sim_directory, "segment_data.csv"))
        # make same
        segs['Sec ID'] = segs['idx_in_section_type']
        segs['Type'] = segs['section']
        segs['Coord X'] = segs['pc_0']
        segs['Coord Y'] = segs['pc_1']
        segs['Coord Z'] = segs['pc_2']
        # segs['Coord X'] = segs.apply(lambda row: np.array([row['p0_0'], row['pc_0'], row['p1_0']]), axis=1)
        # segs['Coord Y'] = segs.apply(lambda row: np.array([row['p0_1'], row['pc_1'], row['p1_1']]), axis=1)
        # segs['Coord Z'] = segs.apply(lambda row: np.array([row['p0_2'], row['pc_2'], row['p1_2']]), axis=1)
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

        return na, hva, lva, ih, nmda, v, spkinds, segs

def compute_na_df(na, segs, spkinds, sim_directory, ben):
    na_df = pd.DataFrame(columns=['segmentID','na_lower_bound'])
    na_df_list = []  # Initialize a list to store individual DataFrames

    for p in segs[(segs.Type=='dend')|(segs.Type=='apic')].index:
        threshold_crossings = np.diff(na[:,p] > 0.003, prepend=False)#['report']['biophysical']['data'][:,p] > 0.003, prepend=False)
        upward_crossings = np.argwhere(threshold_crossings)[::2,0]
        downward_crossings = np.argwhere(threshold_crossings)[1::2,0]
        # Only count if not within 2 ms after a somatic spike
        na_spks = [i for i in upward_crossings if ~np.any((i-spkinds>=-5) & (i-spkinds<50))]
        
        if len(na_spks) > 0:
            na_df_list.append(pd.DataFrame({'segmentID': np.tile(p, len(na_spks)),
                                            'na_lower_bound': na_spks}))
        else:
            na_df_list.append(pd.DataFrame({'segmentID': [p],
                                            'na_lower_bound': [np.nan]}))

    # Concatenate all DataFrames in the list into a single DataFrame
    na_df = pd.concat(na_df_list, ignore_index=True)

    na_df.reset_index(inplace=True, drop=True)
    segs_na_df = segs.set_index('segmentID').join(na_df.set_index('segmentID')).reset_index()
    if ben:segs_na_df.to_csv('na.csv')
    else: segs_na_df.to_csv(os.path.join(sim_directory, 'na.csv'))

def compute_ca_df(v, hva, lva, ih, segs, sim_directory, ben):
    ca_df = pd.DataFrame(columns=['segmentID','ca_lower_bound'])
    segIDs = segs[(segs.Type=='apic')&(segs['Coord Y']>400)&(segs['Coord Y']<1000)]['segmentID']
    ca_df_list = []  # Initialize a list to store individual DataFrames

    for p in segIDs:
        trace = (hva[:,p] + #['report']['biophysical']['data'][:,p] + 
                lva[:,p] + #['report']['biophysical']['data'][:,p] + 
                ih[:,p]) #['report']['biophysical']['data'][:,p])
        m = np.mean(trace)
        s = np.std(trace)

        legit_uc = voltage_criterion(data=v[:-10000,p], v_thresh=-40, time_thresh=200)[1]#['report']['biophysical']['data'][:-10000,p], v_thresh=-40, time_thresh=200)[1]
        legit_dc = voltage_criterion(data=v[:-10000,p], v_thresh=-40, time_thresh=200)[-1]#['report']['biophysical']['data'][:-10000,p], v_thresh=-40, time_thresh=200)[-1]
        
        legit_uc_iso = legit_uc
        legit_dc_iso = legit_dc
        
        if (len(legit_uc_iso) != 0) & (np.min(trace) != 0):
            bnds, sum_curr = current_criterion(legit_uc_iso=legit_uc_iso, 
                                            legit_dc_iso=legit_dc_iso, 
                                            p=p, 
                                            control_inmda=hva)#['report']['biophysical']['data'])
            ca_df_list.append(pd.DataFrame({'segmentID': np.tile(p, len(bnds)),
                                            'ca_lower_bound': np.array(bnds).reshape(-1, 2)[:,0],
                                            'ca_upper_bound': np.array(bnds).reshape(-1, 2)[:,1],
                                            'mag': sum_curr}))
        else:
            ca_df_list.append(pd.DataFrame({'segmentID': np.tile(p, 1),
                                            'ca_lower_bound': [np.nan],
                                            'ca_upper_bound': [np.nan],
                                            'mag': [np.nan]}))

    # Concatenate all DataFrames in the list into a single DataFrame
    ca_df = pd.concat(ca_df_list, ignore_index=True)        

    ca_df.reset_index(inplace=True, drop=True)
    segs_ca_df = segs.set_index('segmentID').join(ca_df.set_index('segmentID')).reset_index()
    if ben:segs_ca_df.to_csv('ca.csv')
    else:segs_ca_df.to_csv(os.path.join(sim_directory,'ca.csv'))

def compute_nmda_df(nmda, v, segs, sim_directory, ben):
    nmda_df = pd.DataFrame(columns=['segmentID','nmda_lower_bound', 'nmda_upper_bound', 'mag'])
    nmda_df_list = []  # Initialize a list to store individual DataFrames

    for p in segs[(segs.Type=='dend') | (segs.Type=='apic')].index:
        legit_uc = voltage_criterion(data=v[:-10000, p], v_thresh=-40, time_thresh=260)[1]#['report']['biophysical']['data'][:-10000, p], v_thresh=-40, time_thresh=260)[1]
        legit_dc = voltage_criterion(data=v[:-10000, p], v_thresh=-40, time_thresh=260)[-1] #['report']['biophysical']['data'][:-10000, p], v_thresh=-40, time_thresh=260)[-1]
        
        legit_uc_iso = legit_uc
        legit_dc_iso = legit_dc
        
        if (len(legit_uc_iso) != 0) & (np.min(nmda[:, p]) != 0): #['report']['biophysical']['data'][:, p]) != 0):
            bnds, sum_curr = current_criterion(legit_uc_iso=legit_uc_iso, 
                                            legit_dc_iso=legit_dc_iso, 
                                            p=p, 
                                            control_inmda=nmda)#['report']['biophysical']['data'])
            nmda_df_list.append(pd.DataFrame({'segmentID': np.tile(p, len(bnds)),
                                            'nmda_lower_bound': np.array(bnds).reshape(-1, 2)[:,0],
                                            'nmda_upper_bound': np.array(bnds).reshape(-1, 2)[:,1],
                                            'mag': sum_curr}))
        else:
            nmda_df_list.append(pd.DataFrame({'segmentID': np.tile(p, 1),
                                            'nmda_lower_bound': [np.nan],
                                            'nmda_upper_bound': [np.nan],
                                            'mag': [np.nan]}))

    # Concatenate all DataFrames in the list into a single DataFrame
    nmda_df = pd.concat(nmda_df_list, ignore_index=True)

    nmda_df.rename(columns={'seg_id':'segmentID'},inplace=True)
    segs_nmda_df = segs.set_index('segmentID').join(nmda_df.set_index('segmentID')).reset_index()
    if ben: segs_nmda_df.to_csv('nmda.csv')
    else: segs_nmda_df.to_csv(os.path.join(sim_directory, 'nmda.csv'))

def compute_dfs(sim_directory, ben):
    na, hva, lva, ih, nmda, v, spkinds, segs = load_data(sim_directory, ben)
    compute_na_df(na, segs, spkinds, sim_directory, ben)
    compute_ca_df(v, hva, lva, ih, segs, sim_directory, ben)
    compute_nmda_df(nmda, v, segs, sim_directory, ben)

if __name__ ==  "__main__":
    ben = False
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
        compute_dfs(sim_directory, ben)
    elif "-f" in sys.argv:
        simulations_directory = sys.argv[sys.argv.index("-f") + 1]
        print(f"simulations_directory: {simulations_directory}")
        for sim_directory in os.listdir(simulations_directory):
            print(f"sim_directory: {sim_directory}")
            compute_dfs(os.path.join(simulations_directory, sim_directory), ben)
    else:
        raise RuntimeError
    