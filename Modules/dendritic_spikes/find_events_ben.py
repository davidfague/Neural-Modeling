'''
scripts/find_events_ben.py
'''
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
import Modules.post_sim.analysis as analysis

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

        base_path = os.path.abspath("../scripts/")
        sys.path.append(base_path)
        sys.path.append(os.path.join(base_path, "Modules/"))
        sim_directory = os.path.join(base_path, sim_directory)
        parameters = analysis.DataReader.load_parameters(sim_directory)
        na = analysis.DataReader.read_data(sim_directory, "gNaTa_t_NaTa_t").T
        spks = analysis.DataReader.read_data(sim_directory, "soma_spikes")
        v = analysis.DataReader.read_data(sim_directory, "v").T
        if os.path.exists(os.path.join(sim_directory, f"raw_data/saved_at_step_{int(parameters.save_every_ms / parameters.h_dt)}", "ica_Ca_HVA" + ".h5")) and os.path.exists(os.path.join(sim_directory, f"raw_data/saved_at_step_{int(parameters.save_every_ms / parameters.h_dt)}", "ica_Ca_HVA" + ".h5")):
            hva = analysis.DataReader.read_data(sim_directory, "ica_Ca_HVA").T
            lva = analysis.DataReader.read_data(sim_directory, "ica_Ca_LVAst").T
        else:
            # print(f"[scripts/find_events_ben.py] Error loading HVA/LVA data: {e}")
            # Warning("[scripts/find_events_ben.py] Falling back onto ica. setting HVA = ica and LVA=zeros (easy fix since their sum will be used later anyway.)")
            hva = analysis.DataReader.read_data(sim_directory, "ica").T
            lva = np.zeros(hva.shape)
        ih = analysis.DataReader.read_data(sim_directory, "ihcn_Ih").T
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
        segs = pd.read_csv(os.path.join(sim_directory, 'Segments.csv'))
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
    segs_na_df.to_csv(os.path.join(sim_directory, 'na.csv'))

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

    # print(f"ca_df_list: {ca_df_list}")
    # Concatenate all DataFrames in the list into a single DataFrame
    ca_df = pd.concat(ca_df_list, ignore_index=True)        

    ca_df.reset_index(inplace=True, drop=True)
    segs_ca_df = segs.set_index('segmentID').join(ca_df.set_index('segmentID')).reset_index()

    segs_ca_df.to_csv(os.path.join(sim_directory, 'ca.csv'))

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
    segs_nmda_df.to_csv(os.path.join(sim_directory, 'nmda.csv'))
    
def compute_dfs(sim_directory, ben):
    if not os.path.exists(os.path.join(sim_directory, 'na.csv')) or not os.path.exists(os.path.join(sim_directory, 'ca.csv')) or not os.path.exists(os.path.join(sim_directory, 'nmda.csv')):
        na, hva, lva, ih, nmda, v, spkinds, segs = load_data(sim_directory, ben)
    else:
        print(f"[scripts/find_events_ben.py] DataFrames already exist in {sim_directory}. Skipping computation.")
        return # skip rest of the function

    if not os.path.exists(os.path.join(sim_directory, 'na.csv')):
        compute_na_df(na, segs, spkinds, sim_directory, ben)
    else:
        print(f"[scripts/find_events_ben.py] na.csv already exists in {sim_directory}. Skipping computation.")

    if not os.path.exists(os.path.join(sim_directory, 'ca.csv')):
        try:
            compute_ca_df(v, hva, lva, ih, segs, sim_directory, ben)
        except Exception as e:
            print(f"[scripts/find_events_ben.py] Error computing CA DataFrame (Likely due to no segments meeting the coordinates criteria  if this is L2/3 instead of L5): {e}")
    else:
        print(f"[scripts/find_events_ben.py] ca.csv already exists in {sim_directory}. Skipping computation.")

    if not os.path.exists(os.path.join(sim_directory, 'nmda.csv')):
        compute_nmda_df(nmda, v, segs, sim_directory, ben)
    else:
        print(f"[scripts/find_events_ben.py] nmda.csv already exists in {sim_directory}. Skipping computation.")
    
    print(f"[scripts/find_events_ben.py] DataFrames computed and saved to {sim_directory}")

if __name__ ==  "__main__":
    ben = False
    # Check for --ben flag to enable Ben's data format
    if "--ben" in sys.argv:
        ben = True
        sys.argv.remove("--ben")  # Remove flag so it doesn't interfere with other parsing
    
    if "-d" in sys.argv:
        sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
        compute_dfs(sim_directory, ben)
    elif "-f" in sys.argv:
        simulations_directory = sys.argv[sys.argv.index("-f") + 1]
        print(f"[scripts/find_events_ben.py] simulations_directory: {simulations_directory}")
        for sim_directory in os.listdir(simulations_directory):
            full_path_sim = os.path.join(simulations_directory, sim_directory)
            print(f"[scripts/find_events_ben.py] sim_directory: {sim_directory}")
            if os.path.exists(os.path.join(full_path_sim, 'parameters.pickle')):
                compute_dfs(full_path_sim, ben)
            else:
                print(f"[scripts/find_events_ben.py] skipping directory because no parameters (likely an analysis folder instead of simulation): {full_path_sim}")
    else:
        raise RuntimeError
    