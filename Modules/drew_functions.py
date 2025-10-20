import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

t_list = None
eqt_list = None
t_inds = None
eqt_inds = None

def set_sta_axes(t_list_, eqt_list_, t_inds_, eqt_inds_):
    """Set global STA axis vectors/indices used by the plotting helpers."""
    global t_list, eqt_list, t_inds, eqt_inds
    t_list = np.asarray(t_list_)
    eqt_list = np.asarray(eqt_list_)
    t_inds = np.asarray(t_inds_)
    eqt_inds = np.asarray(eqt_inds_)

def _ensure_axes():
    if any(x is None for x in (t_list, eqt_list, t_inds, eqt_inds)):
        raise RuntimeError("Call drew_functions.set_sta_axes(...) before using plot_sta_im/plot_sta_stair.")


def join_nexus_atten(ds_fpath): # (nex_fpath, ds_fpath):
    """Join nexus attenuation to dendritic spike events.
    
    Parameters
    ----------
    nex_fpath : string
        path to nexus attenuation file
    ds_fpath : string
        path to dendritic spike events file
    
    Returns
    ----------
    new_ds_df : dataframe
        dendritic spike events with nexus attenuation added
        
    """
    # read in nexus attenuation file
    ds_df = pd.read_csv(ds_fpath)

    # # read in nexus attenuation file
    # nex_df = pd.read_csv(nex_fpath) # commented because we have the Elec_distance_nexus column in the ds file already

    
    # # match the 'Elec_distance_nexus' column to the corresponding 'X' and 'Sec ID' columns
    # new_ds_df = ds_df.merge(nex_df.loc[:,('X', 'Sec ID', 'Elec_distance')], 
                            # on=['X', 'Sec ID'], how='left')
    new_ds_df = ds_df
    
    # calculate rank of nexus attenuation within each 'Type' group
    new_ds_df['Elec_distance_nexusQ'] = new_ds_df.groupby('Type')['Elec_distance_nexus'].rank(pct=True).map(lambda x: int(x*10))

    # save to new dendritic spike events file with _nex.csv suffix
    new_ds_fpath = ds_fpath[:-4] + '_withNexusQ.csv'
    new_ds_df.to_csv(new_ds_fpath, index=False)
    
    return new_ds_df

def plot_sta_im(data_arr,ax,**kwargs):
    _ensure_axes()
    # Plot the data array as an image
    ax_im = ax.imshow(data_arr[eqt_inds[0]:eqt_inds[-2],t_inds[0]:t_inds[-1]], 
                      extent=[t_list[t_inds[0]], t_list[t_inds[-1]],
                              eqt_list[eqt_inds[0]], eqt_list[eqt_inds[-2]]],
                      interpolation='none',
                      cmap='coolwarm',
                      norm=colors.CenteredNorm(vcenter=0), 
                      **kwargs)
    
    # Plot the color bar
    plt.colorbar(ax_im,ax=ax)
    
    # Plot the vertical axis
    ax.axvline(0,color='white',linestyle=':')
    
    # Set the aspect ratio of the plot
    ax.set_aspect('auto')

def plot_sta_stair(data_arr,ax):
    _ensure_axes()
    mean_data = np.mean(data_arr[eqt_inds[0]:eqt_inds[-1],t_inds[0]:t_inds[-1]],0)

    ax_im = ax.stairs(mean_data, np.hstack((t_list[t_inds[:-1]], t_list[t_inds[-1]])),
                     fill=True, baseline=0)
    
    ax.axvline(0,color='black',linestyle=':')
    ax.set_aspect('auto')