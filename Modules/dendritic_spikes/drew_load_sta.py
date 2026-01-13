# this code was written by David Fague to load the stas that are saved in drew_analysis.py

import h5py
import numpy as np

def load_sta_h5(path):
    out = {}
    with h5py.File(path, "r") as f:
        for group_name in f.keys():
            g = f[group_name]
            out[group_name] = {k: g[k][...] for k in g.keys()}  # load arrays
    return out

def load_sta_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
