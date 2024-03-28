import sys
sys.path.append("../Stylized-Cell-Inference")
sys.path.append("../Stylized-Cell-Inference/cell_inference")
sys.path.append("../Stylized-Cell-Inference/cell_inference/cells")

from neuron import h
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Optional, List, Tuple, Union

from cell_inference.config import params, paths
from cell_inference.cells.simulation import Simulation, Simulation_stochastic
from cell_inference.cells.stylizedcell import CellTypes
from cell_inference.utils.currents.recorder import Recorder
from cell_inference.utils.currents.pointconductance import PointConductance
from cell_inference.utils.metrics.measure_passive_properties import measure_passive_properties
from cell_inference.utils.metrics.measure_segment_distance import measure_segment_distance
from cell_inference.utils.feature_extractors.SummaryStats2D import process_lfp
import cell_inference.utils.feature_extractors.SummaryStats2D as ss2
from cell_inference.utils.plotting.plot_morphology import plot_morphology
from cell_inference.utils.plotting.plot_variable_with_morphology import plot_variable_with_morphology
from cell_inference.utils.plotting.plot_results import plot_lfp_heatmap, plot_lfp_traces
from cell_inference.utils.transform.data_transform import log_modulus

def build_ziaos_cell():
    h.load_file('stdrun.hoc')
    # source_directory = os.path.join('cell_inference', 'resources', 'compiled', 'mechanisms_reduced_order')
    # os.chdir(source_directory)
    # os.system(f"nrnivmodl > /dev/null 2>&1")
    # os.chdir("../../../../")
    h.nrn_load_dll(paths.COMPILED_LIBRARY_REDUCED_ORDER)
    geo_standard = pd.read_csv(paths.GEO_REDUCED_ORDER, index_col='id')
    params.DT = 0.1 # comment for default
    h.dt = params.DT
    h.steps_per_ms = 1/h.dt

    h.tstop = 2000.

    # Biophysical parameters
    filepath = '/Users/vladimiromelyusik/Neural-Modeling/Stylized-Cell-Inference/cell_inference/resources/biophys_parameters/ReducedOrderL5_stochastic.json' # active dendrites
    # filepath = './cell_inference/resources/biophys_parameters/ReducedOrderL5.json' # higher gNa dendrites
    with open(filepath) as f:
        biophys_param = json.load(f)

    biophys = [2.04, 0.0213 * 0.6, 0.0213 * 0.6, 0.693 * 2, 0.000261 * 2, 100., 100., 0.0000525, 0.000555, 0.0187, # 2.04, 0.0639, 0.693 (1.0), 0.000261
          np.nan, np.nan, np.nan, np.nan, .6, 2.4]
    # biophys = []
    biophys_comm = {}

    # interpret_params = False  # not using parameter interpreter
    # geo_param = [135, 652, 163, 1.77, 1.26, .99] # 5 - 595 um
    interpret_params = True  # using parameter interpreter
    # geo_param = [1000, 1.]  # total length, radius scale
    geo_param = [950., 0.142, 1., 0.59]  # total length 100-1200, prox prop 0.02-0.35, radius scale 0.4-1.5 (0.6-1.2), dist/prox radius 0.4-0.8 [950., 0.142, 1., 0.59]

    loc_param = [0., 0., 0., 0., 1., 0.] # position (x,y,z,alpha,h,phi)

    ncell = 0
    biophys_rep = np.tile(biophys,(ncell, 1))
    # biophys_rep[:,5] = np.linspace(200, 300, ncell)
    biophys = np.vstack((biophys_rep, biophys))

    geo_param_rep = np.tile(geo_param,(ncell,1))
    # geo_param_rep[:,6] = np.linspace(.1, .2, ncell)
    # geo_param_rep[:,[2,3,4,6]] *= 1.0 # scale radius
    geo_param = np.vstack((geo_param_rep, geo_param))

    Len = {'p': 100 + 20, 'b': 100, 'a': 250}

    attr_kwargs = {}

    point_conductance_division = {'soma': [0], 'perisomatic': [1,4], 'basal': [2,3], 'apical': [7,8,9,10]}
    dens_params = { # ALL SET TO 0 TO REMOVE look on github for original values
        'soma': {'g_e0': 0., 'g_i0': 0., 'std_e': 0., 'std_i': 0.},
        'perisomatic': {'g_e0': 0., 'g_i0': 0., 'std_e': 0., 'std_i': 0.},
        'basal': {'g_e0': 0., 'g_i0': 0., 'std_e': 0., 'std_i': 0.},
        'apical': {'g_e0': 0., 'g_i0': 0., 'std_e': 0., 'std_i': 0.}
    }
    cnst_params = {'tau_e': 2., 'tau_i': 10., 'tau_n': 40.}
    has_nmda = False
    lornomal_gfluct = False #True

    randseed = 0

    sim = Simulation_stochastic(
    cell_type = CellTypes.REDUCED_ORDER,
    ncell = ncell + 1,
    geometry = geo_standard,
    electrodes = params.ELECTRODE_POSITION,
    loc_param = loc_param,
    geo_param = geo_param,
    biophys = biophys,
    full_biophys = biophys_param,
    biophys_comm = biophys_comm,
    interpret_params = interpret_params,
    interpret_type = 3,
    min_distance = params.MIN_DISTANCE,
    spike_threshold = params.SPIKE_THRESHOLD,
    cell_kwargs = {'attr_kwargs': attr_kwargs},
    point_conductance_division=point_conductance_division,
    dens_params=dens_params,
    cnst_params=cnst_params,
    has_nmda=has_nmda,
    lornomal_gfluct=lornomal_gfluct,
    tstart=200.,
    randseed = randseed
    )

    cell = sim.cells[-1]

    cell.dend = [i for i in cell.all if "basal" in str(i)]
    cell.apic = [i for i in cell.all if ("basal" not in str(i)) and ("soma" not in str(i))]
    cell.soma = [i for i in cell.all if "soma" in str(i)]
    cell.axon = []

    return cell