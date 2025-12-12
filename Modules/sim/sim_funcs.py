"""
Modules/sim/sim_funcs.py

Simulation execution functions extracted from scripts/AA_sim_then_post_parallel.py.
These functions handle the core NEURON simulation execution.
"""

import os
from Modules.pre_sim.synapses_file import PreSimSynapseGenerator
from Modules.sim.simulation_slurm import Simulation
from Modules.cell_model.cell_builder import SkeletonCell
from Modules.logger import Logger
import Modules.analysis as analysis


def run_single_sim(sim_dir: str, logger=None):
    """
    Execute a single NEURON simulation.
    
    Extracted from run_one() in AA_sim_then_post_parallel.py.
    
    This function:
    1. Loads parameters from sim_dir
    2. Builds the cell with synapses using PreSimSynapseGenerator
    3. Runs the NEURON simulation via Simulation.run_single_simulation()
    
    Args:
        sim_dir: Path to simulation directory containing parameters.pickle and synapses.csv
        logger: Optional Logger instance. If None, creates a new logger for sim_dir.
                The logger is used for timing the cell building and simulation steps.
    """
    # Create logger if not provided
    if logger is None:
        logger = Logger(sim_dir)
    
    # Load params & build cell
    parameters = analysis.DataReader.load_parameters(sim_dir)
    pssg = PreSimSynapseGenerator(sim_dir)
    cell = pssg.build_synapses_onto_cell_obj()

    # Run simulation
    sim = Simulation(getattr(SkeletonCell, parameters.skeleton_cell_type), create_dir=False)
    sim.path = os.path.split(sim_dir)[0]   # parent sims_dir
    sim.logger = logger
    sim.run_single_simulation(parameters=parameters, cell=cell)
