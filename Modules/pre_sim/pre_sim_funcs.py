"""
Modules/pre_sim/pre_sim_funcs.py

Pre-simulation processing functions extracted from scripts/AA_pre_sim.py.
These functions perform computational tasks for the pre-simulation pipeline stage.
"""

import os
import pandas as pd
import datetime
import Modules.post_sim.analysis as analysis
from Modules.pre_sim.synapses_file import PreSimSynapseGenerator
from Modules.cell_model.plot_morphology import plot_morphology_flex
from Modules.logger import Logger


def _now():
    """Get current timestamp string."""
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{_now()}] {msg}", flush=True)


def generate_synapses_for_sim(sim_dir):
    """
    Generate synapse locations and spike trains for a single simulation.
    
    Renamed from use_pssg() in AA_pre_sim.py.
    
    Args:
        sim_dir: Path to simulation directory
    """
    pssg = PreSimSynapseGenerator(sim_dir)
    pssg.generate_synapse_locations()
    pssg.synapses.to_csv(os.path.join(sim_dir, "synapses.csv"), index=False)
    pssg.generate_spike_trains_for_synapses()


def plot_morphology_for_sim(sim_dir):
    """
    Plot morphology for a single simulation.
    
    Renamed from plot_morphology() in AA_pre_sim.py.
    
    Args:
        sim_dir: Path to simulation directory
    """
    seg_data = pd.read_csv(os.path.join(sim_dir, "segment_data.csv"))
    parameters = analysis.DataReader.load_parameters(sim_dir)
    logger = Logger(sim_dir)
    figs, axs = plot_morphology_flex(
        seg_data,
        option='each_sec_type',
        parameters=parameters,
        out_dir=os.path.join(sim_dir, "morphology"),
        figsize=(10, 6),
        show=True,
        save=True,
        color='red',
        logger=logger,
    )


def plot_firing_rate_distributions_for_sim(simulator, synapse_analyzers, logger):
    """
    Generate firing rate distribution plots for all simulations.
    
    Renamed from plot_firing_rate_distributions() in AA_pre_sim.py.
    
    Args:
        simulator: Simulator object with sims_dir attribute
        synapse_analyzers: List of SynapseAnalyzer objects
        logger: Logger instance for timing
    """
    log(f"\n[AA_pre_sim] Generating firing rate distribution plots on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("firing_rate_distribution_plots")
    
    for synapse_analyzer in synapse_analyzers:
        distributions_folder = os.path.join(synapse_analyzer.sim_dir, "firing_rate_distributions")
        os.makedirs(distributions_folder, exist_ok=True)
        
        # Overall plots by synapse type
        synapse_analyzer.plot_firing_rate_distribution(
            synapse_type='exc',
            save_path=os.path.join(distributions_folder, 'exc_firing_rate_dist.png'),
            show=False
        )
        
        synapse_analyzer.plot_firing_rate_distribution(
            synapse_type='inh',
            save_path=os.path.join(distributions_folder, 'inh_firing_rate_dist.png'),
            show=False
        )
        
        synapse_analyzer.plot_firing_rate_distribution(
            save_path=os.path.join(distributions_folder, 'all_firing_rate_dist.png'),
            show=False
        )
        
        # Per input_source plots
        for input_source in synapse_analyzer.synapses['input_source'].unique():
            synapse_type_prefix = 'exc' if synapse_analyzer.synapses[
                synapse_analyzer.synapses['input_source'] == input_source
            ]['name'].iloc[0].startswith('exc') else 'inh'
            synapse_analyzer.plot_firing_rate_distribution(
                input_source=input_source,
                save_path=os.path.join(
                    distributions_folder, 
                    f'{synapse_type_prefix}_{input_source}_firing_rate_dist.png'
                ),
                show=False
            )
    
    logger.log_runtime("AA_pre_sim", "firing_rate_distribution_plots", timer_name="firing_rate_distribution_plots")


def plot_spike_rasters_for_sim(simulator, synapse_analyzers, logger):
    """
    Generate spike raster plots for all simulations.
    
    Renamed from plot_spike_rasters() in AA_pre_sim.py.
    
    Args:
        simulator: Simulator object with sims_dir attribute
        synapse_analyzers: List of SynapseAnalyzer objects
        logger: Logger instance for timing
    """
    log(f"\n[AA_pre_sim] Generating spike raster plots on all sims in: {simulator.sims_dir}\n")
    logger.start_timer("spike_raster_plots")
    
    for synapse_analyzer in synapse_analyzers:
        parameters = analysis.DataReader.load_parameters(synapse_analyzer.sim_dir)
        spike_rasters_folder = os.path.join(synapse_analyzer.sim_dir, "spike_rasters")
        os.makedirs(spike_rasters_folder, exist_ok=True)
        
        # Excitatory - end of simulation
        synapse_analyzer.plot_spike_raster(
            synapse_types=['exc'],
            time_window=(parameters.h_tstop - 1000, parameters.h_tstop),
            save_path=os.path.join(spike_rasters_folder, 'exc_spike_raster_end.png'),
            title="Excitatory Synapse Spike Raster"
        )
        
        # Inhibitory - end of simulation
        synapse_analyzer.plot_spike_raster(
            synapse_types=['inh'],
            time_window=(parameters.h_tstop - 1000, parameters.h_tstop),
            save_path=os.path.join(spike_rasters_folder, 'inh_spike_raster_end.png'),
            title="Inhibitory Synapse Spike Raster"
        )
        
        # Excitatory - start of simulation
        synapse_analyzer.plot_spike_raster(
            synapse_types=['exc'],
            time_window=(0, 1000),
            save_path=os.path.join(spike_rasters_folder, 'exc_spike_raster_start.png'),
            title="Excitatory Synapse Spike Raster"
        )
        
        # Inhibitory - start of simulation
        synapse_analyzer.plot_spike_raster(
            synapse_types=['inh'],
            time_window=(0, 1000),
            save_path=os.path.join(spike_rasters_folder, 'inh_spike_raster_start.png'),
            title="Inhibitory Synapse Spike Raster"
        )
    
    logger.log_runtime("AA_pre_sim", "spike_raster_plots", timer_name="spike_raster_plots")
