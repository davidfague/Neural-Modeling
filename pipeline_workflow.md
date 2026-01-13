# Pipeline Workflow

## Overview

```
configure_sim_params.py → scripts/run_pipeline.py
```

---

## 1. Configuration: `configure_sim_params.py`

Configure the parameters of the simulation(s) you want to run.

---

## 2. Main Pipeline: `scripts/run_pipeline.py`

Run the entire pipeline: building, running, and analyzing the simulations.

### 2.1 Pre-Simulation: `scripts/pre_sim.py`

Prepare simulations: generate `synapses.csv`, `segments.csv`

**Supporting Module:**
- `Modules/pre_sim/pre_sim_funcs.py` - Functions called in `pre_sim.py`

### 2.2 Simulation: `scripts/sim.py`

Run the simulation.

**Supporting Module:**
- `Modules/sim/sim_funcs.py` - Contains `run_single_sim()`

### 2.3 Post-Simulation Analysis: `scripts/post_sim_analysis.py`

Analyze the simulation.

**Supporting Modules:**
- `Modules/post_sim/post_sim_funcs.py` - Functions for analyzing the simulation
- `Modules/dendritic_spikes/find_events_ben.py` - Event detection
- `Modules/dendritic_spikes/event_histograms.py` - Event histogram generation
- `Modules/dendritic_spikes/drew_analysis.py` - Custom analysis routines
- `Modules/clustering/analyze_clustering.py` - Clustering analysis
- `Modules/spike_synchrony/run_spike_synchrony.py` - Spike synchrony analysis
