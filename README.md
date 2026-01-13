# Neural-Modeling

# Instructions:

Start with Neural-Modleing/setup/README.md

To run a simulation:
    Use configure_sim_params.py to make changes to define and change the simulation(s). Default parameters are in constants.py.

    Use AA_run_pipeline.py to prepare, run, and analyze simulations.
    
    Check simulations/{your_sims_dir}/{your_sim_dir} for results.

# When developing a new feature, create an example:

In configure_sim_params.py, use the 'testing' sim_type and name your simulation according to what you are developing.

In "examples/", create a ".ipynb" notebook named according to what you are developing that references the simulation. 

Move the simulation into "examples/examples_simulations/"

To be disk efficient while adding example simulations to GitHub, use "python scripts/clean_up_data/delete_saved_time_data_for_sim.py {SIM_DIR} --subdir raw_data --keep-files v.h5" to delete the raw_data and keep only voltage data or "python scripts/clean_up_data/delete_saved_time_data_for_sim.py {SIM_DIR} --subdir raw_data" to remove all raw data.

# Additional

Publication: https://mailmissouri-my.sharepoint.com/:w:/r/personal/nairs_umsystem_edu/Documents/MigratedBoxFiles/nairs/AAWork%20in%20progress/PAPERS%20in%20progress/AA-SingleCell%20and%20WM%20Projects/Reduced%20Order%20Modeling%20Project/Manuscript%20-%20EquivalentModel/2025%20IEEE%20MWSCAS/2025_ieee-DetailedCell.docx?d=w14e2c2d2b3e8490d82f28972448e9762&csf=1&web=1&e=nfMC3R

Start with Neural-Modeling/setup/README.md to get started with installing miniconda, creating an environment, and installing necessary packages.

notebook/pre_sim_dev/AA_pre_sim.ipynb is the core example for designing simulations. A folder will be created for the set of designed simulations with folders inside for each simulation in the path Neural-Modeling/simulations/{simulations_set}/{simulation}. @davidfague: replace with actual example instead of dev notebook

notebooks/reduction/AA_reduce_adaptation_dev_clean.ipynb is an example of designing reduced cell simulations. @jack, update with yours.

notebooks/write_synapses_pipeline_dev/AA_sim.ipynb will simulate the specified simulations. @davidfague replace with actual example instead of dev notebook

notebooks/write_synapses_pipeline_dev/AA_post_sim.ipynb is used to analyze simulated data. @davidfague replace with actual example instead of dev notebook

Guidelines for tuning synaptic design for realistic dendritic spikes: https://mailmissouri-my.sharepoint.com/:w:/r/personal/nairs_umsystem_edu/Documents/MigratedBoxFiles/nairs/AAWork%20in%20progress/PAPERS%20in%20progress/AA-SingleCell%20and%20WM%20Projects/Reduced%20Order%20Modeling%20Project/Manuscript%20-%20EquivalentModel/2025%20IEEE%20MWSCAS/Tuning-DetailedCellModel.docx?d=wf09133c47cf045f68b43e41dae8c6776&csf=1&web=1&e=wDmRo4

How to download and run an AllenDB Cell: https://mailmissouri-my.sharepoint.com/:w:/r/personal/drfrbc_umsystem_edu/Documents/How%20To%20-%20Allen%20Cell%20in%20Single-Cell%20Pipeline.docx?d=w88c46ac81e2548648dc84744a9ba5c11&csf=1&web=1&e=E4Hjef