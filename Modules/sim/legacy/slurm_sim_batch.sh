#!/bin/sh
  
#SBATCH --job-name=SingleCellStudy
#SBATCH -N 1
#SBATCH -n 1 # change this based on N_workers
#SBATCH --ntasks-per-node=1 # change this based on N_workers
#SBATCH --cpus-per-task=1 # cores per task; set to one if using MPI
#SBATCH --mem-per-cpu=5G # memory per core; default is 1GB/core
#SBATCH --time=12:00:00  # Set a reasonable max runtime

# Set the simulation title here
SIM_TITLE="L5_Testing_Seeding_inc_tuft_inh_dens"

# Pass SIM_TITLE as a command-line argument to the Python script
mpiexec python slurm_sim.py "$SIM_TITLE"

# seff job_id > "$SIM_TITLE/sbatch_efficiency.txt"