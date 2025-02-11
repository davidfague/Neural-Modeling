#!/bin/sh
  
#SBATCH --job-name=testing
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=1 # cores per task; set to one if using MPI
#SBATCH --mem-per-cpu=5G # memory per core; default is 1GB/core

# Set the simulation title here
SIM_TITLE="testing_new_weights"

# Pass SIM_TITLE as a command-line argument to the Python script
mpiexec python slurm_sim.py "$SIM_TITLE"