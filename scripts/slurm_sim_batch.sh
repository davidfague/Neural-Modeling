#!/bin/sh
  
#SBATCH --job-name=testing
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # cores per task; set to one if using MPI
#SBATCH --mem-per-cpu=5G # memory per core; default is 1GB/core

mpiexec python slurm_sim.py