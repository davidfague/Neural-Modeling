#!/bin/sh
  
#SBATCH --job-name=my_awesome_sim_with_lots_of_cells_AAAA
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

mpiexec python slurm_sim.py
