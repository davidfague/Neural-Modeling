#!/bin/sh
  
#SBATCH --job-name=my_awesome_sim_with_lots_of_cells_AAAA
#SBATCH -N 1
#SBATCH -n 2

mpiexec python slurm_sim.py