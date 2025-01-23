#!/bin/sh
#SBATCH -J  single_cell
#SBATCH -o  single_cell.out
#SBATCH -e  single_cell.error
#SBATCH -t 0-48:00:00  # days-hours:minutes

#SBATCH -N 1
#SBATCH -n 1 # used for MPI codes, otherwise leave at '1'
#SBATCH --ntasks-per-node=1  # don't trust SLURM to divide the cores evenly
#SBATCH --cpus-per-task=1  # cores per task; set to one if using MPI
##SBATCH --exclusive  # using MPI with 90+% of the cores you should go exclusive
#SBATCH --mem-per-cpu=5G  # memory per core; default is 1GB/core


START=$(date)
echo "Started running at $START."

unset DISPLAY
python sim_new3_tuning_synapses.py 

END=$(date)
echo "Done running at $END"