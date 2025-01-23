#!/bin/python3
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

import os
from multiprocessing import Pool

def hello(core_id):
    print(f"Hello World from core {core_id}")

if __name__ == "__main__":
    # Get the number of cores allocated by SLURM
    number_of_cores = int(os.environ.get('SLURM_CPUS_ON_NODE', 1)) # if not there lets default to 1

    # Create a pool of workers
    with Pool(number_of_cores) as pool:
        # Distribute the 'hello' task across the pool
        core_ids = range(number_of_cores)
        pool.map(hello, core_ids)