#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=sim_run

source /home/shared/L5env/bin/activate
python sim_kmeans.py
