#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=256GB
#SBATCH --job-name=l5cell

source /home/shared/L5env/bin/activate
python sim_kmeans_PSCs_new.py
