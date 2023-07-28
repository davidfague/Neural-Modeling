#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=l5cell

source /home/shared/L5env/bin/activate
python sim_func_groups.py
