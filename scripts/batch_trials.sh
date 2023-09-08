#!/bin/bash

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=cell_sim

source /home/shared/L5env/bin/activate

python generate_combinations.py | while read -r combination; do
    echo "Updating constants to $combination"
    python update_constants.py "$combination"
    echo "Completed update for $combination"

done