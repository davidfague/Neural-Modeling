#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=l5cell_analysis

source /home/shared/L5env/bin/activate

# Define the output folder value
OUTPUT_FOLDER_PATH='output/2023-08-18_22-14-47_seeds_125_87L5PCtemplate[0]_196nseg_108nbranch_15814NCs_15814nsyn'

# List of all exam_something.py scripts to run
scripts=("exam_func_group.py" "exam_nmda.py" "exam_axial_currents.py" "exam_clusters.py")

for script in "${scripts[@]}"
do
    python $script $OUTPUT_FOLDER_PATH
done
