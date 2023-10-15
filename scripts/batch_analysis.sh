#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=exam_sim

# source /home/shared/L5env/bin/activate

# Define multiple output folders in an array
OUTPUT_FOLDERS=(
'output/1L5PCtemplate[0]_150000_195nseg_108nbranch_16197NCs_16197nsyn'
)

# List of all exam_something.py scripts to run
scripts=("exam_axial_currents.py" "exam_nmda.py") # "exam_func_group.py" "exam_nmda.py"

# Loop over each output folder
for OUTPUT_FOLDER_PATH in "${OUTPUT_FOLDERS[@]}"
do
    # Loop over each script for the current output folder
    for script in "${scripts[@]}"
    do
        python $script $OUTPUT_FOLDER_PATH
    done
done
