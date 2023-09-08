#!/bin/sh

#SBATCH --qos=normal
#SBATCH --mem-per-cpu=128GB
#SBATCH --job-name=exam_sim

source /home/shared/L5env/bin/activate

# Define multiple output folders in an array
OUTPUT_FOLDERS=("output/FI_in_vitro2023-09-07_19-16-51" "output/FI_in_vitro2023-09-07_19-18-24" "output/FI_in_vitro2023-09-07_19-19-57" "output/FI_in_vitro2023-09-07_19-21-30" "output/FI_in_vitro2023-09-07_19-23-03" "output/FI_in_vitro2023-09-07_19-24-36" "output/FI_in_vitro2023-09-07_19-24-41" "output/FI_in_vitro2023-09-07_19-24-45" "output/FI_in_vitro2023-09-07_19-24-50" "output/FI_in_vitro2023-09-07_19-24-55" "output/FI_in_vitro2023-09-07_19-26-30")

# List of all exam_something.py scripts to run
scripts=("exam_fi.py")

# Loop over each output folder
for OUTPUT_FOLDER_PATH in "${OUTPUT_FOLDERS[@]}"
do
    # Loop over each script for the current output folder
    for script in "${scripts[@]}"
    do
        python $script $OUTPUT_FOLDER_PATH
    done
done
