# delete the recorded time courses only

import sys
import os
import shutil
import argparse

def delete_time_data_only(sim_dir):
    for file in os.listdir(sim_dir):
        if str(file).startswith("saved_at"):
            print(str(file))
            shutil.rmtree(os.path.join(sim_dir,file))


if __name__ == "__main__":
    listing = False
    if "-l" in sys.argv:  # listing multiple mode
        listing = True

    # Process a single directory (-d) or multiple directories (-f)
    if "-d" in sys.argv:
        d_index = sys.argv.index("-d")
        if listing:
            # In listing mode, consider all arguments after -d until the next flag.
            sim_directories = []
            for arg in sys.argv[d_index+1:]:
                if arg.startswith("-"):
                    break
                sim_directories.append(arg)
            for sim_directory in sim_directories:
                delete_time_data_only(sim_directory)
        else:
            sim_directory = sys.argv[d_index+1]
            delete_time_data_only(sim_directory)

    elif "-f" in sys.argv:
        f_index = sys.argv.index("-f")
        if listing:
            # In listing mode, consider all directories after -f until the next flag.
            simulations_directories = []
            for arg in sys.argv[f_index+1:]:
                if arg.startswith("-"):
                    break
                simulations_directories.append(arg)
            for simulations_directory in simulations_directories:
                print(f"simulations_directory: {simulations_directory}")
                # Process each simulation inside the provided directory.
                for sim_directory in os.listdir(simulations_directory):
                    full_path_sim = os.path.join(simulations_directory, sim_directory)
                    print(f"sim_directory: {sim_directory}")
                    delete_time_data_only(full_path_sim)
        else:
            simulations_directory = sys.argv[f_index+1]
            print(f"simulations_directory: {simulations_directory}")
            for sim_directory in os.listdir(simulations_directory):
                full_path_sim = os.path.join(simulations_directory, sim_directory)
                print(f"sim_directory: {sim_directory}")
                delete_time_data_only(full_path_sim)
    else:
        raise RuntimeError("Either '-d' or '-f' flag must be provided.")

#######################

# def delete_time_data_only(sim_dir):
#     for file in os.listdir(sim_dir):
#         if str(file).startswith("saved_at"):
#             print(str(file))
#             shutil.rmtree(file)

# if __name__ ==  "__main__":
#     # listing = False
#     # if "-l" in sys.argv:  # listing multiple mode
#     #     listing = True

#     if "-d" in sys.argv: # pass sim_dir
#         sim_directory = sys.argv[sys.argv.index("-d") + 1] # (global)
#         delete_time_data_only(sim_directory)
#     elif "-f" in sys.argv: # pass sims_dir and process each sims_dir/sim_dir
#         simulations_directory = sys.argv[sys.argv.index("-f") + 1]
#         print(f"simulations_directory: {simulations_directory}")
#         for sim_directory in os.listdir(simulations_directory):
#             full_path_sim = os.path.join(simulations_directory, sim_directory)
#             print(f"sim_directory: {sim_directory}")
#             delete_time_data_only(full_path_sim)
#     else:
#         raise RuntimeError
    
########################

# def delete_time_data_only(sim_directory):
#     """
#     Delete time data files from the specified simulation directory.
#     """
#     if not os.path.isdir(sim_directory):
#         print(f"Warning: '{sim_directory}' is not a valid directory. Skipping...")
#         return
    
#     # Example logic: Delete all files with a specific pattern
#     for file_name in os.listdir(sim_directory):
#         file_path = os.path.join(sim_directory, file_name)
#         if os.path.isdir(file_path) and "saved_at_step" in file_name:
#             try:
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             except OSError as e:
#                 print(f"Error deleting {file_path}: {e}")

# def process_simulation_directory(simulations_directory):
#     """
#     Process a directory containing multiple simulation folders.
#     """
#     if not os.path.isdir(simulations_directory):
#         raise RuntimeError(f"Error: '{simulations_directory}' is not a valid directory.")
    
#     for sim_directory in os.listdir(simulations_directory):
#         full_path_sim = os.path.join(simulations_directory, sim_directory)
        
#         # Skip non-directories
#         if not os.path.isdir(full_path_sim):
#             continue
        
#         print(f"Processing simulation directory: {sim_directory}")
#         delete_time_data_only(full_path_sim)

# def main():
#     """
#     Main function to parse arguments and execute the appropriate task.
#     """
#     parser = argparse.ArgumentParser(description="Delete time data files from simulation directories.")
#     parser.add_argument(
#         "-d", "--directory",
#         help="Path to a single simulation directory."
#     )
#     parser.add_argument(
#         "-f", "--folder",
#         help="Path to a directory containing multiple simulation folders."
#     )
    
#     args = parser.parse_args()

#     # Check which argument is provided
#     if args.directory:
#         delete_time_data_only(args.directory)
#     elif args.folder:
#         process_simulation_directory(args.folder)
#     else:
#         parser.print_help()
#         raise RuntimeError("No valid argument provided. Use -d or -f.")

# if __name__ == "__main__":
#     main()

##################################

# def delete_time_data_only(sim_directory):
#     """
#     Delete time data files from the specified simulation directory.
#     This function will check if the directory exists and then delete files
#     containing the substring 'time_data'.
#     """
#     if not os.path.isdir(sim_directory):
#         print(f"Warning: '{sim_directory}' is not a valid directory. Skipping...")
#         return

#     for file_name in os.listdir(sim_directory):
#         file_path = os.path.join(sim_directory, file_name)
#         if os.path.isfile(file_path) and "saved_at_step" in file_name:
#             try:
#                 # os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             except OSError as e:
#                 print(f"Error deleting {file_path}: {e}")

# def process_sims_directory(sims_directory, excluded_dirs):
#     """
#     Process a sims directory (i.e. a folder containing simulation directories).
#     Only directories that are not in the excluded list are processed.
#     """
#     if not os.path.isdir(sims_directory):
#         print(f"Error: '{sims_directory}' is not a valid directory. Skipping...")
#         return

#     for sim_directory in os.listdir(sims_directory):
#         full_path_sim = os.path.join(sims_directory, sim_directory)
#         # Skip non-directory items
#         if not os.path.isdir(full_path_sim):
#             continue
#         # Skip excluded simulation directories (by name)
#         if sim_directory in excluded_dirs:
#             print(f"Excluding simulation directory: {sim_directory}")
#             continue

#         print(f"Processing simulation directory: {full_path_sim}")
#         delete_time_data_only(full_path_sim)

# def main():
#     parser = argparse.ArgumentParser(
#         description="Delete time data files from simulation directories."
#     )

#     parser.add_argument(
#         "-d", "--sim",
#         help="Path to a single simulation directory."
#     )
#     parser.add_argument(
#         "-f", "--sims",
#         nargs="+",
#         help="Path(s) to directory(ies) containing simulation directories."
#     )
#     parser.add_argument(
#         "-e", "--exclude",
#         nargs="*",
#         default=[],
#         help="List of simulation directory names to exclude when processing sims directories."
#     )

#     args = parser.parse_args()

#     if not args.sim and not args.sims:
#         parser.print_help()
#         raise RuntimeError("No valid argument provided. Use -d (for a single simulation) or -f (for one or more sims directories).")

#     # Process a single simulation directory if provided.
#     if args.sim:
#         print(f"Processing single simulation directory: {args.sim}")
#         delete_time_data_only(args.sim)

#     # Process one or more sims directories if provided.
#     if args.sims:
#         for sims_directory in args.sims:
#             print(f"\nProcessing simulations directory: {sims_directory}")
#             process_sims_directory(sims_directory, args.exclude)

# if __name__ == "__main__":
#     main()
