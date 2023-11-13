import h5py, pickle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append("../")
sys.path.append("../Modules/")

def plot_sim_file(h5file):
	with h5py.File(h5file, 'r') as file:
		data = np.array(file["data"])
	plt.plot(data)
	plt.savefig(f"quickplot.png")

def plot_sim_folder(sim_folder, sim_file_name):
	with open(os.path.join(sim_folder, "parameters.pickle"), "rb") as file:
			parameters = pickle.load(file)

	step_size = int(parameters.save_every_ms / parameters.h_dt) # Timestamps
	steps = range(step_size, int(parameters.h_tstop / parameters.h_dt) + 1, step_size) # Timestamps

	data_to_plot = []
	for step in steps:
		with h5py.File(os.path.join(sim_folder, f"saved_at_step_{step}", sim_file_name + ".h5"), 'r') as file:
			data_to_plot.append(np.array(file["data"]))
	data_to_plot = np.concatenate(data_to_plot)
	plt.plot(data_to_plot)
	plt.savefig(f"quickplot.png")


if __name__ == "__main__":

	if "-f" in sys.argv:
		sim_file = sys.argv[sys.argv.index("-f") + 1]
		plot_sim_file(sim_file)
	
	if ("-d" in sys.argv) and ("-v" in sys.argv):
		sim_folder = sys.argv[sys.argv.index("-d") + 1]
		sim_file_name = sys.argv[sys.argv.index("-v") + 1]
		plot_sim_folder(sim_folder, sim_file_name)

		
			
