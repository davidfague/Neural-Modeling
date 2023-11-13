import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_sim_file(h5file):
	with h5py.File(h5file, 'r') as file:
		data = np.array(file["data"])
	print(h5file)
	plt.plot(data)
	plt.savefig(f"{h5file}.png")

if __name__ == "__main__":

	if "--file" in sys.argv:
		sim_file = sys.argv[sys.argv.index("--file") + 1]
		plot_sim_file(sim_file)
