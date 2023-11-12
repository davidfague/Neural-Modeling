import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

def plotVm(h5file):
	with h5py.File(h5file, 'r') as file:
		data = np.array(file["report"]["biophysical"]["data"])
	
	print(data.shape)
	plt.plot(data)
	plt.savefig("vm.png")

if __name__ == "__main__":
	# Parse cl args
	if len(sys.argv) != 3:
		raise RuntimeError("usage: python file -f file")
	else:
		h5file = sys.argv[sys.argv.index("-f") + 1]

	plotVm(h5file)
