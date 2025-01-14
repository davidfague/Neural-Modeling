import sys
sys.path.append("../")
sys.path.append("../Modules/")

from Modules.constants import HayParameters
import os, pickle

N_workers = 2

if __name__ == "__main__":

    # Populate parameters here
    parameters = []
    parameters.append(HayParameters("0", h_tstop = 2000))
    parameters.append(HayParameters("1", h_tstop = 2000))

    # Self-check
    assert len(parameters) == N_workers

    # Generate parameter pickles
    os.mkdir("params")
    for pid in range(len(parameters)):
        with open(os.path.join("params", f"{pid}.pickle"), 'wb') as file:
            pickle.dump(parameters[pid], file)
