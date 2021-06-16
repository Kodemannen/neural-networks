import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
import os

with open("pop.txt", "r") as popfile:
    data = np.loadtxt(popfile)

with open("rulevec.txt", "r") as f:
    rulevec = np.loadtxt(f, dtype=np.int32)

    # to string as well:
    rulevecstring = ""
    for elem in rulevec: 
        rulevecstring += str(elem)


N_GENERATIONS = data.shape[0]
POPSIZE = data.shape[1]


fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False,  sharey=False)

ax.imshow(data)
plt.savefig("testfig.pdf")
