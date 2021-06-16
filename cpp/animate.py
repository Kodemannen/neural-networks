import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
import os


fps=10

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False,  sharey=False, frameon=False)

indices = np.arange(0, 2000)



# with open("matrix-files/states-H100-W120.txt", "r") as matfile:
#     data = np.loadtxt(matfile)
#     print(data.shape) 
#     exit("fitte")



def update_frame(i):

    print(i)

    ax.clear()
    img = plt.imread(f"images/test{i}.jpg")


    #ax.imshow(img, aspect="auto")
    ax.imshow(img)
    ax.axis('off')
    #fig.tight_layout()

    # ax.spines['right'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # # turn off ticks
    # ax.xaxis.set_ticks_position('none')
    # ax.yaxis.set_ticks_position('none')
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
            



#-------------------------------------------------
# Generating .mp4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist="Me"), bitrate=850)

ani = animation.FuncAnimation(fig, update_frame, indices)   #, fargs=(count,indices))
ani.save("game-of-life.mp4", writer=writer, dpi=200)



