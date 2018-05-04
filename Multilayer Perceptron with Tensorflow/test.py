import numpy as np
ting = np.zeros((4,3))
print(ting)
indices = [0,2,1]
ting[indices,range(3)] = 1
print(ting)