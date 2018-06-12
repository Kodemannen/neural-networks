import numpy as np

cifar10 = np.load("/media/kodemannen/82684759-4f8f-4120-a5ed-a5055afc402d/cifar10_numpy_array.npy")

training_data = cifar10[0]      # 5 batches, each column in a batch is an image
training_labels = cifar10[1]    # 5 batches
test_data = cifar10[2]          # 1 batch, each column is an image
test_labels = cifar10[3]        # 1 batch


print(training_data.shape)
new = training_data.reshape(5, 3,32,32, 10000)

import matplotlib.pyplot as plt
plt.imshow(new[0,0,:,:,0])
plt.show()

##################################
# Defining network architecture: #
##################################

kernelSizes = [3,3,3]
channels = [3,16,16]
strides = [1,1,1]

