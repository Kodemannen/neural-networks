##########################
# Fetching MNIST dataset #
##########################
import numpy as np
from mnist import MNIST

# def get_data():
#     """
#     Fetches training batch 1 and testing batch
#     """
    
#     data_dir = "/media/kodemannen/82684759-4f8f-4120-a5ed-a5055afc402d/MNIST dataset/"
#     file = open(data_dir + "train-labels-idx1-ubyte")
#     mndata = MNIST(data_dir)
#     train_imgs, train_labels = mndata.load_training()
#     test_imgs, test_labels = mndata.load_testing()

#     # filenames = 't10k-labels.idx1-ubyte', 'train-labels.idx1-ubyte', 'train-images.idx3-ubyte', 't10k-images.idx3-ubyte'

#     return {"training_images" : train_imgs, "training_labels" : train_labels, "test_images" : test_imgs, "test_labels" : test_labels}    

# #get_data()