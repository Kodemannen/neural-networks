##########################
# Fetching CIFAR 10 data #
##########################
import numpy as np
import glob, os

def unpickle(file):
    """
    python 3 version
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data():
    """
    Fetches training batch 1 and testing batch
    """

    data_dir = "/media/kodemannen/82684759-4f8f-4120-a5ed-a5055afc402d/CIFAR10/cifar-10-batches-py/"
    batch_name = "data_batch_1"
    test_name = "test_batch"

    training_data = unpickle(data_dir + batch_name)
    test_data = unpickle(data_dir + test_name)

    train_imgs = np.transpose(training_data[b"data"])
    train_labels = training_data[b"labels"]

    test_imgs = np.transpose(test_data[b"data"])
    test_labels = test_data[b"labels"]

    return {"training_images" : train_imgs, "training_labels" : train_labels, "test_images" : test_imgs, "test_labels" : test_labels}