import numpy as np
import tensorflow as tf
import grab_cifar10

data = grab_cifar10.get_data()
#data = grab_mnist.get_data()

training_input = data["training_images"]
training_labels = data["training_labels"]
test_input = data["test_images"]
test_labels = data["test_labels"]

