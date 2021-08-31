import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import struct
import idx2numpy
import matplotlib.pyplot as plt

data_path = "../../data/mnist"

#-----------------------------------------------------------------------------
# Reading binary idx-file and convert to Numpy array                         :
#-----------------------------------------------------------------------------
# The files are in idx format
# 10000 test-data, 60000 training-data
test_labels = idx2numpy.convert_from_file(os.path.join(data_path, 
                                        't10k-labels-idx1-ubyte'))

test_images = idx2numpy.convert_from_file(os.path.join(data_path, 
                                        't10k-images-idx3-ubyte'))


train_labels = idx2numpy.convert_from_file(os.path.join(data_path, 
                                        'train-labels-idx1-ubyte'))

train_images = idx2numpy.convert_from_file(os.path.join(data_path, 
                                        'train-images-idx3-ubyte'))

n_classes = len(np.unique(train_labels, return_index=False))


#-----------------------------------------------------------------------------
# Make onehot encoded labels                                                 :
#-----------------------------------------------------------------------------
test_labels_onehot = np.zeros(shape=(test_labels.shape[0], n_classes))
test_labels_onehot[np.arange(test_labels.shape[0]),test_labels]=1

train_labels_onehot = np.zeros(shape=(train_labels.shape[0], n_classes))
train_labels_onehot[np.arange(train_labels.shape[0]),train_labels]=1



img_example = train_images[0]   # shape = (28, 28)



'''
Keras docs:

Functional API:
https://keras.io/guides/functional_api/

Layer types:
https://keras.io/api/layers/

Losses:
https://keras.io/api/losses/
'''
#-----------------------------------------------------------------------------
# Data preparation                                                           :
#-----------------------------------------------------------------------------
test_images = test_images/255
train_images = train_images/255

#-----------------------------------------------------------------------------
# Construct neural network graph                                             :
#-----------------------------------------------------------------------------
inputs = keras.Input(shape=(28, 28, 1), name="images")

# x = layers.Flatten()(inputs)

x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
#x = layers.Dropout(0.5)(x)
# x = layers.Dense(units=128, activation='relu')(x)
# x = layers.BatchNormalization()(x)

outputs = layers.Dense(units=n_classes, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model") 
model.summary()


#-----------------------------------------------------------------------------
# Training                                                                   : 
#-----------------------------------------------------------------------------
loss = keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(
    loss=loss,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

history = model.fit(train_images, 
                    train_labels_onehot, 
                    batch_size=64,
                    epochs=10,
                    validation_split=0.2)
