from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sys import exit
import matplotlib.image as mpimg
mnist_data = input_data.read_data_sets("Mnist/", one_hot=True)

# batch_size = 16

image1, label_batch = mnist_data.train.next_batch(1)
input_dim = image1.shape[1]

image1 = image1.reshape(28,28)
noise = np.random.randn(*image1.shape)*0.2

####################
# Hyperparameters: #
####################
network_structure = [ input_dim, 128, 64, 10, 64, 128, input_dim]   # the encoder and decoder combined
coding_layer_index = np.argmin(network_structure)
nodes_in_coding_layer = network_structure[coding_layer_index]
learning_rate = 0.01
training_cycles = 10000
batch_size = 128
test_batch_size = 10
beta = 0.001


#########################
# Setting up the graph: #
#########################
data = tf.placeholder( dtype=tf.float32, shape=(None, input_dim))
a = data
for l in range(1,len(network_structure)):
    if l != coding_layer_index:
        w = tf.get_variable(name="w%s" % l, initializer=tf.random_normal(shape=(network_structure[l-1], network_structure[l]), stddev=1./np.sqrt(network_structure[l-1])))
        b = tf.get_variable(name="b%s" % l, shape=(1, network_structure[l]), initializer=tf.zeros_initializer)

        z = tf.matmul(a, w) + b
        a = tf.nn.relu(z, name="a%s" % l)    # last activation has name "a%s" % (len(network_structure)-1)
    
    else:
        w_mu = tf.get_variable(name="w_mu%s" % l, initializer=tf.random_normal(shape=(network_structure[l-1], network_structure[l]), stddev=1./np.sqrt(network_structure[l-1])))
        b_mu = tf.get_variable(name="b_mu%s" % l, shape=(1, network_structure[l]), initializer=tf.zeros_initializer)
        
        w_sigma = tf.get_variable(name="w_sigma%s" % l, initializer=tf.random_normal(shape=(network_structure[l-1], network_structure[l]), stddev=1./np.sqrt(network_structure[l-1])))
        b_sigma = tf.get_variable(name="b_sigma%s" % l, shape=(1, network_structure[l]), initializer=tf.zeros_initializer)
        
        mu = tf.matmul(a, w_mu) + b_mu
        sigma_squared = tf.matmul(a, w_sigma) + b_sigma

        z_tilde = tf.random_normal(shape=(tf.shape(a)[0], network_structure[coding_layer_index]))
        a = mu + sigma_squared*z_tilde
        



################################
# Defining the loss operation: #
################################
#KL_loss = 0.5*tf.reduce_mean(mean*mean + variance - tf.log(variance) - 1)
KL_loss = beta*0.5*tf.reduce_mean(mu*mu + sigma_squared - tf.log(sigma_squared) -1)
loss_reconstruction = tf.losses.mean_squared_error(labels=data, predictions=a)

loss = loss_reconstruction  #- KL_loss

printer = tf.Print(KL_loss, [loss])

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operator = optimizer.minimize(loss=loss, var_list=tf.trainable_variables())

initializer = tf.global_variables_initializer()

##################
# Running graph: #
##################
with tf.Session() as sess:
    sess.run(initializer)

    #############
    # Training: #
    #############
    for cycle in range(training_cycles):
        
        image_batch, label_batch = mnist_data.train.next_batch(batch_size)
        #noise_batch = np.random.randn(*image_batch.shape)*0.2 + image_batch

        loss_val, _ = sess.run([loss, training_operator], feed_dict={data: image_batch} )


    ############
    # Testing: #
    ############
    test_batch = np.random.randn(test_batch_size, nodes_in_coding_layer)
    a = test_batch
    for l in range(coding_layer_index+1, len(network_structure)):
        w, b = sess.run(["w%s:0"%l, "b%s:0"%l])
        z = np.dot(a, w) + b
        a = z*(z>0)     # relu

    created_images = a

    for img_index in range(test_batch_size):
        
        img = created_images[img_index].reshape(28,28)

        plt.imshow(img, cmap="gray")
        plt.savefig("%s.jpg" % img_index)

    ##################
    # INTERPOLATION: #
    ##################

    # image1, label_batch = mnist_data.train.next_batch(1)
    # created_image1 = sess.run(a, feed_dict={data: image_batch} )

    # image2, label_batch = mnist_data.train.next_batch(1)
    # created_image1 = sess.run(a, feed_dict={data: image_batch} )

    # plt.imshow(image_batch[0].reshape(28,28))
    # plt.show()
    # plt.imshow(image_batch[1].reshape(28,28))
    # #plt.imshow