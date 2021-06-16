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

# Hyperparameters:
network_structure = [ input_dim, 128, 32, 128, input_dim]   # the encoder and decoder combined
coding_layer_index = np.argmin(network_structure)
nodes_in_coding_layer = network_structure[coding_layer_index]
learning_rate = 0.01
training_cycles = 100
batch_size = 128
test_batch_size = 10
sparse = False
p = 0.1         # sparsity parameter
beta = 0.1      # sparsity regularization strength


# setting up the graph:
data = tf.placeholder( dtype=tf.float32, shape=(None, input_dim))
noiseless = tf.placeholder( dtype=tf.float32, shape=(None, input_dim))          # JUST FOR TRAINING

# def get_noise(data):
#     noise = np.random.randn(*data.shape)*0.2
#     return noise

a = data
for l in range(1,len(network_structure)):
    w = tf.get_variable(name="w%s" % l, initializer=tf.random_normal(shape=(network_structure[l-1], network_structure[l]), stddev=1./np.sqrt(network_structure[l-1])))
    b = tf.get_variable(name="b%s" % l, shape=(1, network_structure[l]), initializer=tf.zeros_initializer)

    z = tf.matmul(a, w) + b
    a = tf.nn.sigmoid(z, name="a%s" % l)    # last activation has name "a%s" % (len(network_structure)-1)
    # if l == coding_layer_index:
    #     coding_layer_activations = a
 
################################
# Defining the loss operation: #
################################
loss_reconstruction = tf.losses.mean_squared_error(labels=noiseless, predictions=a)

loss_sparsity = 0
if sparse:
    # getting the coding layer (the smallest hidden layer):
    coding_layer_activations = tf.get_default_graph().get_tensor_by_name("a%s:0" % coding_layer_index)  # rows are activations -> coding_layer_activation[i] would be activation of image i
    p_hats = tf.reduce_mean(coding_layer_activations, axis=0)   # is now a row vector
    loss_sparsity = beta*tf.reduce_sum( p*tf.log(p/p_hats) + (1-p)*tf.log((1-p)/(1-p_hats)) )       # regularisation for making the encoding sparse

#loss = tf.losses.mean_squared_error(labels=noiseless, predictions=a)
loss = loss_reconstruction + beta*loss_sparsity
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operator = optimizer.minimize(loss=loss, var_list=tf.trainable_variables())

initializer = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(initializer)

    #############
    # Training: #
    #############
    for cycle in range(training_cycles):
        
        image_batch, label_batch = mnist_data.train.next_batch(batch_size)
        noise_batch = np.random.randn(*image_batch.shape)*0.2 + image_batch

        loss_val, _ = sess.run([loss, training_operator], feed_dict={noiseless: image_batch, data: noise_batch} )
    
    ############
    # Testing: #
    ############
    test_batch, test_labels = mnist_data.test.next_batch(test_batch_size)
    test_batch_noise = np.random.randn(*test_batch.shape)*0.2 + test_batch

    created_images = sess.run(a, feed_dict={data: test_batch_noise})    

    for img_index in range(test_batch_size):
        
        unnoised = test_batch[img_index].reshape(28,28)
        noised = test_batch_noise[img_index].reshape(28,28)
        
        denoised = created_images[img_index].reshape(28,28)

        all_3 = np.concatenate((unnoised, noised, denoised),axis=1)
        plt.imshow(all_3, cmap="gray")
        plt.savefig("%s.jpg" % img_index)
    
    # #############################
    # # Denoising image of a car: #
    # #############################
    bil_unnoised = mpimg.imread("bil.jpg")[:,:,0].reshape(1,input_dim) / 255.
    bil_noise = np.random.randn(*bil_unnoised.shape)*0.2 + bil_unnoised
    
    bil_denoised = sess.run(a, feed_dict={data: bil_noise})

    sammen= np.concatenate((bil_unnoised.reshape(28,28), bil_noise.reshape(28,28), bil_denoised.reshape(28,28)), axis=1)
    plt.imshow(sammen, cmap="gray")
    plt.savefig("bil_denoised.jpg")
    #plt.show()
    print("done")
