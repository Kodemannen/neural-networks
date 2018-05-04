import numpy as np
import tensorflow as tf
import grab_cifar10
import sys

data = grab_cifar10.get_data()
#data = grab_mnist.get_data()


# training_input = tf.convert_to_tensor(data["training_images"])
# training_labels = tf.convert_to_tensor(data["training_labels"])
# test_input = tf.convert_to_tensor(data["test_images"])
# test_labels = tf.convert_to_tensor(data["test_labels"])

training_input = data["training_images"]
training_labels = np.array(data["training_labels"])

test_input = np.transpose(data["test_images"])
test_labels = data["test_labels"]


N = training_input.shape[0] # number of images
input_nodes = training_input[0].shape[0]
classes = 10

test_labels_onehots = np.zeros((classes,len(test_labels)))
test_labels_onehots[test_labels,range(len(test_labels))] = 1

####################################
# Setting up the tensorflow graph: #
####################################
data = tf.placeholder(shape=(input_nodes,None), dtype=tf.float32, name="training_data") # for holding one datapoint at a time (?)
label_vec = tf.placeholder(shape=(classes,None), dtype=tf.int32, name="label_vec")  
global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")                     # for counting training cycles


# Setting up the structure:
node_structure = [input_nodes, 1024, 256, classes]
a = data
for i in range(len(node_structure)-1):
    layer_name = "layer%s" % i
    with tf.variable_scope(layer_name):
        W = tf.get_variable(name="W", shape=(node_structure[i+1], node_structure[i]), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=(node_structure[i+1],1), initializer= tf.contrib.layers.xavier_initializer())

        z = tf.matmul(W,a, name="matmul") + b
        a = tf.tanh(z, name="activation_function")


# Defining loss operator:
#loss = tf.losses.mean_squared_error(labels=label_vec, predictions=a)
loss = tf.losses.softmax_cross_entropy(onehot_labels=label_vec, logits=a)

# defining accuracy operator:
estimated_class = tf.argmax(a, axis=0)  # returns the index of highest element > number from 0 to 9 that says which predicted class of current image
correct_class = tf.argmax(label_vec, axis=0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(estimated_class, correct_class), tf.float32), name="accuracy")   # stores the accuracy and will update it


# defining optimizer operator:
variables_for_training = tf.trainable_variables()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
training_operation = optimizer.minimize(loss=loss, var_list=variables_for_training)



######################
# Running the graph: #
######################
training_steps = 10000
batch_size =500

train_loss = np.zeros(training_steps)
train_accuracy = np.zeros(training_steps)
test_loss_list = []
test_accuracy_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(training_steps):
        
        image_indices = np.random.randint(low=0, high=N , size=(batch_size))
        training_batch_input = np.transpose(training_input[image_indices])
        
        # Transforming class labels to vectors representing them:
        training_batch_label_indices = training_labels[image_indices]
        label_vecs = np.zeros((classes, batch_size))
        label_vecs[training_batch_label_indices,range(batch_size)] = 1 

        loss_val, accuracy_val, _ = sess.run([loss, accuracy, training_operation], feed_dict={data: training_batch_input, label_vec: label_vecs})
        
        train_loss[i] = loss_val
        train_accuracy[i] = accuracy_val


        
        if i % 100 == 0:
            
            test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={data: test_input, label_vec: test_labels_onehots})
            
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)

            print("test loss = %s" % test_loss)
            print("test accuracy = %s" % test_accuracy)

            print("training loss = %.5f" % loss_val)
            print("training accuracy = %.5f" % accuracy_val)

import matplotlib.pyplot as plt
plt.plot(range(training_steps), train_loss, label="training loss")
plt.legend()
plt.show()


plt.plot(np.linspace(0,training_steps, 10),test_loss_list,label="test loss")
plt.legend()
plt.show()
plt.plot(np.linspace(0,training_steps, 10),test_accuracy_list,label="test accuracy")
plt.legend()
plt.show()


    