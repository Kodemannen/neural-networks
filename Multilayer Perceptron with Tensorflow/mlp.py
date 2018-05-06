import numpy as np
import tensorflow as tf
import grab_cifar10
from sys import exit as ex

data = grab_cifar10.get_data()

training_input = data["training_images"]
training_labels = data["training_labels"]

test_input = data["test_images"]
test_labels = data["test_labels"]

#training_input[:,j] er input nr j 

N = training_input.shape[1]                 # number of training images
input_nodes = training_input.shape[0]
classes = 10

# Making the labels in onehot format for both training set and test set:
training_labels_onehots = np.zeros((classes, N))
training_labels_onehots[training_labels,range(N)] = 1

test_labels_onehots = np.zeros((classes,len(test_labels)))
test_labels_onehots[test_labels,range(len(test_labels))] = 1


####################################
# Setting up the tensorflow graph: #
####################################
data = tf.placeholder(shape=(input_nodes,None), dtype=tf.float32, name="training_data") # for holding any number of datapoints at a time
label_vec = tf.placeholder(shape=(classes,None), dtype=tf.int32, name="label_vec")  
global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")                     # for counting training cycles


# Setting up the structure:
node_structure = [input_nodes, 1024, 256, 64, classes]
learning_rate = 5e-3
training_steps = 10000
batch_size =200


a = data
for i in range(len(node_structure)-1):
    layer_name = "layer%s" % i
    with tf.variable_scope(layer_name):
        #W = tf.get_variable(name="W", initializer=tf.random_normal(shape=(node_structure[i+1], node_structure[i])))
        #b = tf.get_variable(name="b", initializer= tf.random_normal(shape=(node_structure[i+1],1)))

        W = tf.get_variable(name="W", shape=(node_structure[i+1], node_structure[i]), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=(node_structure[i+1],1), initializer= tf.contrib.layers.xavier_initializer())


        z = tf.matmul(W,a, name="matmul") + b
        #a = tf.tanh(z, name="activation_function")
        
        # Activation: relu on hidden layer, softmax on the output:
        a = tf.nn.relu(z, name="activation_function") if i != len(node_structure)-2 else tf.nn.softmax(z, name="activation_function")

# Defining loss operator:
loss = tf.losses.mean_squared_error(labels=label_vec, predictions=a)
#loss = tf.losses.softmax_cross_entropy(onehot_labels=label_vec, logits=a) 

# defining accuracy operator:
estimated_class = tf.argmax(a, axis=0)  # returns the index of highest element > number from 0 to 9 that says which predicted class of current image
correct_class = tf.argmax(label_vec, axis=0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(estimated_class, correct_class), tf.float32), name="accuracy")   # stores the accuracy and will update it


# defining optimizer operator:
variables_for_training = tf.trainable_variables()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss=loss, var_list=variables_for_training)



######################
# Running the graph: #
######################

train_loss = np.zeros(training_steps)
train_accuracy = np.zeros(training_steps)
test_loss_list = []
test_accuracy_list = []


number_of_test_inputs = test_input.shape[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(training_steps):
        
        batch_indices = np.random.randint(low=0, high=N , size=(batch_size))
        
        batch_input = training_input[:,batch_indices]
        batch_labels = training_labels_onehots[:,batch_indices]
 

        loss_val, accuracy_val, _ = sess.run([loss, accuracy, training_operation], feed_dict={data: batch_input, label_vec: batch_labels})
        
        train_loss[i] = loss_val
        train_accuracy[i] = accuracy_val
        
        if i % 100 == 0:
            
            test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={data: test_input, label_vec: test_labels_onehots})
            
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)

            print("test loss = %s" % (test_loss / float(number_of_test_inputs)))
            print("test accuracy = %s" % test_accuracy)

            print("training loss = %.5f" % (loss_val / float(batch_size)))
            print("training accuracy = %.5f" % accuracy_val)

import matplotlib.pyplot as plt
plt.plot(range(training_steps), train_loss, label="training loss")
plt.legend()
plt.show()


plt.plot(np.linspace(0,training_steps, training_steps//100),test_loss_list,label="test loss")
plt.legend()
plt.show()
plt.plot(np.linspace(0,training_steps, training_steps//100),test_accuracy_list,label="test accuracy")
plt.legend()
plt.show()