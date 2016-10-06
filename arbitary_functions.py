import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

#size of hidden nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#number of outputs of the function
n_outputs = 1

batch_size = 100

#Random function to approximate
def a_function(x,y,z,a):
    return (x*(y/(z-a)))
#Placeholder for input and output of function
x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float')
#Get some random values for the input of the function
#Here, I am just using random_normal, but anything can be used. A good idea being to truncate the values for the input to a specific range
x_batch = tf.Variable(tf.random_normal([batch_size, 4]))

def neural_network_model(data):
    """
    Function to define a neural network
    Args:
        data: tensorflow placeholder for batch of data
    Returns:
        output: tensorflow opoeration to compute output of size [batch_size, n_outputs]
    """
    #Define the weights and biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([4, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_outputs])),
                    'biases':tf.Variable(tf.random_normal([n_outputs])),}

    #compute the outputs of each layer
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    """
    Function to train the neural network
    Args:
        x: tensorflow placeholder for batch of data
    """
    #our prediction is whatever the model returns
    prediction = neural_network_model(x)
    #the cost here is just the absolute difference between what the network predicts and the actual output of the function
    cost = tf.reduce_mean(tf.abs(prediction- y))
    #the optimizer minimizes the cost
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #we run this for 10 epochs
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            #get a batch of data for x
            epoch_x = sess.run(x_batch)
            #compute the actual value of y using x. Each dimension in x is used as the input to get a batch of y
            epoch_y = a_function(epoch_x[:,0],epoch_x[:,1],epoch_x[:,2],epoch_x[:,3])
            #get the cost of each epoch
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

train_neural_network(x)
