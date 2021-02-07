# import time
import numpy as np
# import h5py
import matplotlib as plt
# import scipy
# from PIL import Image
# from scipy import ndimage
from dnn_app_utils_v3 import *

#---------------------------------------------------------------------------------------------------------------------#
# Plot configuration
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Provide a seed for consistent pseudorandomness
np.random.seed(1)

# Load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# # Picture Example
# index = 10
# plt.imshow(train_x_orig[index])
# plt.show()
# print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Explore the dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the length, width and 'height' into one vector
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Normalize the data
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

#---------------------------------------------------------------------------------------------------------------------#
# Constants for the 2-layer network
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

# Implementation of the 2-layer network
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    # Initialize the parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop - gradient descent
    for i in range(0, num_iterations):

        # Forward propagation
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        # Cost computation
        cost = compute_cost(A2, Y)

        # Initialize backpropagation
        dA2 = - (np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        # Backpropagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')

        # Set grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Unpack parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters # return parameters

# Train the 2-layer network
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

# Test the accuracy on the training set
predictions_train = predict(train_x, train_y, parameters)

# Test the accuracy on the test set
predictions_test = predict(test_x, test_y, parameters)

#---------------------------------------------------------------------------------------------------------------------#
# Constants for the 4-layer network
layers_dims = [12288, 20, 7, 5, 1]

# Implementation of the 4-layer network
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []

    # Parameter initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop - gradient descent
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # Cost computation
        cost = compute_cost(AL, Y)

        # Backpropagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100th training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

# Train the 4-layer network
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

# Test the accuracy on the training set
pred_train = predict(train_x, train_y, parameters)

# Test the accuracy on the test set
pred_test = predict(test_x, test_y, parameters)

#---------------------------------------------------------------------------------------------------------------------#
print_mislabeled_images(classes, test_x, test_y, pred_test)
