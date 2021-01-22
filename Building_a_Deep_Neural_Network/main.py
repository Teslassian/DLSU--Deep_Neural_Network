#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v4a import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
import sys

# set defaults for plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# seed for randomly initialized weights
np.random.seed(1)

# # Function for initializing the parameters W and b
# def initialize_parameters(n_x, n_h, n_y):
#     """
#     Argument:
#     n_x -- size of the input layer
#     n_h -- size of the hidden layer
#     n_y -- size of the output layer
#
#     Returns:
#     parameters -- python dictionary containing your parameters:
#                     W1 -- weight matrix of shape (n_h, n_x)
#                     b1 -- bias vector of shape (n_h, 1)
#                     W2 -- weight matrix of shape (n_y, n_h)
#                     b2 -- bias vector of shape (n_y, 1)
#     """
#
#     np.random.seed(1)
#
#     W1 = np.random.randn(n_h, n_x) * 0.01
#     b1 = np.zeros((n_h, 1))
#     W2 = np.random.randn(n_y, n_h) * 0.01
#     b2 = np.zeros((n_y, 1))
#
#     assert(W1.shape == (n_h, n_x))
#     assert(b1.shape == (n_h, 1))
#     assert(W2.shape == (n_y, n_h))
#     assert(b2.shape == (n_y, 1))
#
#     parameters = {"W1": W1,
#                   "b1": b1,
#                   "W2": W2,
#                   "b2": b2}
#
#     return parameters
# # # Test the initialize_parameters function
# # parameters = initialize_parameters(3,2,1)
# # print("W1 = " + str(parameters["W1"]))
# # print("b1 = " + str(parameters["b1"]))
# # print("W2 = " + str(parameters["W2"]))
# # print("b2 = " + str(parameters["b2"]))

# Function for initializing the parameters W and b for a deep network
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters
# # Test the initialize_parameters_deep function
# parameters = initialize_parameters_deep([5,4,3])
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# Function for the linear part of forward propagation
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W @ A + b # Note that A refers to the inputs received from the previous layer,
                  # other than in the linear_activation_forward function, where A_prev
                  # is used to refer to those inputs

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache
# # Test the linear_forward function
# A, W, b = linear_forward_test_case()
# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))

# Function for the activation part of forward propagation
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == 'sigmoid':
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b) # linear_cache is just (A_prev, W, b)
        A, activation_cache = sigmoid(Z) # activation_cache is just Z

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b) # linear_cache is just (A_prev, W, b)
        A, activation_cache = relu(Z) # activation_cache is just Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
# # Test the linear_activation_forward function
# A_prev, W, b = linear_activation_forward_test_case()
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

# Function for forward propagation over all L layers
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2 # to obtain the number of layers

    # Implementation of relu(Linear) for hidden layers
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)

    # Implementation of sigmoid(Linear) for output layer
    AL, cache = linear_activation_forward(A, parameters['W'+str(l+1)], parameters['b'+str(l+1)], 'sigmoid')
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches
# # Test the L_model_forward function
# X, parameters = L_model_forward_test_case_2hidden()
# AL, caches = L_model_forward(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

# Function for computing the cross entropy cost J
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    cost = -1/m * (Y @ np.log(AL).T + (1-Y) @ np.log(1-AL).T)

    return cost
# # Test the compute_cost function
# Y, AL = compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))

# Function for the linear part of backpropagation
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache;
    m = A_prev.shape[1]

    dW = 1/m * dZ @ A_prev.T
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db
# # Test the linear_backward function
# dZ, linear_cache = linear_backward_test_case()
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))
