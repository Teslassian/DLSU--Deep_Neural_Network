import time
import numpy as np
import h5py
import matplotlib as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *

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
