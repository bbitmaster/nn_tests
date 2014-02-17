#!/usr/bin/env python
import numpy as np
import pylab
from mnist_numpy import read_mnist
from nnet_toolkit import nnet
from autoconvert import autoconvert
import cluster_select_func as csf
import sys
import time
import os
import h5py as h5

num_centroids = 100

def load_data(filename):
    f = h5.File(filename,'r')
    centroids = np.array(f['centroids'])
    weights_0 = np.array(f['weights_0'])
    weights_1 = np.array(f['weights_1'])
    mse_list = np.array(f['mse_list'])
    print('centroid data loaded. Shape: ' + str(centroids.shape))
    f.close()
    return (centroids,weights_0,weights_1,mse_list)

(centroids,weights_0,weights_1,mse_list) = load_data('../results/mnist_train_dictionary_initialtest_v2.5.h5py')

num_rows = 20
num_columns = 40

tilegrid = np.zeros((28*num_rows,28*num_columns))

for i in range(num_rows):
    for j in range(num_columns):
        index = i*num_columns + j
        tile = np.reshape(weights_1.transpose()[index,0:784],(28,28))
        tile = (tile - np.min(tile))/(np.max(tile) - np.min(tile))
        tilegrid[i*28:(i*28+28),j*28:(j*28+28)] = tile
#tilegrid = np.abs(tilegrid)
#tilegrid = tilegrid**2
#tilegrid = (tilegrid - np.min(tilegrid))/(np.max(tilegrid) - np.min(tilegrid))
print(str(np.min(tilegrid)) + " " + str(np.max(tilegrid)))
pylab.imshow(tilegrid,cmap='gray',interpolation='nearest')
pylab.show()

