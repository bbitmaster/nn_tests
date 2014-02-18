#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    param = {}
    for (key,val) in f['parameters'].iteritems():
        param[key] = val.value
    print('centroid data loaded. Shape: ' + str(centroids.shape))
    f.close()
    return (centroids,weights_0,weights_1,mse_list,param)


def plot_dictionary(num_rows,num_columns,dictionary,title=None,save_file=None):
    tilegrid = np.zeros((28*num_rows,28*num_columns))

    for i in range(num_rows):
        for j in range(num_columns):
            index = i*num_columns + j
            tile = np.reshape(dictionary[index,0:784],(28,28))
            tile = (tile - np.min(tile))/(np.max(tile) - np.min(tile))
            tilegrid[i*28:(i*28+28),j*28:(j*28+28)] = tile
    #tilegrid = np.abs(tilegrid)
    #tilegrid = tilegrid**2
    #tilegrid = (tilegrid - np.min(tilegrid))/(np.max(tilegrid) - np.min(tilegrid))
    print(str(np.min(tilegrid)) + " " + str(np.max(tilegrid)))
    plt.imshow(tilegrid,cmap='gray',interpolation='nearest')
    if(title is not None):
        plt.title(title);
    if(save_file is None):
        plt.show()
    else:
        f = plt.gcf()
        f.set_size_inches(19.2,10.8)
        plt.savefig(save_file,dpi=100)
        plt.close()

def plot_mse(mse_list,label_list):
    for i,mse in enumerate(mse_list):
        c = cm.hsv(np.linspace(0,1,len(mse_list)+1))
        data_x = range(mse.shape[0])
        data_y = mse;
        plt.plot(data_x,data_y,color=c[i,:])
    plt.title('Sparse autoencoder training curve MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(label_list,'upper right')
    plt.grid(True)
    plt.xlim(0,2000)
    plt.ylim(0,100)
    f = plt.gcf();
    f.set_size_inches(19.2,10.8)
    plt.savefig('../result_images/mnist_dictionary_training.png',dpi=100)
    plt.close()
filename_list = ['../results/mnist_train_dictionary_initialtest_v1.1.h5py',
                 '../results/mnist_train_dictionary_initialtest_v1.2.h5py',
                 '../results/mnist_train_dictionary_initialtest_v1.3.h5py',
                 '../results/mnist_train_dictionary_initialtest_v1.4.h5py',
                 '../results/mnist_train_dictionary_initialtest_v1.5.h5py',
                 '../results/mnist_train_dictionary_initialtest_v2.1.h5py',
                 '../results/mnist_train_dictionary_initialtest_v2.2.h5py',
                 '../results/mnist_train_dictionary_initialtest_v2.3.h5py',
                 '../results/mnist_train_dictionary_initialtest_v2.4.h5py',
                 '../results/mnist_train_dictionary_initialtest_v2.5.h5py']

fig_name_list = ['../result_images/mnist_train_dictionary_initialtest_v1.1',
                '../result_images/mnist_train_dictionary_initialtest_v1.2',
                '../result_images/mnist_train_dictionary_initialtest_v1.3',
                '../result_images/mnist_train_dictionary_initialtest_v1.4',
                '../result_images/mnist_train_dictionary_initialtest_v1.5',
                '../result_images/mnist_train_dictionary_initialtest_v2.1',
                '../result_images/mnist_train_dictionary_initialtest_v2.2',
                '../result_images/mnist_train_dictionary_initialtest_v2.3',
                '../result_images/mnist_train_dictionary_initialtest_v2.4',
                '../result_images/mnist_train_dictionary_initialtest_v2.5']

num_rows = 20
num_columns = 40
mse_list_list = []
mse_title_list = []
for i in range(10):
    filename = filename_list[i]
    (centroids,weights_0,weights_1,mse_list,param) = load_data(filename)
    if(i < 5):
        cosine_distance = 'Euclidean Distance'
    else:
        cosine_distance = 'Cosine Distance'
    
    mse_list_list.append(np.array(mse_list))
    mse_title = 'Clusters Selected: ' + str(param['clusters_selected']) + ' ' + cosine_distance;
    mse_title_list.append(mse_title)

    fig_name = fig_name_list[i] + str('_centroids.png')
    title = 'K-Means Selection Centroids for ' + str(param['clusters_selected']) + ' clusters selected ' + cosine_distance;
    plot_dictionary(num_rows,num_columns,centroids,title=title,save_file=fig_name)

    fig_name = fig_name_list[i] + str('_weights0.png')
    title = 'Input->Hidden Weights (encoding dictionary) for ' + str(param['clusters_selected']) + ' clusters selected ' + cosine_distance;
    plot_dictionary(num_rows,num_columns,weights_0,title=title,save_file=fig_name)

    fig_name = fig_name_list[i] + str('_weights1.png')
    title = 'Hidden->Output Weights (The dictionary) for ' + str(param['clusters_selected']) + ' clusters selected ' + cosine_distance;
    plot_dictionary(num_rows,num_columns,weights_1.transpose(),title=title,save_file=fig_name)

plot_mse(mse_list_list,mse_title_list);
