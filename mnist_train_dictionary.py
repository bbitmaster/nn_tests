#!/usr/bin/env python
import numpy as np
from mnist_numpy import read_mnist
from nnet_toolkit import nnet
from autoconvert import autoconvert
import cluster_select_func as csf
import sys
import time
import os
import h5py as h5

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
    params_file = sys.argv[1]
else:
    params_file = 'mnist_train_dictionary_params.py'
    
p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    p[k] = v
    print(str(k) + ":" + str(v))
    
#first load mnist
def load_data(digits,dataset,p):
    images, labels = read_mnist(digits,dataset,path=p['data_dir']);
    labels = labels.transpose()[0] #put labels in an array
    images = np.float64(images)
    #(normalize between 0-1)
    images /= 255.0 
    #normalize between -1 and 1 for hyperbolic tangent
    images = images - 0.5;
    images = images*2.0; 
    
    train_size = labels.shape[0]
    sample_data = images.reshape(train_size,28*28)

    #build classification data in the form of neuron outputs
    incorrect_target = -1;
    class_data = np.ones((labels.shape[0],10))*incorrect_target
    for i in range(labels.shape[0]):
        class_data[i,labels[i]] = 1.0;

    if(p['use_float32']):
        sample_data = np.asarray(sample_data,np.float32)
        class_data = np.asarray(class_data,np.float32)
        
    return (sample_data,class_data)

(sample_data,class_data) = load_data(range(10),"training",p)
train_size = sample_data.shape[0]

#cluster using k-means
num_centroids = p['num_centroids']

select_indices = np.random.randint(60000,size=num_centroids);

def do_kmeans(sample_data):
    #init clusters
    centroids = sample_data[select_indices,:]

    max_iters = 1000
    stop_flag = False
    num_iters = 0
    matching = 0
    while num_iters < max_iters:
        #find the clusters that belong
        if(p['do_cosinedistance']):
            #we use NEGATIVE cosine distance so the argmin function works properly. (otherwise it would be argmax)
            distances = -np.dot(centroids,sample_data.transpose())/(np.sqrt(np.sum(centroids**2.,1)[:,np.newaxis]* \
            np.sum(sample_data.transpose()**2.,0)[np.newaxis,:]))
        else:
            distances = np.sum(centroids**2,1)[:,np.newaxis] - 2*np.dot(centroids,sample_data.transpose()) + \
                        np.sum(sample_data.transpose()**2,0)[np.newaxis,:]
        distances_indices = np.argmin(distances,axis=0);
        if(num_iters > 0):
            matching = np.sum(distances_indices == distances_indices_old)
        distances_indices_old = np.copy(distances_indices)
        unused = 0
        for k in range(num_centroids):
            #TODO: check if none belong to this centroid
            if(np.sum(distances_indices == k) == 0):
                unused = unused + 1
                continue;
            centroids[k,:] = np.mean(sample_data[distances_indices == k,:],axis=0)
        num_iters = num_iters + 1
        print("iteration " + str(num_iters) + " matching: " + str(matching) + " unused: " + str(unused))
        if(matching == 60000):
            break
    return(centroids)

if(p['load_centroids'] and os.path.exists(p['data_dir'] + 'mnist_initial_centroids_' + str(num_centroids) + '.h5py')):
    f = h5.File(p['data_dir'] + 'mnist_initial_centroids_' + str(num_centroids) + '.h5py','r')
    centroids = np.array(f['centroids'])
    print('centroid data loaded. Shape: ' + str(centroids.shape))
    f.close()
else:
    centroids = do_kmeans(sample_data);
    f = h5.File(p['data_dir'] + 'mnist_initial_centroids_' + str(num_centroids) + '.h5py','w')
    f['centroids'] = centroids
    f.close()

#now we have a k-means clustered set of centroids.

#create an autoencoder network

layers = [];
layers.append(nnet.layer(28*28))
layers.append(nnet.layer(num_centroids,p['activation_function'],
                         initialization_scheme=p['initialization_scheme'],
                         initialization_constant=p['initialization_constant'],
                         dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                         sparse_target=p['sparse_target'],use_float32=p['use_float32'],
                         momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))
layers.append(nnet.layer(28*28,
                         initialization_scheme=p['initialization_scheme_final'],
                         initialization_constant=p['initialization_constant_final'],
                         use_float32=p['use_float32'],
                         momentum=p['momentum_final'],step_size=p['learning_rate']))


#feed mnist through the network

#init net
net = nnet.net(layers)
if(p['do_cosinedistance']):
    net.layer[0].do_cosinedistance = True
net.layer[0].centroids = centroids
net.layer[0].centroids = np.append(net.layer[0].centroids,np.ones((1,net.layer[0].centroids.shape[1]),dtype=net.layer[0].centroids.dtype),axis=0)
net.layer[0].centroids = np.append(net.layer[0].centroids,np.ones((net.layer[0].centroids.shape[0],1),dtype=net.layer[0].centroids.dtype),axis=1)
net.layer[0].select_func = csf.select_names[p['cluster_func']]
net.layer[0].centroid_speed = p['cluster_speed']
net.layer[0].num_selected = p['clusters_selected']

training_epochs = p['training_epochs']

mse_list = []

minibatch_size = p['minibatch_size']
save_interval = p['save_interval']
save_and_exit=False
save_time = time.time()
for i in range(training_epochs):
    train_size = sample_data.shape[0]
    minibatch_count = int(train_size/minibatch_size)
    
    #shuffle data
    rng_state = np.random.get_state();
    np.random.shuffle(sample_data)
    np.random.set_state(rng_state)
    np.random.shuffle(class_data)

    train_mse = 0
    for j in range(minibatch_count):
        #grab a minibatch
        net.input = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        tmp = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        net.feed_forward()
        net.error = net.output - np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        train_mse = train_mse + np.sum(net.error**2)

        net.back_propagate()
        net.update_weights()
        #update cluster centroids
        csf.update_names[p['cluster_func']](net.layer[0])

    train_mse = float(train_mse)/float(train_size)
    print("epoch " + str(i) + " mse " + str(train_mse));
    mse_list.append(train_mse)
    if(time.time() - save_time > save_interval or i == training_epochs-1 or save_and_exit==True):
        print('saving results...');
        f = h5.File(str(p['results_dir']) + str(p['simname']) + '_' + str(p['version']) + '.h5py','w')
        f['centroids'] = net.layer[0].centroids
        f['weights_0'] = net.layer[0].weights
        f['weights_1'] = net.layer[1].weights
        f['epoch'] = i
        f['mse_list'] = np.array(mse_list)
        #iterate through all parameters and save them in the parameters group
        p_group = f.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f.close()
        save_time = time.time()

