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
    params_file = 'mnist_train_dictionary_classifier_params.py'
    
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
    incorrect_target = p['incorrect_target'];
    correct_target = p['correct_target'];
    class_data = np.ones((labels.shape[0],10))*incorrect_target
    for i in range(labels.shape[0]):
        class_data[i,labels[i]] = correct_target;

    if(p['use_float32']):
        sample_data = np.asarray(sample_data,np.float32)
        class_data = np.asarray(class_data,np.float32)
        
    return (sample_data,class_data)

if(p.has_key('data_fname') and p['data_fname'] is not None):
    print('Loading training data from:\n' + str(p['data_fname']))
    f = h5.File(p['data_fname'])
    print('data was at epoch: ' + str(f['epoch'].value))
    sample_data = np.array(f['sample_data']).transpose()
    print('sample_data_max:(before normalization) ' + str(np.max(np.abs(sample_data))))
    #sample_data_mean = np.mean(sample_data,axis=0)
    #sample_data_std = np.max(np.abs(sample_data),axis=0)
    #sample_data = sample_data/sample_data_std
    #print('sample_data_max: ' + str(np.max(np.abs(sample_data))))
    class_data = np.array(f['class_data'])
    test_data = np.array(f['test_data']).transpose()
    #test_data = test_data/sample_data_std
    #print('test_data_max: ' + str(np.max(np.abs(test_data))))
    test_class = np.array(f['test_class'])
    class_data[class_data > .01] = p['correct_target']
    class_data[class_data < -.01] = p['incorrect_target']
    test_class[test_class > .01] = p['correct_target']
    test_class[test_class < -.01] = p['incorrect_target']
    print('data successfully loaded...')
else:
    (sample_data,class_data) = load_data(range(10),"training",p)
    (test_data,test_class) = load_data(range(10),"testing",p)
test_class = test_class.transpose()
train_size = sample_data.shape[0]
test_size  = test_data.shape[0]

input_size = sample_data.shape[1]

#cluster using k-means
num_centroids = int(p['num_centroids'])

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
        for k in range(int(num_centroids)):
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
#if clusters selected is the same as the number of neurons then we can skip this step (since everything will always be selected)
elif(p['clusters_selected'] != p['num_centroids']):
    centroids = do_kmeans(sample_data);
    f = h5.File(p['data_dir'] + 'mnist_initial_centroids_' + str(num_centroids) + '.h5py','w')
    f['centroids'] = centroids
    f.close()

#now we have a k-means clustered set of centroids.

#create a classifier network

layers = [];
layers.append(nnet.layer(input_size))

if(not p.has_key('do_logistic') or p['do_logistic'] == False):
    layers.append(nnet.layer(num_centroids,p['activation_function'],
                             initialization_scheme=p['initialization_scheme'],
                             initialization_constant=p['initialization_constant'],
                             dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                             sparse_target=p['sparse_target'],use_float32=p['use_float32'],
                             momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

layers.append(nnet.layer(10,p['activation_function_final'],
                         initialization_scheme=p['initialization_scheme_final'],
                         initialization_constant=p['initialization_constant_final'],
                         use_float32=p['use_float32'],
                         momentum=p['momentum_final'],step_size=p['learning_rate']))


#feed mnist through the network

#init net
net = nnet.net(layers)
if(p['clusters_selected'] != p['num_centroids']):
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

#these are the variables to save
train_mse_list = [];
train_missed_list = [];
train_missed_percent_list = [];

test_mse_list = [];
test_missed_list = [];
test_missed_percent_list = [];

best_missed_percent = 1.0
best_epoch=1

end_type = 'fixed'
if(p.has_key('end_type')):
    end_type = p['end_type']
if(p.has_key('num_epochs')):
    num_epochs = p['num_epochs']

minibatch_size = p['minibatch_size']
save_interval = p['save_interval']
save_and_exit=False
save_time = time.time()
t = time.time()

for i in range(training_epochs):
    train_size = sample_data.shape[0]
    minibatch_count = int(train_size/minibatch_size)
    
    #shuffle data
    rng_state = np.random.get_state();
    np.random.shuffle(sample_data)
    np.random.set_state(rng_state)
    np.random.shuffle(class_data)

    train_mse = 0.0
    train_missed = 0.0
    for j in range(minibatch_count):
        #grab a minibatch
        net.input = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        tmp = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        net.feed_forward()
        classification = np.transpose(class_data[j*minibatch_size:(j+1)*minibatch_size])
        net.error = net.output - classification
        guess = np.argmax(net.output,0)
        c = np.argmax(classification,0)
        train_mse = train_mse + np.sum(net.error**2)
        train_missed = train_missed + np.sum(c != guess)

        net.back_propagate()
        net.update_weights()
        #update cluster centroids
        if(p['clusters_selected'] != p['num_centroids']):
            csf.update_names[p['cluster_func']](net.layer[0])
    train_missed_percent = float(train_missed)/float(train_size)

    net.train = False

    #feed test set through to get test 1 rates
    net.input = np.transpose(test_data)
    net.feed_forward()
    test_guess = np.argmax(net.output,0)
    c = np.argmax(test_class,0)
    test_missed = np.sum(c != test_guess)
    net.error = net.output - test_class
    test_mse = np.sum(net.error**2)
    test_mse = float(test_mse)/float(test_size)
    test_missed_percent = float(test_missed)/float(test_size)
    net.train = True

    #log everything for saving
    train_mse_list.append(train_mse)
    train_missed_list.append(train_missed)
    train_missed_percent_list.append(train_missed_percent)
    test_mse_list.append(test_mse)
    test_missed_list.append(test_missed)
    test_missed_percent_list.append(test_missed_percent)

    #record best
    if(best_missed_percent >= test_missed_percent):
        best_missed_percent = test_missed_percent
        best_epoch = i

    if(end_type == 'no_improvement' and i > (best_epoch + num_epochs)):
        save_and_exit = True

    train_mse = float(train_mse)/float(train_size)
    time_elapsed = time.time() - t
    t = time.time()

    print("epoch " + str(i) + " mse " + str(train_mse));
    print('Train :               epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(train_mse) + " percent missed: " + "{0:<8.4f}".format(train_missed_percent) + str(time_elapsed))
    print('Test  : (P1 Weights): epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(test_mse) + " missed: " + "{0: 5d}".format(test_missed) +
    " percent missed: " + "{0:<8.4f}".format(test_missed_percent) + "* {0:<8.4f}".format(best_missed_percent))

    mse_list.append(train_mse)

    if(time.time() - save_time > save_interval or i == training_epochs-1 or save_and_exit==True):
        print('saving results...');
        f = h5.File(str(p['results_dir']) + str(p['simname']) + str(p['version']) + '.h5py','w')
        if(hasattr(net.layer[0],'centroids')):
            f['centroids'] = net.layer[0].centroids
        f['weights_0'] = net.layer[0].weights
        if(not p.has_key('do_logistic') or p['do_logistic'] == False):
            f['weights_1'] = net.layer[1].weights
        f['epoch'] = i
        f['mse_list'] = np.array(mse_list)
        f['train_mse_list'] = np.array(train_mse_list)
        f['train_missed_list'] = np.array(train_missed_list)
        f['train_missed_percent_list'] = np.array(train_missed_percent_list)
        f['test_mse_list'] = np.array(test_mse_list)
        f['test_missed_list'] = np.array(test_missed_list)
        f['test_missed_percent_list'] = np.array(test_missed_percent_list)

        #iterate through all parameters and save them in the parameters group
        p_group = f.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f.close()
        save_time = time.time()

        if(save_and_exit):
            break
