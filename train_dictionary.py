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
from sklearn import linear_model

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
    params_file = sys.argv[1]
else:
    params_file = 'mnist_train_dictionary_params.py'
    
p = {}
execfile(params_file,p)

def apply_weight_decay(net,decay_constant):
    for l in reversed(net.layer):
        l.weights = l.weights*(1.0 - decay_constant)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    p[k] = v
    print(str(k) + ":" + str(v))
    
#first load mnist
def load_mnist_data(digits,dataset,p):
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


def load_newsgroup_data(indices_to_load,dataset,p,max_features=2000):
    f_handle = h5.File('/home/bgoodric/research/python/nn_experiments/data/dataset_20newsgroups_'+str(max_features)+'.h5py','r')
    if(dataset == 'training'):
        sample_data  = f_handle['train_data']
        labels= f_handle['train_class']
    if(dataset == 'testing'):
        sample_data  = f_handle['test_data']
        labels = f_handle['test_class']
    
    class_count = max(labels) + 1
    #build classification data in the form of neuron outputs
    class_data = np.ones((labels.shape[0],class_count))*p['incorrect_target']

    for i in range(labels.shape[0]):
        class_data[i,labels[i]] = 1.0;

    if(p['use_float32']):
        sample_data = np.asarray(sample_data,np.float32)
        class_data = np.asarray(class_data,np.float32)

    return (sample_data,class_data)


if(p.has_key('dataset_name') and p['dataset_name'] == 'newsgroups'):
    (sample_data,class_data) = load_newsgroup_data(range(10),"training",p)
    (test_data,test_class) = load_newsgroup_data(range(10),"testing",p)
else:
    (sample_data,class_data) = load_mnist_data(range(10),"training",p)
    (test_data,test_class) = load_mnist_data(range(10),"testing",p)

train_size = sample_data.shape[0]
input_size=sample_data.shape[1]
test_size = test_data.shape[0]
print('input size: ' + str(input_size) + ' train_size: ' + str(train_size))
#init random number generator
np.random.seed(p['random_seed']);

#cluster using k-means
num_centroids = p['num_centroids']

select_indices = np.random.randint(train_size,size=num_centroids);

def do_kmeans(sample_data):
    train_size = sample_data.shape[0]
    #init clusters
    centroids = sample_data[select_indices,:]
    if(p.has_key('max_iters')):
        max_iters = p['max_iters']
    else:
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
        if(matching == train_size):
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
if(p.has_key('nodes_per_group')):
    nodes_per_group=p['nodes_per_group']
else:
    nodes_per_group=p['num_hidden']

layers = [];
layers.append(nnet.layer(input_size))
layers.append(nnet.layer(num_centroids,p['activation_function'],
                         nodes_per_group=nodes_per_group,
                         initialization_scheme=p['initialization_scheme'],
                         initialization_constant=p['initialization_constant'],
                         dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                         sparse_target=p['sparse_target'],use_float32=p['use_float32'],
                         momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))
layers.append(nnet.layer(input_size,
                         initialization_scheme=p['initialization_scheme_final'],
                         initialization_constant=p['initialization_constant_final'],
                         use_float32=p['use_float32'],
                         momentum=p['momentum_final'],step_size=p['learning_rate']))


#feed mnist through the network

#init net
net = nnet.net(layers)
if(p['do_cosinedistance']):
    net.layer[0].do_cosinedistance = True

if(p['cluster_func'] is not None):
    net.layer[0].centroids = centroids
    net.layer[0].centroids = np.append(net.layer[0].centroids,np.ones((1,net.layer[0].centroids.shape[1]),dtype=net.layer[0].centroids.dtype),axis=0)
    net.layer[0].centroids = np.append(net.layer[0].centroids,np.ones((net.layer[0].centroids.shape[0],1),dtype=net.layer[0].centroids.dtype),axis=1)
    net.layer[0].select_func = csf.select_names[p['cluster_func']]
    net.layer[0].centroid_speed = p['cluster_speed']
    net.layer[0].num_selected = p['clusters_selected']

training_epochs = p['training_epochs']

mse_list = []
if(p.has_key('l2_weight_decay_constant')):
    l2_weight_decay_constant = p['l2_weight_decay_constant']
else:
    l2_weight_decay_constant = 0.0

minibatch_size = p['minibatch_size']
save_interval = p['save_interval']
#save_and_exit=False
save_time = time.time()
t = time.time()
best_mse = 1e99
best_mse_count = 0


def save_results():
    print('saving results...');
    f = h5.File(str(p['results_dir']) + str(p['simname']) + '_' + str(p['version']) + '.h5py','w')
    if(hasattr(net.layer[0],'centroids')):
        f['centroids'] = net.layer[0].centroids
    f['weights_0'] = net.layer[0].weights
    f['weights_1'] = net.layer[1].weights
    f['epoch'] = i
    f['mse_list'] = np.array(mse_list)
    if(p.has_key('save_hidden') and p['save_hidden'] == True):
        print('saving hidden layer')
        net.input = np.transpose(sample_data)
        net.feed_forward()
        f['sample_data'] = np.array(net.layer[0].output)
        f['class_data'] = np.array(class_data)
        net.input = np.transpose(test_data)
        net.feed_forward()
        f['test_data'] = np.array(net.layer[0].output)
        f['test_class'] = np.array(test_class)
    try:
        f['miss_class_percentage'] = miss_class_percentage
        f['miss_class'] = miss_class
    except NameError:
        pass
    #iterate through all parameters and save them in the parameters group
    p_group = f.create_group('parameters');
    for param in p.iteritems():
        #only save the ones that have a data type that is supported
        if(type(param[1]) in (int,float,str)):
            p_group[param[0]] = param[1];
    f.close()
 

i=0
while 1:
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
        #tmp = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        net.feed_forward()
        net.error = net.output - np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        train_mse = train_mse + np.sum(net.error**2)

        net.back_propagate()
        net.update_weights()
        apply_weight_decay(net,l2_weight_decay_constant)

        #update cluster centroids
        if(p['cluster_func'] is not None):
            csf.update_names[p['cluster_func']](net.layer[0])

    train_mse = float(train_mse)/float(train_size)

    net.input = np.transpose(test_data)
    net.feed_forward()
    net.error = net.output - np.transpose(test_data)
    test_mse = np.sum(net.error**2)

    test_mse = float(test_mse)/float(test_size)
    best_mse_count = best_mse_count+1
    if(best_mse > test_mse+0.000000001):
        best_mse_count = 0;
        best_mse = test_mse

    time_elapsed = time.time() - t
    t = time.time()
    print("epoch " + str(i) + " train_mse: " + str(train_mse) + " test_mse: " + str(test_mse) +
        " best_count: " + str(best_mse_count) +
        " layer 0 weight norm: " + str(np.sum(net.layer[0].weights**2))+ " " + str(time_elapsed));
    sys.stdout.flush()
    mse_list.append(train_mse)


    #compute logistic regression mse if we're done
    if(i == training_epochs-1 or best_mse_count==15):
        net.input = np.transpose(sample_data)
        net.feed_forward()

        class_data_index = np.argmax(class_data,axis=1)
        test_class_index = np.argmax(test_class,axis=1)

        #logistic_regression time
        logreg = linear_model.LogisticRegression()

        print("Doing Logistic Regression...")
        logtime = time.time()
        logreg.fit(net.layer[0].output.transpose(),class_data_index)
        print("time taken: " + str(time.time() - logtime))
        
        print("Predicting")
        net.input = np.transpose(test_data)
        net.feed_forward()
        test_class_prediction = logreg.predict(net.layer[0].output.transpose())

        miss_class = test_class_prediction != test_class_index
        miss_class_percentage = float(np.sum(miss_class))/float(miss_class.shape[0])

        print("missclass percentage: " + str(miss_class_percentage))
        save_results()
        break

    if(time.time() - save_time > save_interval or i == training_epochs-1):
        save_results()
        save_time = time.time()
    i = i+1

