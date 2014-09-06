#!/usr/bin/env python
import numpy as np
from mnist_numpy import read_mnist
from nnet_toolkit import nnet
import cluster_select_func as csf
from autoconvert import autoconvert
import sys
import time

#h5py used for saving results to a file
import h5py

#constants
MODE_P1 = 0
MODE_P2 = 1

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
    params_file = sys.argv[1]
else:
    params_file = 'mnist_train_nonstationary_cluster_subset_multilayer_ensemble_params.py'
    
p = {}
execfile(params_file,p)

#Code commented below - feel free to remove later
#for ensemble networks we DO NOT support command line parameters for now
#grab extra parameters from command line
#for i in range(2,len(sys.argv)):
#    (k,v) = sys.argv[i].split('=')
#    v = autoconvert(v)
#    p[k] = v
#    print(str(k) + ":" + str(v))
    
def replace_centroids(net_layer,mask):
    number_to_replace = net_layer.number_to_replace
    neuron_used_indices = net_layer.eligibility_count.argsort()
    replace_indices = neuron_used_indices[0:number_to_replace]
    samples_tmp = net_layer.input[:,mask]
    samples = samples_tmp[:,0:number_to_replace]
    net_layer.centroids[replace_indices,:] = samples.transpose()
    #TODO: weighted euclidean stuff here

    #set the neurons we replaced to most used
    net_layer.eligibility_count[replace_indices] += 1.0 #np.max(net_layer.eligibility_count)

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
    class_data = np.ones((labels.shape[0],10))*p['incorrect_target']
    for i in range(labels.shape[0]):
        class_data[i,labels[i]] = 1.0;

    if(p['use_float32']):
        sample_data = np.asarray(sample_data,np.float32)
        class_data = np.asarray(class_data,np.float32)

    return (sample_data,class_data)

def load_newsgroup_data(indices_to_load,dataset,p,max_features):
    f_handle = h5py.File('/home/bgoodric/research/python/nn_experiments/data/dataset_20newsgroups_'+str(max_features)+'.h5py','r')
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


def pca_reduce(data):
    data_means = np.mean(data,axis=0)
    data = np.copy(data - data_means)
    data_cov = np.dot(data.transpose(),data);
    pca_transform = np.real(np.linalg.eig(data_cov)[1])
    pca_transform.astype(np.float32) #We don't need complex component
    return (pca_transform,data_means)

def pca_restore(pca_transform,data_reduced,data_means):
    data_restored = np.dot(data_reduced,pca_transform.transpose()[0:data_reduced.shape[1],:])
    data_restored += data_means
    return data_restored

def normalize_data(data,means,stds):
    data_reduced = (data - means)/stds
    data_reduced = data_reduced*(2./3.)
    data_reduced[data_reduced > 1.5]  =  1.5
    data_reduced[data_reduced < -1.5] = -1.5
    return data_reduced

P1_list = p['P1_list'][1:]
P1_list = [int(x) for x in P1_list.split('L')]
P2_list = p['P2_list'][1:]
P2_list = [int(x) for x in P2_list.split('L')]
#P1_list = [int(x) for x in P1_list]
#P2_list = [int(x) for x in P2_list]
print('P1_list: ' + str(P1_list))
print('P2_list: ' + str(P2_list))

total_list = P1_list + P2_list

print("Loading Data...")
if(p['dataset'] == "MNIST"):
    (data_full,class_data) = load_data(tuple(total_list),"training",p)
    (test_data_full,test_class_data) = load_data(tuple(total_list),"testing",p)
else:
    (data_full,class_data) = load_newsgroup_data(tuple(total_list),"training",p,2000)
    (test_data_full,test_class_data) = load_newsgroup_data(tuple(total_list),"testing",p,2000)
train_size = data_full.shape[0]

print("Splitting classes")
#Only get classes for 1,2,3,4
#class_data = class_data[:,1:5]
#test_class_data = test_class_data[:,1:5]

#split data into two parts P1 and P2, based on class
P1_mask = (np.argmax(class_data,axis=1) == P1_list[0])
for d in P1_list:
    P1_mask = np.logical_or(P1_mask,(np.argmax(class_data,axis=1) == d))

P2_mask = (np.argmax(class_data,axis=1) == P2_list[0])
for d in P2_list:
    P2_mask = np.logical_or(P2_mask,(np.argmax(class_data,axis=1) == d))

#split test into two parts P1 and P2 based on Class
P1_test_mask = (np.argmax(test_class_data,axis=1) == P1_list[0])
for d in P1_list:
    P1_test_mask = np.logical_or(P1_test_mask,(np.argmax(test_class_data,axis=1) == d))

P2_test_mask = (np.argmax(test_class_data,axis=1) == P2_list[0])
for d in P2_list:
    P2_test_mask = np.logical_or(P2_test_mask,(np.argmax(test_class_data,axis=1) == d))

num_labels = len(P1_list)

print("P1 Samples: " + str(np.sum(P1_mask)) + " P2 Samples: " + str(np.sum(P2_mask)))
print("P2 Test Samples: " + str(np.sum(P1_test_mask)) + " P2 Test Samples: " + str(np.sum(P2_test_mask)))

print("Normalizing...")
#we should normalize the pca reduced data
if(p.has_key('skip_pca') and p['skip_pca'] == True):
    print("Skipping PCA Reduction...")
    data_reduced = data_full
    test_data_reduced = test_data_full
    reduce_to = data_full.shape[1];
else:
    print("Doing PCA Reduction...")
    reduce_to = p['reduce_to']

    #pca reduce
    (pca_transform,data_means) = pca_reduce(data_full)
    data_reduced = np.dot(data_full,pca_transform[:,0:reduce_to])
    test_data_reduced = np.dot(test_data_full,pca_transform[:,0:reduce_to])
    pca_data_means = np.mean(data_reduced,axis=0)
    pca_data_std = np.std(data_reduced,axis=0)
    data_reduced = normalize_data(data_reduced,pca_data_means,pca_data_std)
    test_data_reduced = normalize_data(test_data_reduced,pca_data_means,pca_data_std)

sample_data1 = data_reduced[P1_mask,:]
sample_data2 = data_reduced[P2_mask,:]

class_data1 = class_data[P1_mask,:]
class_data2 = class_data[P2_mask,:]

class_data1 = class_data1.transpose()
class_data2 = class_data2.transpose()

class_data1 = class_data1[P1_list,:]
class_data2 = class_data2[P2_list,:]

#make size of P1 and P2 the same size
#sample_size = min(sample_data1.shape[0],sample_data2.shape[0])
#sample_data1 = sample_data1[0:sample_size]
#sample_data2 = sample_data2[0:sample_size]
#class_data1 = class_data1[0:sample_size]
#class_data2 = class_data2[0:sample_size]

test_data1 = test_data_reduced[P1_test_mask,:]
test_data2 = test_data_reduced[P2_test_mask,:]

test_class_data1 = test_class_data[P1_test_mask,:]
test_class_data2 = test_class_data[P2_test_mask,:]

test_class1 = test_class_data1.transpose()
test_class2 = test_class_data2.transpose()

test_class1 = test_class1[P1_list,:]
test_class2 = test_class2[P2_list,:]
print("Network Initialization...")

np.random.seed(p['random_seed']);

if(p.has_key('random_variance')):
    sample_data1 = sample_data1 + np.random.normal(0,float(p['random_variance']),sample_data1.shape)
    sample_data2 = sample_data2 + np.random.normal(0,float(p['random_variance']),sample_data2.shape)
    sample_data1 = np.asarray(sample_data1,np.float32)
    sample_data2 = np.asarray(sample_data2,np.float32)
    sample_data1[sample_data1 > 1.5] = 1.5
    sample_data1[sample_data1 < -1.5] = -1.5
    sample_data2[sample_data2 > 1.5] = 1.5
    sample_data2[sample_data2 < -1.5] = -1.5

    print('random_variance... ' + str(float(p['random_variance'])) + ' ' + str(np.max(np.abs(sample_data1))) + ' ' + str(np.max(np.abs(sample_data2))))

training_epochs = p['training_epochs']

minibatch_size = p['minibatch_size']

def init_network(pnet,input_size):
    num_hidden = pnet['num_hidden']

    nodes_per_group = pnet['nodes_per_group']
    nodes_per_group2 = pnet['nodes_per_group2']
    nodes_per_group3 = pnet['nodes_per_group3']


    layers = [];
    layers.append(nnet.layer(input_size))
    layers.append(nnet.layer(pnet['num_hidden'],pnet['activation_function'],nodes_per_group=nodes_per_group,
                                 initialization_scheme=pnet['initialization_scheme'],
                                 initialization_constant=pnet['initialization_constant'],
                                 dropout=pnet['dropout'],sparse_penalty=pnet['sparse_penalty'],
                                 sparse_target=pnet['sparse_target'],use_float32=pnet['use_float32'],
                                 momentum=pnet['momentum'],maxnorm=pnet['maxnorm'],step_size=pnet['learning_rate']))

    #Add 2nd and 3rd hidden layers if there are parameters indicating that we should
    if(pnet.has_key('num_hidden2') and pnet['num_hidden2'] is not None):
        layers.append(nnet.layer(pnet['num_hidden2'],pnet['activation_function2'],nodes_per_group=nodes_per_group2,
                                 initialization_scheme=pnet['initialization_scheme2'],
                                 initialization_constant=pnet['initialization_constant2'],
                                 dropout=pnet['dropout2'],sparse_penalty=pnet['sparse_penalty2'],
                                 sparse_target=pnet['sparse_target2'],use_float32=pnet['use_float32'],
                                 momentum=pnet['momentum2'],maxnorm=pnet['maxnorm2'],step_size=pnet['learning_rate2']))

    if(pnet.has_key('num_hidden3') and pnet['num_hidden3'] is not None):
        layers.append(nnet.layer(pnet['num_hidden3'],pnet['activation_function3'],nodes_per_group=nodes_per_group3,
                                 initialization_scheme=pnet['initialization_scheme3'],
                                 initialization_constant=pnet['initialization_constant3'],
                                 dropout=pnet['dropout3'],sparse_penalty=pnet['sparse_penalty3'],
                                 sparse_target=pnet['sparse_target3'],use_float32=pnet['use_float32'],
                                 momentum=pnet['momentum3'],maxnorm=pnet['maxnorm3'],step_size=pnet['learning_rate3']))

    layers.append(nnet.layer(num_labels,pnet['activation_function_final'],use_float32=pnet['use_float32'],
                                 step_size=pnet['learning_rate_final'],momentum=pnet['momentum_final']))
    #init net
    net = nnet.net(layers)

    if(pnet.has_key('cluster_func') and pnet['cluster_func'] is not None):
        #net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
        net.layer[0].centroids = np.asarray(((np.ones((net.layer[0].weights.shape))*10.0)),np.float32)
        net.layer[0].select_func = csf.select_names[pnet['cluster_func']]
        print('cluster_func: ' + str(csf.select_names[pnet['cluster_func']]))
        net.layer[0].centroid_speed = pnet['cluster_speed']
        net.layer[0].num_selected = pnet['clusters_selected']
        net.layer[0].number_to_replace = pnet['number_to_replace']
        if(p.has_key('do_cosinedistance') and pnet['do_cosinedistance']):
            net.layer[0].do_cosinedistance = True
            print('cosine set to true')

    if(pnet.has_key('num_hidden2') and pnet.has_key('cluster_func2') and pnet['cluster_func2'] is not None):
        #net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
        net.layer[1].centroids = np.asarray(((np.ones((net.layer[1].weights.shape))*10.0)),np.float32)
        net.layer[1].select_func = csf.select_names[pnet['cluster_func2']]
        print('cluster_func: ' + str(csf.select_names[pnet['cluster_func2']]))
        net.layer[1].centroid_speed = pnet['cluster_speed2']
        net.layer[1].num_selected = pnet['clusters_selected2']
        net.layer[1].number_to_replace = pnet['number_to_replace']
        if(p.has_key('do_cosinedistance') and pnet['do_cosinedistance']):
            net.layer[1].do_cosinedistance = True
            print('cosine set to true')

    if(pnet.has_key('num_hidden3') and pnet.has_key('cluster_func3') and pnet['cluster_func3'] is not None):
        #net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
        net.layer[2].centroids = np.asarray(((np.ones((net.layer[2].weights.shape))*10.0)),np.float32)
        net.layer[2].select_func = csf.select_names[pnet['cluster_func3']]
        print('cluster_func: ' + str(csf.select_names[pnet['cluster_func3']]))
        net.layer[2].centroid_speed = pnet['cluster_speed3']
        net.layer[2].num_selected = pnet['clusters_selected3']
        net.layer[2].number_to_replace = pnet['number_to_replace']
        if(p.has_key('do_cosinedistance') and pnet['do_cosinedistance']):
            net.layer[2].do_cosinedistance = True
            print('cosine set to true')
    
    net.do_clustering = False
    if(pnet['cluster_func'] is not None):
        net.do_clustering = True
    if(pnet['cluster_func2'] is not None):
        net.do_clustering = True
    if(pnet['cluster_func3'] is not None):
        net.do_clustering = True

    return net

net_list = []
num_ensembles = int(len(p['num_hidden']))
for e in range(num_ensembles):
    pnet = {}
    pnet['num_hidden']                    = p['num_hidden'][e]
    pnet['nodes_per_group']               = p['nodes_per_group'][e]
    pnet['activation_function']           = p['activation_function'][e]
    pnet['initialization_scheme']         = p['initialization_scheme'][e]
    pnet['initialization_constant']       = p['initialization_constant'][e]
    pnet['dropout']                       = p['dropout'][e]
    pnet['sparse_penalty']                = p['sparse_penalty'][e]
    pnet['sparse_target']                 = p['sparse_target'][e]
    pnet['momentum']                      = p['momentum'][e]
    pnet['maxnorm']                       = p['maxnorm'][e]
    pnet['learning_rate']                 = p['learning_rate'][e]

    pnet['num_hidden2']                   = p['num_hidden2'][e]
    pnet['nodes_per_group2']              = p['nodes_per_group2'][e]
    pnet['activation_function2']          = p['activation_function2'][e]
    pnet['initialization_scheme2']        = p['initialization_scheme2'][e]
    pnet['initialization_constant2']      = p['initialization_constant2'][e]
    pnet['dropout2']                      = p['dropout2'][e]
    pnet['sparse_penalty2']               = p['sparse_penalty2'][e]
    pnet['sparse_target2']                = p['sparse_target2'][e]
    pnet['momentum2']                     = p['momentum2'][e]
    pnet['maxnorm2']                      = p['maxnorm2'][e]
    pnet['learning_rate2']                = p['learning_rate2'][e]

    pnet['num_hidden3']                   = p['num_hidden3'][e]
    pnet['nodes_per_group3']              = p['nodes_per_group3'][e]
    pnet['activation_function3']          = p['activation_function3'][e]
    pnet['initialization_scheme3']        = p['initialization_scheme3'][e]
    pnet['initialization_constant3']      = p['initialization_constant3'][e]
    pnet['dropout3']                      = p['dropout3'][e]
    pnet['sparse_penalty3']               = p['sparse_penalty3'][e]
    pnet['sparse_target3']                = p['sparse_target3'][e]
    pnet['momentum3']                     = p['momentum3'][e]
    pnet['maxnorm3']                      = p['maxnorm3'][e]
    pnet['learning_rate3']                = p['learning_rate3'][e]

    pnet['activation_function_final']     = p['activation_function_final'][e]
    pnet['learning_rate_final']           = p['learning_rate_final'][e]
    pnet['momentum_final']                = p['momentum_final'][e]


    pnet['cluster_func']                  = p['cluster_func'][e]
    pnet['cluster_speed']                 = p['cluster_speed'][e]
    pnet['clusters_selected']             = p['clusters_selected'][e]

    pnet['cluster_func2']                 = p['cluster_func2'][e]
    pnet['cluster_speed2']                = p['cluster_speed2'][e]
    pnet['clusters_selected2']            = p['clusters_selected2'][e]

    pnet['cluster_func3']                 = p['cluster_func3'][e]
    pnet['cluster_speed3']                = p['cluster_speed3'][e]
    pnet['clusters_selected3']            = p['clusters_selected3'][e]

    pnet['number_to_replace']             = p['number_to_replace'][e]

    pnet['do_cosinedistance']             = p['do_cosinedistance'][e]
    pnet['use_float32'] = p['use_float32']
    pnet['reduce_to'] = p['reduce_to']


    nnetwork = init_network(pnet,reduce_to)
    nnetwork.ens_count = e
    net_list.append((nnetwork))

save_interval = p['save_interval']

save_time = time.time()

#these are the variables to save
train_mse_list = [];
train_missed_list = [];
train_missed_percent_list = [];
train_missed_majorityvote_list = [];
train_missed_percent_majorityvote_list = [];

test_mse1_list = [];
test_missed1_list = [];
test_missed_percent1_list = [];
test_missed1_majorityvote_list = [];
test_missed_percent1_majorityvote_list = [];

test_mse2_list = [];
test_missed2_list = [];
test_missed_percent2_list = [];
test_missed2_majorityvote_list = [];
test_missed_percent2_majorityvote_list = [];

training_mode_list = [];

test_missed_percent1 = 1.0
test_missed_percent2 = 1.0
 
best_percent_missed1 = 1.0
best_percent_missed2 = 1.0
best_percent_missed1_epoch = 0
best_percent_missed2_epoch = 0
 
if(p.has_key('shuffle_rate')):
    shuffle_rate = p['shuffle_rate']
if(p.has_key('shuffle_type')):
    shuffle_type = p['shuffle_type']
if(p.has_key('shuffle_missed_percent')):
    shuffle_missed_percent = p['shuffle_missed_percent']
if(p.has_key('shuffle_max_epochs')):
    shuffle_max_epochs = p['shuffle_max_epochs']

shuffle_epoch = -1
quit_epoch = training_epochs
p2_training_epochs = 300

training_mode = MODE_P1
(sample_data,class_data) = (sample_data1,class_data1)


save_and_exit=False
t = time.time()

def do_train_iteration(net,train_sample_data,classification,epoch_count,minibatch_count,do_shuffle):
    #--- do this for  each network in the ensemble
    net.input = train_sample_data
    net.feed_forward()
    net.error = net.output - classification

    if(net.do_clustering):
        #between back_propagate and update weights, we check for errors
        for l in range(num_labels):
            #Get a mask that tells which elements refer to this label
            mask = np.equal(l,np.argmax(classification,0))
            #get MSE for this label
            net.error_mean[l] = np.mean(net.error[:,mask]**2)
            #if error goes up, difference will be positive
            net.error_mean_difference[l] = net.error_mean[l]/net.error_mean_avg[l]
            net.error_mean_avg[l] = var_alpha*net.error_mean_avg[l] + (1.0 - var_alpha)*net.error_mean[l]
        #Append error mean difference for each class label to the log
        net.error_mean_difference_log.append(np.copy(net.error_mean_difference));
        
        for l in range(num_labels):
            #if we have threshold_cheat on then it automatically lays down centroids when P1->P2 switch occurs, else try to detect it using threshold
            exceeded = False
            if(p.has_key('threshold_cheat') and p['threshold_cheat'] is not None):
                #on very first minibatch of new epoch, do switch
                if((epoch_count == 0 or do_shuffle) and minibatch_count == 0):
                    exceeded = True
            elif(net.error_mean_difference[l] > error_difference_threshold):
                exceeded = True
            if(exceeded):
                net.error_thresh_list.append((i,l))
                mask = np.equal(l,np.argmax(classification,0))
                print("NETWORK "+str(net.ens_count)+": ERROR EXCEEDED TRESHOLD FOR LABEL " + str(l))
                for layer_num in range(len(net.layer)):
                    str_list = ("","2","3","final")
                    str_to_append = str_list[layer_num]
                    if(net.layer[layer_num].select_func is not None):
                        net.feed_forward()
                        net.error = net.output - classification
                        replace_centroids(net.layer[layer_num],mask)
                        net.feed_forward()
                #ensure error does not jump up again immediately
                net.error_mean_avg[l] = net.error_mean[l] + 1.0

                #only lower learning rate on first label (not on all of them)
                if(l == 0):
                    if(p.has_key('lowerlearningrate') and p['lowerlearningrate'] == True):
                        random_select = np.random.random()
                        if(random_select < p['lowerlearningrate_probability']):
                            print("NETWORK " + str(net.ens_count) + ": lowering learning rate" )
                            for layer in net.layer:
                                layer.step_size = layer.step_size*p['lowerlearningrate_factor']
                    for layer_index,layer in enumerate(net.layer):
                        print("NETWORK " + str(net.ens_count) + ": Layer " + str(layer_index) + ": learning rate: " + str(layer.step_size))

print("init centroid detection stuff")
for e in range(int(len(p['num_hidden']))):
    nnetwork = net_list[e]
    nnetwork.error_mean = np.zeros((num_labels,1),dtype=np.float32)
    nnetwork.error_mean_avg = np.ones((num_labels,1),dtype=np.float32)*.001
    nnetwork.error_mean_difference = np.zeros((num_labels,1),dtype=np.float32)
    nnetwork.error_thresh_list = []
    nnetwork.error_mean_difference_log = []

error_difference_threshold = p['error_difference_threshold']
var_alpha = p['var_alpha']

print("Begin Training...")
for i in range(training_epochs):

    do_shuffle = False
    if(shuffle_type == 'shuffle_rate'):
         if(i > 0 and (not (i%shuffle_rate))):
             do_shuffle = True
    elif(shuffle_type == 'missed_percent'):
         if(test_missed_percent1 < shuffle_missed_percent):
             shuffle_epoch = i
             do_shuffle = True
             shuffle_type = '' #don't shuffle again
    elif(shuffle_type == 'no_improvement'):
         if(training_mode == MODE_P1 and (i > best_percent_missed1_epoch + shuffle_max_epochs)):
             print('It has been '+str(shuffle_max_epochs)+' epochs since last improvement...')
             best_percent_missed2 = i #make sure to reset the counter
             shuffle_epoch = i
             do_shuffle = True
             quit_epoch = i + p2_training_epochs
         if(training_mode == MODE_P2 and (i > quit_epoch)):
             print('we have trained on P2 for '+str(p2_training_epochs)+' epochs... quitting')
             save_and_exit=True


    if(do_shuffle):
        #if we're in mode P2, then we need to switch to mode P1
        if(training_mode == MODE_P2):
            (sample_data,class_data) = (sample_data1,class_data1)
            training_mode = MODE_P1
        #if we're in mode P1, then swap to P2
        elif(training_mode == MODE_P1):
            (sample_data,class_data) = (sample_data2,class_data2)
            training_mode = MODE_P2
        print('shuffled to ' + str(training_mode));
    
    train_size = sample_data.shape[0]
    minibatch_count = int(train_size/minibatch_size)
    
    #shuffle data
    rng_state = np.random.get_state();
    np.random.shuffle(sample_data)
    np.random.set_state(rng_state)
    np.random.shuffle(class_data.transpose())
    
    #count number of correct
    train_missed = 0.0
    train_mse = 0.0
    train_missed_majorityvote = 0.0
    for j in range(minibatch_count):
        #grab a minibatch
        train_sample_data = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        classification = class_data[:,j*minibatch_size:(j+1)*minibatch_size]
        #classification = np.transpose(class_data[j*minibatch_size:(j+1)*minibatch_size]) 
        #if(training_mode == MODE_P1):
        #    classification = classification[0:2,:]
        #elif(training_mode == MODE_P2):
        #    classification = classification[2:4,:]

        ens_error = np.zeros(classification.shape)
        ens_output = np.zeros(classification.shape)
        ens_output_majorityvote = np.zeros(classification.shape)
        for e in range(num_ensembles):
            nnetwork = net_list[e]
            do_train_iteration(nnetwork,train_sample_data,classification,i,j,do_shuffle)


            #NOTE: Error is calculated in do_train_iteration
            nnetwork.back_propagate()
            nnetwork.update_weights()

            #update cluster centroids
            for k in range(len(nnetwork.layer)):
                str_list = ("","2","3","final")
                str_to_append = str_list[k]
                if(nnetwork.layer[k].select_func is not None):
                    csf.update_names[p['cluster_func' + str_to_append][e]](nnetwork.layer[k])
            ens_output = ens_output + nnetwork.output
            ens_error = ens_error + (nnetwork.error**2)
            #get a one hot encoded vote of the output (for majority calculation)
            vote = (nnetwork.output == np.max(nnetwork.output,axis=0))
            #add this vote to the majority vote
            ens_output_majorityvote = ens_output_majorityvote + vote.astype(int)

        ens_output /= num_ensembles
        ens_error /= num_ensembles
        #use ensemble output to get error rates
        train_mse = train_mse + np.sum(ens_error)
        c = np.argmax(classification,0)
        guess = np.argmax(ens_output,0)
        train_missed = train_missed + np.sum(c != guess)

        guess_majorityvote = np.argmax(ens_output_majorityvote,0)
        train_missed_majorityvote = train_missed_majorityvote + np.sum(c != guess_majorityvote)


    train_mse = float(train_mse)/float(train_size)
    train_missed_percent = float(train_missed)/float(train_size)
    train_missed_percent_majorityvote = float(train_missed_majorityvote)/float(train_size)
    for e in range(num_ensembles):
        net_list[e].train = False

    #feed test set through to get test 1 rates
    ens_output = np.zeros(test_class1.shape)
    ens_error = np.zeros(test_class1.shape)
    ens_output_majorityvote = np.zeros(test_class1.shape)

    sys.stdout.write("ensemble 1 miss rate: ")
    for e in range(num_ensembles):
        nnetwork = net_list[e]
        nnetwork.input = np.transpose(test_data1)
        nnetwork.feed_forward()
        nnetwork.error = nnetwork.output - test_class1
        ens_error = ens_error + nnetwork.error**2
        ens_output = ens_output + nnetwork.output
        c = np.argmax(test_class1,0)
        test_guess1 = np.argmax(nnetwork.output,0)
        test_missed1 = np.sum(c != test_guess1)
        test_size1 = test_data1.shape[0]
        sys.stdout.write(str(float(test_missed1)/float(test_size1)) + ' ')
        #majority vote
        vote = (nnetwork.output == np.max(nnetwork.output,axis=0))
        ens_output_majorityvote = ens_output_majorityvote + vote.astype(int)
    sys.stdout.write('\n')
    ens_output /= num_ensembles
    ens_error /= num_ensembles
    test_guess1 = np.argmax(ens_output,0)
    c = np.argmax(test_class1,0)
    test_missed1 = np.sum(c != test_guess1)
    test_size1 = test_data1.shape[0]
    test_mse1 = np.sum(ens_error)
    test_mse1 = float(test_mse1)/float(test_size1)
    test_missed_percent1 = float(test_missed1)/float(test_size1)
    #majority vote
    test_guess1_majorityvote = np.argmax(ens_output_majorityvote,0)
    test_missed1_majorityvote = np.sum(c != test_guess1_majorityvote)
    test_missed_percent1_majorityvote = float(test_missed1_majorityvote)/float(test_size1)

    ens_output = np.zeros(test_class2.shape)
    ens_error = np.zeros(test_class2.shape)
    ens_output_majorityvote = np.zeros(test_class2.shape)
    sys.stdout.write("ensemble 2 miss rate: ")
    for e in range(num_ensembles):
        nnetwork = net_list[e]
        nnetwork.input = np.transpose(test_data2)
        nnetwork.feed_forward()
        nnetwork.error = nnetwork.output - test_class2
        ens_error = ens_error + nnetwork.error**2
        ens_output = ens_output + nnetwork.output
        c = np.argmax(test_class2,0)
        test_guess2 = np.argmax(nnetwork.output,0)
        test_missed2 = np.sum(c != test_guess2)
        test_size2 = test_data2.shape[0]
        sys.stdout.write(str(float(test_missed2)/float(test_size2)) + ' ')
        #majority vote
        vote = (nnetwork.output == np.max(nnetwork.output,axis=0))
        ens_output_majorityvote = ens_output_majorityvote + vote.astype(int)
    sys.stdout.write('\n')
    ens_output /= num_ensembles
    ens_error /= num_ensembles
    #get test 2 rates
    test_guess2 = np.argmax(ens_output,0)
    c = np.argmax(test_class2,0)
    test_missed2 = np.sum(c != test_guess2)
    test_size2 = test_data2.shape[0]
    test_mse2 = np.sum(ens_error)
    test_mse2 = float(test_mse2)/float(test_size2)
    test_missed_percent2 = float(test_missed2)/float(test_size2)
    #majority vote
    test_guess2_majorityvote = np.argmax(ens_output_majorityvote,0)
    test_missed2_majorityvote = np.sum(c != test_guess2_majorityvote)
    test_missed_percent2_majorityvote = float(test_missed2_majorityvote)/float(test_size2)

    for e in range(num_ensembles):
        net_list[e].train = True

    #log everything for saving
    train_mse_list.append(train_mse)
    train_missed_list.append(train_missed)
    train_missed_percent_list.append(train_missed_percent)
    train_missed_majorityvote_list.append(train_missed_majorityvote)
    train_missed_percent_majorityvote_list.append(train_missed_percent_majorityvote)
    
    #test rate 1 and 2
    test_mse1_list.append(test_mse1)
    test_missed1_list.append(test_missed1)
    test_missed_percent1_list.append(test_missed_percent1)
    test_missed1_majorityvote_list.append(test_missed1_majorityvote)
    test_missed_percent1_majorityvote_list.append(test_missed_percent1_majorityvote)
    
    test_mse2_list.append(test_mse2)
    test_missed2_list.append(test_missed2)
    test_missed_percent2_list.append(test_missed_percent2)
    test_missed2_majorityvote_list.append(test_missed2_majorityvote)
    test_missed_percent2_majorityvote_list.append(test_missed_percent2_majorityvote)

    training_mode_list.append(training_mode)
#    print('epoch ' + "{: 4d}".format(i) + ": " + " mse_old: " + "{:<8.4f}".format(test_mse_old) + " acc_old: " + "{:.4f}".format(test_accuracy_old)
#    + " mse_new: " + "{:8.4f}".format(test_mse_new) + " acc_new: " + "{:.4f}".format(test_accuracy_new))
    time_elapsed = time.time() - t
    t = time.time()
    if(best_percent_missed1 > test_missed_percent1):
        best_percent_missed1 = test_missed_percent1
        best_percent_missed1_epoch = i

    if(best_percent_missed2 > test_missed_percent2):
        best_percent_missed2 = test_missed_percent2
        best_percent_missed2_epoch = i



    print('Train :               epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(train_mse) + " percent missed: " + "{0:<8.4f}".format(train_missed_percent) + " majority: " + "{0:<8.4f}".format(train_missed_percent_majorityvote) + str(time_elapsed))
    print('Test 1: (P1 Weights): epoch ' + "{0: 4d}".format(i) + 
    " mse: " + "{0:<8.4f}".format(test_mse1) + " missed: " + "{0: 5d}".format(test_missed1) + " percent missed: " + "{0:<8.4f}".format(test_missed_percent1) 
    + " majority: " + "({0:<8.4f})".format(test_missed_percent1_majorityvote) + " * {0:<8.4f}".format(best_percent_missed1));
    print('Test 2: (P2 weights): epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(test_mse2) + " missed: " + "{0: 5d}".format(test_missed2) + " percent missed: " + "{0:<8.4f}".format(test_missed_percent2) 
    + " majority: " + "({0:<8.4f})".format(test_missed_percent2_majorityvote) + " * {0:<8.4f}".format(best_percent_missed2));


    #f_out.write(str(train_mse) + "," + str(train_missed_percent) + "," + str(test_missed_percent) + "\n")
    if(time.time() - save_time > save_interval or i == training_epochs-1 or save_and_exit==True):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['train_mse_list'] = np.array(train_mse_list);
        f_handle['train_missed_list'] = np.array(train_missed_list);
        f_handle['train_missed_percent_list'] = np.array(train_missed_percent_list);
        f_handle['train_missed_majorityvote_list'] = np.array(train_missed_majorityvote_list);
        f_handle['train_missed_percent_majorityvote_list'] = np.array(train_missed_percent_majorityvote_list);
        
        f_handle['test_mse1_list'] = np.array(test_mse1_list);
        f_handle['test_missed1_list'] = np.array(test_missed1_list);
        f_handle['test_missed_percent1_list'] = np.array(test_missed_percent1_list);
        f_handle['test_missed1_majorityvote_list'] = np.array(test_missed1_majorityvote_list);
        f_handle['test_missed_percent1_majorityvote_list'] = np.array(test_missed_percent1_majorityvote_list);
        
        f_handle['test_mse2_list'] = np.array(test_mse2_list);
        f_handle['test_missed2_list'] = np.array(test_missed2_list);
        f_handle['test_missed_percent2_list'] = np.array(test_missed_percent2_list);
        f_handle['test_missed2_majorityvote_list'] = np.array(test_missed2_majorityvote_list);
        f_handle['test_missed_percent2_majorityvote_list'] = np.array(test_missed_percent2_majorityvote_list);

        f_handle['training_mode_list'] = np.array(training_mode_list)

        #log when error difference exceeded threshold, and the actual error difference
        for e in range(num_ensembles):
            f_handle['error_thresh_list_ens' + str(e+1)] = np.array(net_list[e].error_thresh_list)
            f_handle['error_mean_difference_log_ens' + str(e+1)] = np.array(net_list[e].error_mean_difference_log)
        f_handle['minibatch_count'] = minibatch_count
        f_handle['shuffle_epoch'] = int(shuffle_epoch)

   
        #iterate through all parameters and save them in the parameters group
        p_group = f_handle.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f_handle.close();
        save_time = time.time();
        if(save_and_exit):
            break
