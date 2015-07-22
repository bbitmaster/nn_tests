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

from gas_loader import load_gas_data 

#constants
MODE_P1 = 0
MODE_P2 = 1

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
    params_file = sys.argv[1]
else:
    params_file = 'mnist_train_nonstationary_cluster_subset_multilayer_params.py'
    
p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    p[k] = v
    print(str(k) + ":" + str(v))
    
def replace_centroids(net_layer,mask):
    number_to_replace = p['number_to_replace']
    neuron_used_indices = net_layer.eligibility_count.argsort()
    replace_indices = neuron_used_indices[0:number_to_replace]
    samples_tmp = net_layer.input[:,mask]
    samples = samples_tmp[:,0:number_to_replace]
    net_layer.centroids[replace_indices,:] = samples.transpose()
    #TODO: weighted euclidean stuff here

    #set the neurons we replaced to most used
    net_layer.eligibility_count[replace_indices] += 1.0 #np.max(net_layer.eligibility_count)

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

#P1_list = p['P1_list'][1:]
#P1_list = [int(x) for x in P1_list.split('L')]
#P2_list = p['P2_list'][1:]
#P2_list = [int(x) for x in P2_list.split('L')]
#P1_list = [int(x) for x in P1_list]
#P2_list = [int(x) for x in P2_list]
#print('P1_list: ' + str(P1_list))
#print('P2_list: ' + str(P2_list))

#total_list = P1_list + P2_list

#print("total list: " + str(total_list))

print("Loading Data...")
(data_full,class_data) = load_gas_data(p['num_samples'],p['correct_target'],p['incorrect_target'])

num_samples = data_full.shape[0]

np.random.seed(p['random_seed']);
rng_state = np.random.get_state();
np.random.shuffle(data_full)
np.random.set_state(rng_state)
np.random.shuffle(class_data)

test_size = int(num_samples*0.05)
train_size = num_samples - test_size

sample_data1 = np.copy(data_full[0:train_size,:])
class_data1 = np.copy(class_data[0:train_size,:])

#test_data1 = np.copy(data_full[0:train_size,:])
#test_class1 = np.copy(class_data[0:train_size,:])


test_data1 = np.copy(data_full[train_size:num_samples,:])
test_class1 = np.copy(class_data[train_size:num_samples,:])

sample_data2 = np.copy(sample_data1)
test_data2 = np.copy(test_data1)

#shuffle test and train in the same way
rng_state = np.random.get_state();
np.random.shuffle(sample_data2.transpose())
np.random.set_state(rng_state)
np.random.shuffle(test_data2.transpose())

class_data2 = np.copy(class_data1)
test_class2 = np.copy(test_class1)

class_data1 = class_data1.transpose()
class_data2 = class_data2.transpose()

test_class1 = test_class1.transpose()
test_class2 = test_class2.transpose()

print(test_class1[0:5,:])
print(test_class2[0:5,:])


if(p.has_key('random_variance')):
    print("adding noise variance: " + str(p['random_variance']))
    sample_data1 = sample_data1 + np.random.normal(0,float(p['random_variance']),sample_data1.shape)
    sample_data2 = sample_data2 + np.random.normal(0,float(p['random_variance']),sample_data2.shape)
    sample_data1 = np.asarray(sample_data1,np.float32)
    sample_data2 = np.asarray(sample_data2,np.float32)
    sample_data1[sample_data1 > 1.5] = 1.5
    sample_data1[sample_data1 < -1.5] = -1.5
    sample_data2[sample_data2 > 1.5] = 1.5
    sample_data2[sample_data2 < -1.5] = -1.5



#print(sample_data1[0,:])
#print(sample_data2[0,:])
#print(test2[0,:])
#print(sample_data2[0,:])

print("train size: " + str(train_size))
print("test size: " + str(test_size))

print("Splitting classes")
#Only get classes for 1,2,3,4
#class_data = class_data[:,1:5]
#test_class_data = test_class_data[:,1:5]

#split data into two parts P1 and P2, based on class
#P1_mask = (np.argmax(class_data,axis=1) == P1_list[0])
#for d in P1_list:
#    P1_mask = np.logical_or(P1_mask,(np.argmax(class_data,axis=1) == d))

#P2_mask = (np.argmax(class_data,axis=1) == P2_list[0])
#for d in P2_list:
#    P2_mask = np.logical_or(P2_mask,(np.argmax(class_data,axis=1) == d))

#split test into two parts P1 and P2 based on Class
#P1_test_mask = (np.argmax(test_class_data,axis=1) == P1_list[0])
#for d in P1_list:
#    P1_test_mask = np.logical_or(P1_test_mask,(np.argmax(test_class_data,axis=1) == d))

#P2_test_mask = (np.argmax(test_class_data,axis=1) == P2_list[0])
#for d in P2_list:
#    P2_test_mask = np.logical_or(P2_test_mask,(np.argmax(test_class_data,axis=1) == d))

num_labels = 2

#print("P1 Samples: " + str(np.sum(P1_mask)) + " P2 Samples: " + str(np.sum(P2_mask)))
#print("P2 Test Samples: " + str(np.sum(P1_test_mask)) + " P2 Test Samples: " + str(np.sum(P2_test_mask)))

#print("Doing PCA Reduction...")
#reduce_to = p['reduce_to']

#pca reduce
#(pca_transform,data_means) = pca_reduce(data_full)
#data_reduced = np.dot(data_full,pca_transform[:,0:reduce_to])
#test_data_reduced = np.dot(test_data_full,pca_transform[:,0:reduce_to])

print("Normalizing...")
#we should normalize the pca reduced data
#if(p.has_key('skip_pca') and p['skip_pca'] == True):
#    print("Skipping PCA Reduction...")
#data_reduced = data_full
#test_data_reduced = test_data_full
input_dims = data_full.shape[1];
#else:
#    pca_data_means = np.mean(data_reduced,axis=0)
#    pca_data_std = np.std(data_reduced,axis=0)
#    data_reduced = normalize_data(data_reduced,pca_data_means,pca_data_std)
#    test_data_reduced = normalize_data(test_data_reduced,pca_data_means,pca_data_std)

#sample_data1 = data_reduced[P1_mask,:]
#sample_data2 = data_reduced[P2_mask,:]

#class_data1 = class_data[P1_mask,:]
#class_data2 = class_data[P2_mask,:]

#class_data1 = class_data1.transpose()
#class_data2 = class_data2.transpose()

#class_data1 = class_data1[P1_list,:]
#class_data2 = class_data2[P2_list,:]

#make size of P1 and P2 the same size

#test_data1 = test_data_reduced[P1_test_mask,:]
#test_data2 = test_data_reduced[P2_test_mask,:]

#test_class_data1 = test_class_data[P1_test_mask,:]
#test_class_data2 = test_class_data[P2_test_mask,:]

#test_class1 = test_class_data1.transpose()
#test_class2 = test_class_data2.transpose()

#test_class1 = test_class1[P1_list,:]
#test_class2 = test_class2[P2_list,:]
print("Network Initialization...")

num_hidden = p['num_hidden']

training_epochs = p['training_epochs']

minibatch_size = p['minibatch_size']

if(p.has_key('nodes_per_group')):
    nodes_per_group = p['nodes_per_group']
else:
    nodes_per_group = None

if(p.has_key('nodes_per_group2')):
    nodes_per_group2 = p['nodes_per_group2']
else:
    nodes_per_group2 = None

if(p.has_key('nodes_per_group3')):
    nodes_per_group3 = p['nodes_per_group3']
else:
    nodes_per_group3 = None

layers = [];
layers.append(nnet.layer(input_dims))
layers.append(nnet.layer(p['num_hidden'],p['activation_function'],nodes_per_group=nodes_per_group,
                         initialization_scheme=p['initialization_scheme'],
                         initialization_constant=p['initialization_constant'],
                         dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                         sparse_target=p['sparse_target'],use_float32=p['use_float32'],
                         momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

#Add 2nd and 3rd hidden layers if there are parameters indicating that we should
if(p.has_key('num_hidden2')):
    layers.append(nnet.layer(p['num_hidden2'],p['activation_function2'],nodes_per_group=nodes_per_group2,
                             initialization_scheme=p['initialization_scheme2'],
                             initialization_constant=p['initialization_constant2'],
                             dropout=p['dropout2'],sparse_penalty=p['sparse_penalty2'],
                             sparse_target=p['sparse_target2'],use_float32=p['use_float32'],
                             momentum=p['momentum2'],maxnorm=p['maxnorm2'],step_size=p['learning_rate2']))

if(p.has_key('num_hidden3')):
    layers.append(nnet.layer(p['num_hidden3'],p['activation_function3'],nodes_per_group=nodes_per_group3,
                             initialization_scheme=p['initialization_scheme3'],
                             initialization_constant=p['initialization_constant3'],
                             dropout=p['dropout3'],sparse_penalty=p['sparse_penalty3'],
                             sparse_target=p['sparse_target3'],use_float32=p['use_float32'],
                             momentum=p['momentum3'],maxnorm=p['maxnorm3'],step_size=p['learning_rate3']))

layers.append(nnet.layer(num_labels,p['activation_function_final'],use_float32=p['use_float32'],
                             step_size=p['learning_rate_final'],momentum=p['momentum_final']))

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

#init net
net = nnet.net(layers)

if(p.has_key('cluster_func') and p['cluster_func'] is not None):
#    net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
    net.layer[0].centroids = np.asarray(((np.ones((net.layer[0].weights.shape))*10.0)),np.float32)
    net.layer[0].select_func = csf.select_names[p['cluster_func']]
    print('cluster_func: ' + str(csf.select_names[p['cluster_func']]))
    net.layer[0].centroid_speed = p['cluster_speed']
    net.layer[0].num_selected = p['clusters_selected']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[0].do_cosinedistance = True
        print('cosine set to true')

if(p.has_key('num_hidden2') and p.has_key('cluster_func2') and p['cluster_func2'] is not None):
#    net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
    net.layer[1].centroids = np.asarray(((np.ones((net.layer[1].weights.shape))*10.0)),np.float32)
    net.layer[1].select_func = csf.select_names[p['cluster_func2']]
    print('cluster_func: ' + str(csf.select_names[p['cluster_func2']]))
    net.layer[1].centroid_speed = p['cluster_speed']
    net.layer[1].num_selected = p['clusters_selected']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[1].do_cosinedistance = True
        print('cosine set to true')

if(p.has_key('num_hidden3') and p.has_key('cluster_func3') and p['cluster_func3'] is not None):
#    net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
    net.layer[2].centroids = np.asarray(((np.ones((net.layer[2].weights.shape))*10.0)),np.float32)
    net.layer[2].select_func = csf.select_names[p['cluster_func3']]
    print('cluster_func: ' + str(csf.select_names[p['cluster_func3']]))
    net.layer[2].centroid_speed = p['cluster_speed']
    net.layer[2].num_selected = p['clusters_selected']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[2].do_cosinedistance = True
        print('cosine set to true')

save_interval = p['save_interval']

save_time = time.time()

#these are the variables to save
train_mse_list = [];
train_missed_list = [];
train_missed_percent_list = [];

test_mse1_list = [];
test_missed1_list = [];
test_missed_percent1_list = [];

test_mse2_list = [];
test_missed2_list = [];
test_missed_percent2_list = [];

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
p2_training_epochs = 100

training_mode = MODE_P1
(sample_data,class_data) = (sample_data1,class_data1)


save_and_exit=False
t = time.time()

print("init centroid detection stuff")
error_mean = np.zeros((num_labels,1),dtype=np.float32)
error_mean_avg = np.ones((num_labels,1),dtype=np.float32)*.001
error_mean_difference = np.zeros((num_labels,1),dtype=np.float32)
error_thresh_list = []
error_mean_difference_log = []

number_to_replace = p['number_to_replace']
error_difference_threshold = p['error_difference_threshold']
var_alpha = p['var_alpha']

do_clustering = False
if(p.has_key('cluster_func') and p['cluster_func'] is not None):
    do_clustering = True
if(p.has_key('cluster_func2') and p['cluster_func2'] is not None):
    do_clustering = True
if(p.has_key('cluster_func3') and p['cluster_func3'] is not None):
    do_clustering = True

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
    train_missed = 0.0;
    train_mse = 0.0;
    for j in range(minibatch_count):
        #grab a minibatch
        train_sample_data = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        net.input = train_sample_data
        classification = class_data[:,j*minibatch_size:(j+1)*minibatch_size]
        #classification = np.transpose(class_data[j*minibatch_size:(j+1)*minibatch_size]) 
        #if(training_mode == MODE_P1):
        #    classification = classification[0:2,:]
        #elif(training_mode == MODE_P2):
        #    classification = classification[2:4,:]
        net.feed_forward()
        net.error = net.output - classification

        if(do_clustering):
            #between back_propagate and update weights, we check for errors
            for l in range(num_labels):
                #Get a mask that tells which elements refer to this label
                mask = np.equal(l,np.argmax(classification,0))
                #get MSE for this label
                error_mean[l] = np.mean(net.error[:,mask]**2)
                #if error goes up, difference will be positive
                error_mean_difference[l] = error_mean[l]/error_mean_avg[l]
                error_mean_avg[l] = var_alpha*error_mean_avg[l] + (1.0 - var_alpha)*error_mean[l]
        
            for l in range(num_labels):
                #if we have threshold_cheat on then it automatically lays down centroids when P1->P2 switch occurs, else try to detect it using threshold
                exceeded = False

                if(p.has_key('threshold_cheat') and p['threshold_cheat'] is not None):
                    #on very first minibatch of new epoch, do switch
                    if((i == 0 or do_shuffle) and j == 0):
                        exceeded = True
                elif(error_mean_difference[l] > error_difference_threshold):
                    exceeded = True
                if(exceeded):
                    error_thresh_list.append((i,l))
                    mask = np.equal(l,np.argmax(classification,0))
                    print("ERROR EXCEEDED TRESHOLD FOR LABEL " + str(l))
                    for layer_num in range(len(net.layer)):
                        str_list = ("","2","3","final")
                        str_to_append = str_list[layer_num]
                        if(p.has_key('cluster_func' + str_to_append) and p['cluster_func' + str_to_append] is not None):
                            net.feed_forward()
                            net.error = net.output - classification
                            replace_centroids(net.layer[layer_num],mask)
                            net.feed_forward()
                    #ensure error does not jump up again immediately
                    error_mean_avg[l] = error_mean[l] + 1.0

        guess = np.argmax(net.output,0)
        c = np.argmax(classification,0)
        train_mse = train_mse + np.sum(net.error**2)
        train_missed = train_missed + np.sum(c != guess)

        #Append error mean difference for each class label to the log
        error_mean_difference_log.append(np.copy(error_mean_difference));

#    np.savetxt("dmp/distances_epoch" + str(epoch) + ".csv",net.layer[0].distances,delimiter=",");
        #print(net.layer[0].saved_selected_neurons)
        net.back_propagate()
        net.update_weights()
#        print("selected is zero: " + str( np.sum(net.layer[0].saved_selected_neurons == 0,axis=0)))
#        print("output is not zero: " + str( np.sum(net.layer[0].output != 0,axis=0)))
#        import pdb; pdb.set_trace();
        #update cluster centroids
        for k in range(len(net.layer)):
            str_list = ("","2","3","final")
            str_to_append = str_list[k]
            if(p.has_key('cluster_func' + str_to_append) and p['cluster_func' + str_to_append] is not None):
                csf.update_names[p['cluster_func' + str_to_append]](net.layer[k])

    train_mse = float(train_mse)/float(train_size)
    train_missed_percent = float(train_missed)/float(train_size)
    net.train = False

    #feed test set through to get test 1 rates
    net.input = np.transpose(test_data1)
    net.feed_forward()
    test_guess1 = np.argmax(net.output,0)
    c = np.argmax(test_class1,0)
    test_missed1 = np.sum(c != test_guess1)
    net.error = net.output - test_class1
    test_size1 = test_data1.shape[0]
    test_mse1 = np.sum(net.error**2)
    test_mse1 = float(test_mse1)/float(test_size1)
    test_missed_percent1 = float(test_missed1)/float(test_size1)

    #get test 2 rates
    net.input = np.transpose(test_data2)
    net.feed_forward()
    test_guess2 = np.argmax(net.output,0)
    c = np.argmax(test_class2,0)
    test_missed2 = np.sum(c != test_guess2)
    net.error = net.output - test_class2
    test_size2 = test_data2.shape[0]
    test_mse2 = np.sum(net.error**2)
    test_mse2 = float(test_mse2)/float(test_size2)
    test_missed_percent2 = float(test_missed2)/float(test_size2)

    net.train = True

    #log everything for saving
    train_mse_list.append(train_mse)
    train_missed_list.append(train_missed)
    train_missed_percent_list.append(train_missed_percent)
    
    #test rate 1 and 2
    test_mse1_list.append(test_mse1)
    test_missed1_list.append(test_missed1)
    test_missed_percent1_list.append(test_missed_percent1)
    
    test_mse2_list.append(test_mse2)
    test_missed2_list.append(test_missed2)
    test_missed_percent2_list.append(test_missed_percent2)


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
    " mse: " + "{0:<8.4f}".format(train_mse) + " percent missed: " + "{0:<8.4f}".format(train_missed_percent) + str(time_elapsed))
    print('Test 1: (P1 Weights): epoch ' + "{0: 4d}".format(i) + 
     " mse: " + "{0:<8.4f}".format(test_mse1) + " missed: " + "{0: 5d}".format(test_missed1) + " percent missed: " + "{0:<8.4f}".format(test_missed_percent1) + " * {0:<8.4f}".format(best_percent_missed1));
    print('Test 2: (P2 weights): epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(test_mse2) + " missed: " + "{0: 5d}".format(test_missed2) + " percent missed: " + "{0:<8.4f}".format(test_missed_percent2) + " * {0:<8.4f}".format(best_percent_missed2));


    #f_out.write(str(train_mse) + "," + str(train_missed_percent) + "," + str(test_missed_percent) + "\n")
    if(time.time() - save_time > save_interval or i == training_epochs-1 or save_and_exit==True):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['train_mse_list'] = np.array(train_mse_list);
        f_handle['train_missed_list'] = np.array(train_missed_list);
        f_handle['train_missed_percent_list'] = np.array(train_missed_percent_list);
        
        f_handle['test_mse1_list'] = np.array(test_mse1_list);
        f_handle['test_missed1_list'] = np.array(test_missed1_list);
        f_handle['test_missed_percent1_list'] = np.array(test_missed_percent1_list);
        
        f_handle['test_mse2_list'] = np.array(test_mse2_list);
        f_handle['test_missed2_list'] = np.array(test_missed2_list);
        f_handle['test_missed_percent2_list'] = np.array(test_missed_percent2_list);

        f_handle['training_mode_list'] = np.array(training_mode_list)

        #log when error difference exceeded threshold, and the actual error difference
        f_handle['error_thresh_list'] = np.array(error_thresh_list)
        f_handle['error_mean_difference_log'] = np.array(error_mean_difference_log)
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
