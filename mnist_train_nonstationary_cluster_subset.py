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
    params_file = 'mnist_train_nonstationary_cluster_subset_params.py'
    
p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    p[k] = v
    print(str(k) + ":" + str(v))
    
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

P1_list = list(p['P1_list'][1:])
P1_list = [int(x) for x in P1_list]
P2_list = list(p['P2_list'][1:])
P2_list = [int(x) for x in P2_list]
total_list = P1_list + P2_list

print("Loading Data...")
#get only first 4 digits
(data_full,class_data) = load_data(tuple(total_list),"training",p)
(test_data_full,test_class_data) = load_data(tuple(total_list),"testing",p)
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

print("Doing PCA Reduction...")
reduce_to = p['reduce_to']

#pca reduce
(pca_transform,data_means) = pca_reduce(data_full)
data_reduced = np.dot(data_full,pca_transform[:,0:reduce_to])
test_data_reduced = np.dot(test_data_full,pca_transform[:,0:reduce_to])

print("Normalizing...")
#we should normalize the pca reduced data
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

num_hidden = p['num_hidden']

training_epochs = p['training_epochs']

minibatch_size = p['minibatch_size']

layers = [];
layers.append(nnet.layer(reduce_to))
layers.append(nnet.layer(p['num_hidden'],p['activation_function'],
                         initialization_scheme=p['initialization_scheme'],
                         initialization_constant=p['initialization_constant'],
                         dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                         sparse_target=p['sparse_target'],use_float32=p['use_float32'],
                         momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

#Add 2nd and 3rd hidden layers if there are parameters indicating that we should
if(p.has_key('num_hidden2')):
    layers.append(nnet.layer(p['num_hidden2'],p['activation_function2'],
                             initialization_scheme=p['initialization_scheme2'],
                             initialization_constant=p['initialization_constant2'],
                             dropout=p['dropout2'],sparse_penalty=p['sparse_penalty2'],
                             sparse_target=p['sparse_target2'],use_float32=p['use_float32'],
                             momentum=p['momentum2'],maxnorm=p['maxnorm2'],step_size=p['learning_rate2']))

if(p.has_key('num_hidden3')):
    layers.append(nnet.layer(p['num_hidden3'],p['activation_function3'],
                             initialization_scheme=p['initialization_scheme3'],
                             initialization_constant=p['initialization_constant3'],
                             dropout=p['dropout3'],sparse_penalty=p['sparse_penalty3'],
                             sparse_target=p['sparse_target3'],use_float32=p['use_float32'],
                             momentum=p['momentum3'],maxnorm=p['maxnorm3'],step_size=p['learning_rate3']))

layers.append(nnet.layer(num_labels,p['activation_function_final'],use_float32=p['use_float32'],
                             step_size=p['learning_rate_final'],momentum=p['momentum_final']))

np.random.seed(p['random_seed']);

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

if(p.has_key('cluster_func2') and p['cluster_func2'] is not None):
    net.layer[1].centroids = np.asarray((((np.random.random((net.layer[1].weights.shape)) - 0.5)*2.0)),np.float32)
    net.layer[1].select_func = csf.select_names[p['cluster_func2']]
    net.layer[1].centroid_speed = p['cluster_speed2']
    net.layer[1].num_selected = p['clusters_selected2']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[0].do_cosinedistance = True

if(p.has_key('cluster_func3') and p['cluster_func3'] is not None):
    net.layer[2].centroids = np.asarray((((np.random.random((net.layer[2].weights.shape)) - 0.5)*2.0)),np.float32)
    net.layer[2].select_func = csf.select_names[p['cluster_func3']]
    net.layer[2].centroid_speed = p['cluster_speed3']
    net.layer[2].num_selected = p['clusters_selected3']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[0].do_cosinedistance = True

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

shuffle_rate = p['shuffle_rate'];

training_mode = MODE_P1
(sample_data,class_data) = (sample_data1,class_data1)


save_and_exit=False
t = time.time()

print("init centroid detection stuff")
error_mean = np.zeros((num_labels,1),dtype=np.float32)
error_mean_avg = np.ones((num_labels,1),dtype=np.float32)*.001
error_mean_difference = np.zeros((num_labels,1),dtype=np.float32)

number_to_replace = p['number_to_replace']
error_difference_threshold = p['error_difference_threshold']
var_alpha = p['var_alpha']

print("Begin Training...")
for i in range(training_epochs):

    do_shuffle = False
    if(i > 0 and (not (i%shuffle_rate))):
        do_shuffle = True

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
        guess = np.argmax(net.output,0)
        c = np.argmax(classification,0)
        train_mse = train_mse + np.sum(net.error**2)
        train_missed = train_missed + np.sum(c != guess)

        net.back_propagate()
        if(p.has_key('cluster_func') and p['cluster_func'] is not None):
            #between back_propagate and update weights, we check for errors
            for l in range(num_labels):
                #Get a mask that tells which elements refer to this label
                mask = np.equal(l,np.argmax(classification,0))
                #get MSE for this label
                error_mean[l] = np.mean(net.error[:,mask]**2)
                #if error goes up, difference will be positive
                error_mean_difference[l] = error_mean[l]/error_mean_avg[l]
                error_mean_avg[l] = var_alpha*error_mean_avg[l] + (1.0 - var_alpha)*error_mean[l]
        
            neuron_used_indices = net.layer[0].eligibility_count.argsort()
            for l in range(num_labels):
                #print("error" + str(l) + ": " + str(error_mean[l]) + " avg " + str(error_mean_avg[l]) + " difference " + str(error_mean_difference[l]))
                if(error_mean_difference[l] > error_difference_threshold):
                    print("ERROR EXCEEDED TRESHOLD FOr LABEL " + str(l))
                    #get the 8 least selected neurons
                    replace_indices = neuron_used_indices[0:number_to_replace]
                    #print("replace indices: " + str(replace_indices))
                    neuron_used_indices = neuron_used_indices[number_to_replace:]
                    #need some sample data points -- could use k-means -- for now sample randomly
                    #samples is S x N where S is number of samples, and N is input size
                    mask = np.equal(l,np.argmax(classification,0))
                    sample_data_tmp = train_sample_data[:,mask]
                    samples = sample_data_tmp[:,0:number_to_replace]

                    #need to tack on the bias
                    samples = np.append(samples,np.ones((1,samples.shape[1]),dtype=samples.dtype),axis=0)
            
                    #replace centroids with new ones drawn from samples
                    net.layer[0].centroids[replace_indices,:] = samples.transpose()
                    if(p.has_key('do_weighted_euclidean') and p['do_weighted_euclidean']):
                        net.layer[0].centroids[replace_indices,:] = net.layer[0].centroids[replace_indices,:]*net.layer[0].weights[replace_indices,:]

                    #reset error mean
                    #add a bit of a bias, to ensure it doesn't exceed the threshold immediately again
                    error_mean_avg[l] = error_mean[l] + 1.0

#    np.savetxt("dmp/distances_epoch" + str(epoch) + ".csv",net.layer[0].distances,delimiter=",");
        #print(net.layer[0].saved_selected_neurons)
        net.update_weights()
#        print("selected is zero: " + str( np.sum(net.layer[0].saved_selected_neurons == 0,axis=0)))
#        print("output is not zero: " + str( np.sum(net.layer[0].output != 0,axis=0)))
#        import pdb; pdb.set_trace();
        #update cluster centroids
        for k in range(len(net.layer)):
            str_list = ("","2","3")
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
    print('Train :               epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(train_mse) + " percent missed: " + "{0:<8.4f}".format(train_missed_percent) + str(time_elapsed))
    print('Test 1: (P1 Weights): epoch ' + "{0: 4d}".format(i) + 
    " mse: " + "{0:<8.4f}".format(test_mse1) + " missed: " + "{0: 5d}".format(test_missed1) + " percent missed: " + "{0:<8.4f}".format(test_missed_percent1));
    print('Test 2: (P2 weights): epoch ' + "{0: 4d}".format(i) +
    " mse: " + "{0:<8.4f}".format(test_mse2) + " missed: " + "{0: 5d}".format(test_missed2) + " percent missed: " + "{0:<8.4f}".format(test_missed_percent2));

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