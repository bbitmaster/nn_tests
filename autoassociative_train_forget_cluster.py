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
    params_file = 'autoassociative_train_forget_cluster_params.py'
    
p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    p[k] = v
    print(str(k) + ":" + str(v))
    
def replace_centroids(net_layer):
    neuron_used_indices = net_layer.eligibility_count.argsort()
    replace_indices = neuron_used_indices[0:number_to_replace]
    samples = net_layer.input
    samples = samples.transpose()
    samples = np.repeat(samples,number_to_replace,axis=0)
    np.random.shuffle(samples)
    net_layer.centroids[replace_indices,:] = samples[0:number_to_replace]
    #TODO: weighted euclidean stuff here

    #set the neurons we replaced to most used
    net_layer.eligibility_count[replace_indices] += 1.0 #np.max(net_layer.eligibility_count)


np.random.seed(p['random_seed']);

sample_size = p['sample_size'];
num_old_samples = p['num_old_samples'];
num_new_samples = p['num_new_samples'];

new_sample_data = np.random.randint(0,2,(num_new_samples,sample_size))
old_sample_data = np.random.randint(0,2,(num_old_samples,sample_size))

new_sample_targets = np.random.randint(0,2,(num_new_samples,sample_size))
old_sample_targets = np.random.randint(0,2,(num_old_samples,sample_size))

if(p['zerosandones'] == False):
    old_sample_data = (old_sample_data-0.5)*2.0
    new_sample_data = (new_sample_data-0.5)*2.0
    old_sample_targets = (old_sample_targets-0.5)*2.0
    new_sample_targets = (new_sample_targets-0.5)*2.0

num_hidden = p['num_hidden']

training_epochs = p['training_epochs']

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
layers.append(nnet.layer(sample_size))
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

layers.append(nnet.layer(sample_size,p['activation_function_final'],use_float32=p['use_float32'],
                             step_size=p['learning_rate_final'],momentum=p['momentum_final']))

np.random.seed(p['random_seed']);




layers.append(nnet.layer(sample_size,p['activation_function_final']))

learning_rate = p['learning_rate']


#init net
net = nnet.net(layers,learning_rate)

#init net
net = nnet.net(layers)
if(p.has_key('cluster_func') and p['cluster_func'] is not None):
    net.layer[0].centroids = np.asarray(((np.ones((net.layer[0].weights.shape))*10.0)),np.float32)
    net.layer[0].select_func = csf.select_names[p['cluster_func']]
    print('cluster_func: ' + str(csf.select_names[p['cluster_func']]))
    net.layer[0].centroid_speed = p['cluster_speed']
    net.layer[0].num_selected = p['clusters_selected']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[0].do_cosinedistance = True
        print('cosine set to true')

if(p.has_key('num_hidden2') and p.has_key('cluster_func2') and p['cluster_func2'] is not None):
    net.layer[1].centroids = np.asarray(((np.ones((net.layer[1].weights.shape))*10.0)),np.float32)
    net.layer[1].select_func = csf.select_names[p['cluster_func2']]
    print('cluster_func2: ' + str(csf.select_names[p['cluster_func2']]))
    net.layer[1].centroid_speed = p['cluster_speed']
    net.layer[1].num_selected = p['clusters_selected']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[1].do_cosinedistance = True
        print('cosine set to true')

if(p.has_key('num_hidden3') and p.has_key('cluster_func3') and p['cluster_func3'] is not None):
    net.layer[2].centroids = np.asarray(((np.ones((net.layer[2].weights.shape))*10.0)),np.float32)
    net.layer[2].select_func = csf.select_names[p['cluster_func3']]
    print('cluster_func3: ' + str(csf.select_names[p['cluster_func3']]))
    net.layer[2].centroid_speed = p['cluster_speed']
    net.layer[2].num_selected = p['clusters_selected']
    if(p.has_key('do_cosinedistance') and p['do_cosinedistance']):
        net.layer[2].do_cosinedistance = True
        print('cosine set to true')


save_interval = p['save_interval']

save_and_exit=False
save_time = time.time()
t = time.time()


best_percent_old = 10000.0
best_percent_new = 10000.0
best_percent_old_epoch = 0
best_percent_new_epoch = 0

#these are the variables to save
test_mse_old_list = []
test_missed_old_list = []
test_mse_new_list = []
test_missed_new_list = []

sample_data = old_sample_data;
sample_targets = old_sample_targets
num_samples = num_old_samples;

if(p.has_key('shuffle_rate')):
    shuffle_rate = p['shuffle_rate']
if(p.has_key('shuffle_type')):
    shuffle_type = p['shuffle_type']
if(p.has_key('shuffle_max_epochs')):
    shuffle_max_epochs = p['shuffle_max_epochs']


print("init centroid detection stuff")
error_mean = 0
error_mean_avg = .001
error_mean_difference = 0.0
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


training_mode = MODE_P1
shuffle_epoch = -1
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
        if(training_mode == MODE_P1 and (i > best_percent_old_epoch + shuffle_max_epochs)):
            print('It has been '+str(shuffle_max_epochs)+' epochs since last improvement...')
            best_percent_new = i #make sure to reset the counter
            shuffle_epoch = i
            do_shuffle = True
        if(training_mode == MODE_P2 and (i > best_percent_new_epoch + shuffle_max_epochs)):
            print('It has been '+str(shuffle_max_epochs)+' epochs since last improvement... quitting...')
            save_and_exit=True

    if(do_shuffle):
        #if we're in mode P2, then we need to switch to mode P1
        if(training_mode == MODE_P2):
            (sample_data,sample_target) = (old_sample_data,old_sample_targets)
            training_mode = MODE_P1
        #if we're in mode P1, then swap to P2
        elif(training_mode == MODE_P1):
            (sample_data,sample_targets) = (new_sample_data,new_sample_targets)
            training_mode = MODE_P2
        print('shuffled to ' + str(training_mode));

    for b in range(p['batches_per_epoch']):
        #grab a batch
        net.input = np.transpose(sample_data)
        target = np.transpose(sample_targets)
#        print(str(net.layer[1].weights.shape))
        net.feed_forward()
        net.error = net.output - target
        
        if(do_clustering):
            #between back_propagate and update weights, we check for errors
            error_mean = np.mean(net.error**2)
            #if error goes up, difference will be positive
            error_mean_difference = error_mean/error_mean_avg
            error_mean_avg = var_alpha*error_mean_avg + (1.0 - var_alpha)*error_mean
        
            exceeded = False
            if(p.has_key('threshold_cheat') and p['threshold_cheat'] is not None):
                exceeded = True
            elif(error_mean_difference > error_difference_threshold):
                exceeded = True
#            if(b == 0):
#                print("error difference: " + str(error_mean_difference))
            if(exceeded):
                error_thresh_list.append(i)
                print("ERROR EXCEEDED TRESHOLD")
                for layer_num in range(len(net.layer)):
                    str_list = ("","2","3","final")
                    str_to_append = str_list[layer_num]
                    if(p.has_key('cluster_func' + str_to_append) and p['cluster_func' + str_to_append] is not None):
                        net.feed_forward()
                        net.error = net.output - target
                        replace_centroids(net.layer[layer_num])
                        net.feed_forward()
                #ensure error does not jump up again immediately
                error_mean = np.mean(net.error**2)
                error_mean_avg = error_mean + 1.0
        net.back_propagate()
        net.update_weights()





    #get old data test error rate and accuracy
    net.train=False
    net.input = np.transpose(old_sample_data)
    net.feed_forward()
    net.error = net.output - np.transpose(old_sample_targets)
    test_mse_old = (np.sum(net.error**2)/num_old_samples)
    if(p['zerosandones']):
        threshold = 0.5
    else:
        threshold = 0.0
    binary_output = net.output > threshold;
    binary_output = binary_output.astype(np.float64)
    if(p['zerosandones'] == False):
        binary_output = (binary_output-0.5)*2.0
    accuracy = np.abs(binary_output - np.transpose(old_sample_targets))
    accuracy = accuracy > .0001
    test_missed_old = np.float64(np.sum(accuracy))/(num_old_samples*sample_size)
    net.train=True

    #get new data test error rate and accuracy
    net.train=False
    net.input = np.transpose(new_sample_data)
    net.feed_forward()
    net.error = net.output - np.transpose(new_sample_targets)
    test_mse_new = (np.sum(net.error**2)/num_new_samples)
    if(p['zerosandones']):
        threshold = 0.5
    else:
        threshold = 0.0
    binary_output = net.output > threshold;
    binary_output = binary_output.astype(np.float64)
    if(p['zerosandones'] == False):
        binary_output = (binary_output-0.5)*2.0
    accuracy = np.abs(binary_output - np.transpose(new_sample_targets))
    accuracy = accuracy > .0001
    test_missed_new = np.float64(np.sum(accuracy))/(num_new_samples*sample_size)
    net.train=True

    #log everything for saving
    test_mse_old_list.append(test_mse_old)
    test_missed_old_list.append(test_missed_old)
    test_mse_new_list.append(test_mse_new)
    test_missed_new_list.append(test_missed_new)

    error_mean_difference_log.append(error_mean_difference)

    time_elapsed = time.time() - t
    t = time.time()

    if(best_percent_old > test_missed_old):
        best_percent_old = test_missed_old
        best_percent_old_epoch = i

    if(best_percent_new > test_missed_new):
        best_percent_new = test_missed_new
        best_percent_new_epoch = i

    
    print('epoch ' + "{: 4d}".format(i) + ":" + " mse_old:" + "{:<8.4f}".format(test_mse_old) + " miss_old:" + "{:.4f}".format(test_missed_old) + " *:" + "{:<8.4f}".format(best_percent_old)
    + " mse_new: " +"{:8.4f}".format(test_mse_new)  + " missed_new: " + "{:.4f}".format(test_missed_new) + " *:" + "{:<8.4f}".format(best_percent_new) + " {:8.4f}".format(time_elapsed))
     
    if(time.time() - save_time > save_interval or i == training_epochs-1 or save_and_exit==True):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['test_mse_old_list'] = np.array(test_mse_old_list);
        f_handle['test_missed_old_list'] = np.array(test_missed_old_list);

        f_handle['test_mse_new_list'] = np.array(test_mse_new_list);
        f_handle['test_missed_new_list'] = np.array(test_missed_new_list);

        f_handle['error_thresh_list'] = np.array(error_thresh_list);
        f_handle['error_mean_difference_log'] = np.array(error_mean_difference_log);
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
print("error\nepoch")
print(str(np.min(np.array(test_missed_new_list) + np.array(test_missed_old_list))) + "\n" + str(np.argmin(np.array(test_missed_new_list) + np.array(test_missed_old_list))))
#f_out.close();
