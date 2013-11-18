#!/usr/bin/env python
import numpy as np
from nnet_toolkit import nnet
from nnet_toolkit import select_funcs as sf
from autoconvert import autoconvert
import sys
import time

#h5py used for saving results to a file
import h5py

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
    params_file = sys.argv[1]
else:
    params_file = 'autoassociative_train_forget_params.py'
    
p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    if(v == 'minabs'):
        v = sf.minabs_select_func
    elif(v == 'maxabs'):
        v = sf.maxabs_select_func
    elif(v == 'most_negative'):
        v = sf.most_negative_select_func
    elif(v == 'most_positive'):
        v = sf.most_positive_select_func
    elif(v == 'minabs_normalized'):
        v = sf.minabs_normalized_select_func
    elif(v == 'maxabs_normalized'):
        v = sf.maxabs_normalized_select_func
    elif(v == 'most_negative_normalized'):
        v = sf.most_negative_normalized_select_func
    elif(v == 'most_positive_normalized'):
        v = sf.most_positive_normalized_select_func
    p[k] = v
    print(str(k) + ":" + str(v))

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

minibatch_size = p['minibatch_size'];

layers = [];
layers.append(nnet.layer(sample_size))
layers.append(nnet.layer(p['num_hidden'],p['activation_function'],select_func=p['select_func'],
                         select_func_params=p['num_selected_neurons'],
                         initialization_scheme=p['initialization_scheme'],
                         initialization_constant=p['initialization_constant'],
                         dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                         sparse_target=p['sparse_target']))
                         
#Add 2nd and 3rd hidden layers if there are parameters indicating that we should
if(p.has_key('num_hidden2')):
    layers.append(nnet.layer(p['num_hidden2'],p['activation_function2'],select_func=p['select_func2'],
                             select_func_params=p['num_selected_neurons2'],
                             initialization_scheme=p['initialization_scheme2'],
                             initialization_constant=p['initialization_constant2'],
                             dropout=p['dropout2'],sparse_penalty=p['sparse_penalty2'],
                             sparse_target=p['sparse_target2']))

if(p.has_key('num_hidden3')):
    layers.append(nnet.layer(p['num_hidden3'],p['activation_function3'],select_func=p['select_func3'],
                             select_func_params=p['num_selected_neurons3'],
                             initialization_scheme=p['initialization_scheme3'],
                             initialization_constant=p['initialization_constant3'],
                             dropout=p['dropout3'],sparse_penalty=p['sparse_penalty3'],
                             sparse_target=p['sparse_target3']))

layers.append(nnet.layer(sample_size,p['activation_function_final']))

learning_rate = p['learning_rate']


#init net
net = nnet.net(layers,learning_rate)

save_interval = p['save_interval']

save_time = time.time()

#these are the variables to save
test_mse_old_list = []
test_accuracy_old_list = []
test_mse_new_list = []
test_accuracy_new_list = []

sample_data = old_sample_data;
sample_targets = old_sample_targets
num_samples = num_old_samples;

#comment/uncomment to make old data contain new
if(p['nonstationary_training'] == False):
    sample_data = np.append(new_sample_data,old_sample_data,axis=0)
    sample_targets = np.append(new_sample_targets,old_sample_targets,axis=0)
    num__samples = num_old_samples + num_new_samples;

for i in range(training_epochs):
    
    if(i == p['forget_epoch']):
        sample_data = new_sample_data;
        sample_targets = new_sample_targets
        num_samples = num_new_samples;
        print('shuffling')
    minibatch_count = int(num_samples/minibatch_size)
    
    #shuffle data
    rng_state = np.random.get_state();
    np.random.shuffle(sample_data)
    np.random.set_state(rng_state)
    np.random.shuffle(sample_targets)
    
    #count number of correct

    for j in range(minibatch_count+1):
        #grab a minibatch
        net.input = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        target = np.transpose(sample_targets[j*minibatch_size:(j+1)*minibatch_size])
        net.feed_forward()
        net.error = net.output - target
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
    test_accuracy_old = 1.0 - np.float64(np.sum(accuracy))/(num_old_samples*sample_size)
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
    test_accuracy_new = 1.0 - np.float64(np.sum(accuracy))/(num_new_samples*sample_size)
    net.train=True

    #log everything for saving
    test_mse_old_list.append(test_mse_old)
    test_accuracy_old_list.append(test_accuracy_old)
    test_mse_new_list.append(test_mse_new)
    test_accuracy_new_list.append(test_accuracy_new)
    
     print('epoch ' + "{: 4d}".format(i) + ": " + " mse_old: " + "{:<8.4f}".format(test_mse_old) + " acc_old: " + "{:.4f}".format(test_accuracy_old)
     + " mse_new: " + "{:8.4f}".format(test_mse_new) + " acc_new: " + "{:.4f}".format(test_accuracy_new))
     
    if(time.time() - save_time > save_interval or i == training_epochs-1):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['train_mse_old_list'] = np.array(test_mse_old_list);
        f_handle['test_accuracy_old_list'] = np.array(test_accuracy_old_list);
        f_handle['train_mse_new_list'] = np.array(test_mse_new_list);
        f_handle['test_accuracy_new_list'] = np.array(test_accuracy_new_list);
        
        #iterate through all parameters and save them in the parameters group
        p_group = f_handle.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f_handle.close();
        save_time = time.time();
         
         
#f_out.close();
