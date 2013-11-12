#!/usr/bin/env python
import numpy as np
from mnist_numpy import read_mnist
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
    params_file = 'mnist_train_nonstationary_params.py'
    
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
    return (sample_data,class_data)

(sample_data,class_data) = load_data(range(10),"training",p)
train_size = sample_data.shape[0]

#(test_data,test_class) = load_data(range(10),"testing",p)
#test_size = test_data.shape[0]

num_hidden = p['num_hidden']

training_epochs = p['training_epochs']

minibatch_size = p['minibatch_size']


layers = [];
layers.append(nnet.layer(28*28))
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
                             
layers.append(nnet.layer(5,p['activation_function_final']))

learning_rate = p['learning_rate']

np.random.seed(p['random_seed']);

#init net
net = nnet.net(layers,learning_rate)

#f_out = open('output.txt','w')

save_interval = p['save_interval']

save_time = time.time()

#these are the variables to save
train_mse_list = [];
train_missed_list = [];
train_missed_percent_list = [];
test_mse_list1 = [];
test_missed_list1 = [];
test_missed_percent_list1 = [];
test_mse_list2 = [];
test_missed_list2 = [];
test_missed_percent_list2 = [];

shuffle_rate = p['shuffle_rate'];

train_set_to_use = 0
(sample_data1,class_data1) = load_data(range(5),"training",p);
(sample_data2,class_data2) = load_data(range(5,10),"training",p);
(test_data1,test_class1) = load_data(range(5),"testing",p);
(test_data2,test_class2) = load_data(range(5,10),"testing",p);
test_size1 = test_data1.shape[0]
test_size2 = test_data2.shape[0]
test_class1 = np.transpose(test_class1)
test_class2 = np.transpose(test_class2)
test_class1 = test_class1[0:5,:]
test_class2 = test_class2[5:10,:]

do_weight_restoration = p['do_weight_restoration']

#save output layer weights for reinitializing
if do_weight_restoration:
    outweights1 = np.copy(net.layer[-1].weights)
    outweights2 = np.copy(net.layer[-1].weights)
    
for i in range(training_epochs):
    if(not (i%shuffle_rate)):
        print('shuffling to ' + str(train_set_to_use));
        if(train_set_to_use == 0):
            (sample_data,class_data) = (sample_data1,class_data1)
            if do_weight_restoration:
                print("restoring output weights to P1")
                outweights2 = np.copy(net.layer[-1].weights)
                net.layer[-1].weights = np.copy(outweights1)
            train_set_to_use = 1
        elif(train_set_to_use == 1):
            (sample_data,class_data) = (sample_data2,class_data2)
            if do_weight_restoration:
                print("restoring output weights to P2")
                outweights1 = np.copy(net.layer[-1].weights)
                net.layer[-1].weights = np.copy(outweights2)
            train_set_to_use = 0
        train_size = sample_data.shape[0]

    minibatch_count = int(train_size/minibatch_size)
    
    #shuffle data
    rng_state = np.random.get_state();
    np.random.shuffle(sample_data)
    np.random.set_state(rng_state)
    np.random.shuffle(class_data)
    
    #count number of correct
    train_missed = 0;
    train_mse = 0;
    for j in range(minibatch_count+1):
        #grab a minibatch
        net.input = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        classification = np.transpose(class_data[j*minibatch_size:(j+1)*minibatch_size]) 
        if(train_set_to_use == 1):
            classification = classification[0:5,:]
        if(train_set_to_use == 0):
            classification = classification[5:10,:]
        net.feed_forward()
        net.error = net.output - classification
        guess = np.argmax(net.output,0)
        c = np.argmax(classification,0)
        train_missed = train_missed + np.sum(c != guess)
        train_mse = train_mse + np.sum(net.error**2)
        net.back_propagate()
        net.update_weights()
    train_missed_percent = float(train_missed)/float(train_size)

    if do_weight_restoration:
        tmpweights = np.copy(net.layer[-1].weights)
        net.layer[-1].weights = np.copy(outweights1)
    
    #feed test set through to get test 1 rates
    net.input = np.transpose(test_data1)
    net.feed_forward()
    test_guess1 = np.argmax(net.output,0)
    c = np.argmax(test_class1,0)
    test_missed1 = np.sum(c != test_guess1)
    test_mse1 = np.sum(net.error**2)
    test_missed_percent1 = float(test_missed1)/float(test_size1)

	#feed test set through to get test 2 rates
    net.input = np.transpose(test_data2)
    net.feed_forward()
    test_guess2 = np.argmax(net.output,0)
    c = np.argmax(test_class2,0)
    test_missed2 = np.sum(c != test_guess2)
    test_mse2 = np.sum(net.error**2)
    test_missed_percent2 = float(test_missed2)/float(test_size2)
    
    if do_weight_restoration:
        net.layer[-1].weights = tmpweights
    
    #log everything for saving
    train_mse_list.append(train_mse)
    train_missed_list.append(train_missed)
    train_missed_percent_list.append(train_missed_percent)
    
	#test rate 1 and 2
    test_mse_list1.append(test_mse1)
    test_missed_list1.append(test_missed1)
    test_missed_percent_list1.append(test_missed_percent1)
    test_mse_list2.append(test_mse2)
    test_missed_list2.append(test_missed2)
    test_missed_percent_list2.append(test_missed_percent2)
    
    print('epoch ' + str(i) + ": test-missed1: " + str(test_missed1) + " MSE1: " + str(test_mse1) + " percent missed1: " + str(test_missed_percent1) + " train percent missed: " + str(train_missed_percent));
    print('epoch ' + str(i) + ": test-missed2: " + str(test_missed2) + " MSE2: " + str(test_mse2) + " percent missed2: " + str(test_missed_percent2));

    #f_out.write(str(train_mse) + "," + str(train_missed_percent) + "," + str(test_missed_percent) + "\n")
    if(time.time() - save_time > save_interval or i == training_epochs-1):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['train_mse_list'] = np.array(train_mse_list);
        f_handle['train_missed_list'] = np.array(train_missed_list);
        f_handle['train_missed_percent_list'] = np.array(train_missed_percent_list);
        
        f_handle['test_mse_list1'] = np.array(test_mse_list1);
        f_handle['test_missed_list1'] = np.array(test_missed_list1);
        f_handle['test_missed_percent_list1'] = np.array(test_missed_percent_list1);
        f_handle['test_mse_list2'] = np.array(test_mse_list2);
        f_handle['test_missed_list2'] = np.array(test_missed_list2);
        f_handle['test_missed_percent_list2'] = np.array(test_missed_percent_list2);
        
        #iterate through all parameters and save them in the parameters group
        p_group = f_handle.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f_handle.close();
        save_time = time.time();
         
         
#f_out.close();
