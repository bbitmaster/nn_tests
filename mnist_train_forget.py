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
    params_file = 'mnist_train_forget_params.py'
    
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
        class_data[i,labels[i]] = p['correct_target'];

    if(p['use_float32']):
        sample_data = np.asarray(sample_data,np.float32)
        class_data = np.asarray(class_data,np.float32)
        
    return (sample_data,class_data)

(sample_data,class_data) = load_data(range(10),"training",p)
train_size = sample_data.shape[0]

(test_data,test_class) = load_data(range(10),"testing",p)
test_size = test_data.shape[0]

num_hidden = p['num_hidden']

training_epochs = p['training_epochs']

minibatch_size = p['minibatch_size']

#layers = [nnet.layer(28*28),
#          nnet.layer(num_hidden,'tanh',select_func=p['select_func'],select_func_params=p['num_selected_neurons']),
#          nnet.layer(10,'tanh')]

layers = [];
layers.append(nnet.layer(28*28))
layers.append(nnet.layer(p['num_hidden'],p['activation_function'],select_func=p['select_func'],
                         select_func_params=p['num_selected_neurons'],
                         initialization_scheme=p['initialization_scheme'],
                         initialization_constant=p['initialization_constant'],
                         dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
                         sparse_target=p['sparse_target'],use_float32=p['use_float32'],
                         momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

#Add 2nd and 3rd hidden layers if there are parameters indicating that we should
if(p.has_key('num_hidden2')):
    layers.append(nnet.layer(p['num_hidden2'],p['activation_function2'],select_func=p['select_func2'],
                             select_func_params=p['num_selected_neurons2'],
                             initialization_scheme=p['initialization_scheme2'],
                             initialization_constant=p['initialization_constant2'],
                             dropout=p['dropout2'],sparse_penalty=p['sparse_penalty2'],
                             sparse_target=p['sparse_target2'],use_float32=p['use_float32'],
                             momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

if(p.has_key('num_hidden3')):
    layers.append(nnet.layer(p['num_hidden3'],p['activation_function3'],select_func=p['select_func3'],
                             select_func_params=p['num_selected_neurons3'],
                             initialization_scheme=p['initialization_scheme3'],
                             initialization_constant=p['initialization_constant3'],
                             dropout=p['dropout3'],sparse_penalty=p['sparse_penalty3'],
                             sparse_target=p['sparse_target3'],use_float32=p['use_float32'],
                             momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

layers.append(nnet.layer(10,p['activation_function_final'],use_float32=p['use_float32'],step_size=p['learning_rate_final']))


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
test_mse_list = [];
test_missed_list = [];
test_missed_percent_list = [];

for i in range(training_epochs):
    
    #This is the forgetting test. If the epoch == forget epoch, then we want to switch the training dataset so that it only gets digits 0-5
    if(i == p['forget_epoch']):
        print('forget epoch - switching training data.')
        (sample_data,class_data) = load_data(range(5),"training",p)
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
        net.feed_forward()
        net.error = net.output - classification
        guess = np.argmax(net.output,0)
        c = np.argmax(classification,0)
        train_missed = train_missed + np.sum(c != guess)
        train_mse = train_mse + np.sum(net.error**2)
        net.back_propagate()
        net.update_weights()
    train_missed_percent = float(train_missed)/float(train_size)
    
    #feed test set through to get test rates
    net.train=False
    net.input = np.transpose(test_data)
    net.feed_forward()
    test_guess = np.argmax(net.output,0)
    c = np.argmax(np.transpose(test_class),0)
    test_missed = np.sum(c != test_guess)
    test_mse = np.sum(net.error**2)
    test_missed_percent = float(test_missed)/float(test_size)
    net.train=True
    
    #log everything for saving
    train_mse_list.append(train_mse)
    train_missed_list.append(train_missed)
    train_missed_percent_list.append(train_missed_percent)
    test_mse_list.append(test_mse)
    test_missed_list.append(test_missed)
    test_missed_percent_list.append(test_missed_percent)
    
    print('epoch ' + str(i) + ": test-missed: " + str(test_missed) + " MSE: " + str(test_mse) + " percent missed: " + str(test_missed_percent) + " train percent missed: " + str(train_missed_percent));
    #f_out.write(str(train_mse) + "," + str(train_missed_percent) + "," + str(test_missed_percent) + "\n")
    if(time.time() - save_time > save_interval or i == training_epochs-1):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['train_mse_list'] = np.array(train_mse_list);
        f_handle['train_missed_list'] = np.array(train_missed_list);
        f_handle['train_missed_percent_list'] = np.array(train_missed_percent_list);
        f_handle['test_mse_list'] = np.array(test_mse_list);
        f_handle['test_missed_list'] = np.array(test_missed_list);
        f_handle['test_missed_percent_list'] = np.array(test_missed_percent_list);
        
        #iterate through all parameters and save them in the parameters group
        p_group = f_handle.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f_handle.close();
        save_time = time.time();
         
         
#f_out.close();
