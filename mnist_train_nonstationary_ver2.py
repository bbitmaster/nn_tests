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

#constants
MODE_P1 = 0
MODE_P2 = 1

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

    if(p['use_float32']):
        sample_data = np.asarray(sample_data,np.float32)
        class_data = np.asarray(class_data,np.float32)
        
    return (sample_data,class_data)

#This class simplifies swapping the output layer of the neural network
class net_swapper(object):
    def __init__(self,do_weight_restoration,outweights1,outweigts2):
        self.outweights1 = outweights1
        self.outweights2 = outweights2
        self.mode = MODE_P1 #assume we are in mode P1 to start with
        self.do_weight_restoration = do_weight_restoration
    def swap_output_layer(self,net,mode):
        #if weight restoration is turned off then do nothing
        if(self.do_weight_restoration == False):
            return
        #if we are in mode P2 but P1 was requested, then swap weights to P1
        if(mode == MODE_P1 and self.mode == MODE_P2):
#            print('swapping to p1')
            self.outweights2 = np.copy(net.layer[-1].weights)
            net.layer[-1].weights = np.copy(self.outweights1)
            self.mode = MODE_P1
        # ... P1 but P2 was requested, then swap weights to P2
        elif(mode == MODE_P2 and self.mode == MODE_P1):
#            print('swapping to p2')
            self.outweights1 = np.copy(net.layer[-1].weights)
            net.layer[-1].weights = np.copy(self.outweights2)
            self.mode = MODE_P2
    

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
                             momentum=p['momentum2'],maxnorm=p['maxnorm2'],step_size=p['learning_rate2']))

if(p.has_key('num_hidden3')):
    layers.append(nnet.layer(p['num_hidden3'],p['activation_function3'],select_func=p['select_func3'],
                             select_func_params=p['num_selected_neurons3'],
                             initialization_scheme=p['initialization_scheme3'],
                             initialization_constant=p['initialization_constant3'],
                             dropout=p['dropout3'],sparse_penalty=p['sparse_penalty3'],
                             sparse_target=p['sparse_target3'],use_float32=p['use_float32'],
                             momentum=p['momentum3'],maxnorm=p['maxnorm3'],step_size=p['learning_rate3']))

layers.append(nnet.layer(5,p['activation_function_final'],use_float32=p['use_float32'],
                             step_size=p['learning_rate_final'],momentum=p['momentum_final']))

np.random.seed(p['random_seed']);

#init net
net = nnet.net(layers)

#f_out = open('output.txt','w')

save_interval = p['save_interval']

save_time = time.time()

#these are the variables to save
train_mse_list = [];
train_missed_list = [];
train_missed_percent_list = [];
train_nll_list = [];

test_mse1_list = [];
test_missed1_list = [];
test_missed_percent1_list = [];
test_nll1_list = [];

test_mse2_list = [];
test_missed2_list = [];
test_missed_percent2_list = [];
test_nll2_list = [];

test_mse1_p2weights_list = [];
test_missed1_p2weights_list = [];
test_missed_percent1_p2weights_list = [];
test_nll1_p2weights_list = [];

test_mse2_p1weights_list = [];
test_missed2_p1weights_list = [];
test_missed_percent2_p1weights_list = [];
test_nll2_p1weights_list = [];

training_mode_list = [];

shuffle_rate = p['shuffle_rate'];

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
outweights2 = np.copy(net.layer[-1].weights)

#get other weights from a different network
net2 = nnet.net(layers)
outweights1 = np.copy(net2.layer[-1].weights)
del net2

nll_shuffle = p['nll_shuffle']
train_nll = 1e99

nswp = net_swapper(do_weight_restoration,outweights1,outweights2)
nswp.swap_output_layer(net,MODE_P2);
nswp.swap_output_layer(net,MODE_P1);
nswp.swap_output_layer(net,MODE_P1);

training_mode = MODE_P2
for i in range(training_epochs):

    #if nll_shuffle is none then shuffle every n epochs
    do_shuffle = False
    if(nll_shuffle is None):
        if(not (i%shuffle_rate)):
            do_shuffle = True
    else:
        if(train_nll < nll_shuffle):
            do_shuffle = True
            

    if(do_shuffle or i == 0):
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
    np.random.shuffle(class_data)

    #swap the network output layer to the correct one for training mode    
    nswp.swap_output_layer(net,training_mode)
    
    #count number of correct
    train_missed = 0.0;
    train_mse = 0.0;
    train_nll = 0.0
    for j in range(minibatch_count):
        #grab a minibatch
        net.input = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        tmp = np.transpose(sample_data[j*minibatch_size:(j+1)*minibatch_size])
        classification = np.transpose(class_data[j*minibatch_size:(j+1)*minibatch_size]) 
        if(training_mode == MODE_P1):
            classification = classification[0:5,:]
        elif(training_mode == MODE_P2):
            classification = classification[5:10,:]
        net.feed_forward()
        net.error = net.output - classification
        #print('isnan net.error: ' + str(np.max(np.isnan(net.error))))
        guess = np.argmax(net.output,0)
        c = np.argmax(classification,0)
        train_nll = train_nll - np.sum(np.log(net.output)*classification)
        train_mse = train_mse + np.sum(net.error**2)
        train_missed = train_missed + np.sum(c != guess)

        net.back_propagate()
        net.update_weights()
    train_nll = float(train_nll)/float(train_size)
    train_mse = float(train_mse)/float(train_size)
    train_missed_percent = float(train_missed)/float(train_size)
    

    nswp.swap_output_layer(net,MODE_P1)    
    net.train = False
    #print('weights during test: ' + str(np.sum(net.layer[-1].weights**2.0)))
    
    #feed test set through to get test 1 rates
    net.input = np.transpose(test_data1)
    net.feed_forward()
    test_guess1 = np.argmax(net.output,0)
    c = np.argmax(test_class1,0)
    test_missed1 = np.sum(c != test_guess1)
    net.error = net.output - test_class1
    test_mse1 = np.sum(net.error**2)
    test_nll1 = -np.sum(np.log(net.output)*test_class1)
    test_nll1 = float(test_nll1)/float(test_size1)
    test_mse1 = float(test_mse1)/float(test_size1)
    test_missed_percent1 = float(test_missed1)/float(test_size1)

    net.input = np.transpose(test_data2)
    net.feed_forward()
    test_guess2 = np.argmax(net.output,0)
    c = np.argmax(test_class2,0)
    test_missed2_p1weights = np.sum(c != test_guess2)
    net.error = net.output - test_class2
    test_mse2_p1weights = np.sum(net.error**2)
    test_nll2_p1weights = -np.sum(np.log(net.output)*test_class2)
    test_nll2_p1weights = float(test_nll2_p1weights)/float(test_size2)
    test_mse2_p1weights = float(test_mse2_p1weights)/float(test_size2)
    test_missed_percent2_p1weights = float(test_missed2_p1weights)/float(test_size2)


    nswp.swap_output_layer(net,MODE_P2)
                #feed test set through to get test 2 rates
    net.input = np.transpose(test_data2)
    net.feed_forward()
    test_guess2 = np.argmax(net.output,0)
    c = np.argmax(test_class2,0)
    test_missed2 = np.sum(c != test_guess2)
    net.error = net.output - test_class2
    test_mse2 = np.sum(net.error**2)
    test_nll2 = -np.sum(np.log(net.output)*test_class2)
    test_nll2 = float(test_nll2)/float(test_size2)
    test_mse2 = float(test_mse2)/float(test_size2)
    test_missed_percent2 = float(test_missed2)/float(test_size2)

    net.input = np.transpose(test_data1)
    net.feed_forward()
    test_guess1 = np.argmax(net.output,0)
    c = np.argmax(test_class1,0)
    test_missed1_p2weights = np.sum(c != test_guess1)
    net.error = net.output - test_class1
    test_mse1_p2weights = np.sum(net.error**2)
    test_nll1_p2weights = -np.sum(np.log(net.output)*test_class1)
    test_nll1_p2weights = float(test_nll1_p2weights)/float(test_size1)
    test_mse1_p2weights = float(test_mse1_p2weights)/float(test_size1)
    test_missed_percent1_p2weights = float(test_missed1_p2weights)/float(test_size1)

    net.train = True

    #log everything for saving
    train_nll_list.append(train_nll)
    train_mse_list.append(train_mse)
    train_missed_list.append(train_missed)
    train_missed_percent_list.append(train_missed_percent)
    
    #test rate 1 and 2
    test_mse1_list.append(test_mse1)
    test_missed1_list.append(test_missed1)
    test_missed_percent1_list.append(test_missed_percent1)
    test_nll1_list.append(test_nll1)
    
    test_mse2_list.append(test_mse2)
    test_missed2_list.append(test_missed2)
    test_missed_percent2_list.append(test_missed_percent2)
    test_nll2_list.append(test_nll2)

    test_mse1_p2weights_list.append(test_mse1)
    test_missed1_p2weights_list.append(test_missed1)
    test_missed_percent1_p2weights_list.append(test_missed_percent1)
    test_nll1_p2weights_list.append(test_nll1)
    
    test_mse2_p1weights_list.append(test_mse2)
    test_missed2_p1weights_list.append(test_missed2)
    test_missed_percent2_p1weights_list.append(test_missed_percent2)
    test_nll2_p1weights_list.append(test_nll2)

    training_mode_list.append(training_mode)
#    print('epoch ' + "{: 4d}".format(i) + ": " + " mse_old: " + "{:<8.4f}".format(test_mse_old) + " acc_old: " + "{:.4f}".format(test_accuracy_old)
#    + " mse_new: " + "{:8.4f}".format(test_mse_new) + " acc_new: " + "{:.4f}".format(test_accuracy_new))

    print('Train :               epoch ' + "{: 4d}".format(i) + ": NLL: " + "{:<8.4f}".format(train_nll) + " percent missed: " + "{:<8.4f}".format(train_missed_percent))
    print('Test 1: (P1 Weights): epoch ' + "{: 4d}".format(i) + ": NLL: " + "{:<8.4f}".format(test_nll1) + " missed: " + "{: 5d}".format(test_missed1) + " percent missed: " + "{:<8.4f}".format(test_missed_percent1));
    print('Test 2: (P2 weights): epoch ' + "{: 4d}".format(i) + ": NLL: " + "{:<8.4f}".format(test_nll2) + " missed: " + "{: 5d}".format(test_missed2) + " percent missed: " + "{:<8.4f}".format(test_missed_percent2));
    print('Test 1: (P2 Weights): epoch ' + "{: 4d}".format(i) + ": NLL: " + "{:<8.4f}".format(test_nll1_p2weights) +
    " missed: " + "{: 5d}".format(test_missed1_p2weights) + " percent missed: " + "{:<8.4f}".format(test_missed_percent1_p2weights));
    print('Test 2: (P1 Weights): epoch ' + "{: 4d}".format(i) + ": NLL: " + "{:<8.4f}".format(test_nll2_p1weights) +
    " missed: " + "{: 5d}".format(test_missed2_p1weights) + " percent missed: " + "{:<8.4f}".format(test_missed_percent2_p1weights));

    #f_out.write(str(train_mse) + "," + str(train_missed_percent) + "," + str(test_missed_percent) + "\n")
    if(time.time() - save_time > save_interval or i == training_epochs-1):
        print('saving results...')
        f_handle = h5py.File(p['results_dir'] + p['simname'] + p['version'] + '.h5py','w')
        f_handle['train_mse_list'] = np.array(train_mse_list);
        f_handle['train_missed_list'] = np.array(train_missed_list);
        f_handle['train_missed_percent_list'] = np.array(train_missed_percent_list);
        
        f_handle['test_mse1_list'] = np.array(test_mse1_list);
        f_handle['test_missed1_list'] = np.array(test_missed1_list);
        f_handle['test_missed_percent1_list'] = np.array(test_missed_percent1_list);
        f_handle['test_nll1_list'] = np.array(test_nll1_list)
        
        f_handle['test_mse2_list'] = np.array(test_mse2_list);
        f_handle['test_missed2_list'] = np.array(test_missed2_list);
        f_handle['test_missed_percent2_list'] = np.array(test_missed_percent2_list);
        f_handle['test_nll2_list'] = np.array(test_nll2_list)

        f_handle['test_mse1_p2weights_list'] = np.array(test_mse1_p2weights_list);
        f_handle['test_missed1_p2weights_list'] = np.array(test_missed1_p2weights_list);
        f_handle['test_missed_percent1_p2weights_list'] = np.array(test_missed_percent1_p2weights_list);
        f_handle['test_nll1_p2weights_list'] = np.array(test_nll1_p2weights_list)
        
        f_handle['test_mse2_p1weights_list'] = np.array(test_mse2_p1weights_list);
        f_handle['test_missed2_p1weights_list'] = np.array(test_missed2_p1weights_list);
        f_handle['test_missed_percent2_p1weights_list'] = np.array(test_missed_percent2_p1weights_list);
        f_handle['test_nll2_p1weights_list'] = np.array(test_nll2_p1weights_list)

        f_handle['training_mode_list'] = np.array(training_mode_list)
   
        #iterate through all parameters and save them in the parameters group
        p_group = f_handle.create_group('parameters');
        for param in p.iteritems():
            #only save the ones that have a data type that is supported
            if(type(param[1]) in (int,float,str)):
                p_group[param[0]] = param[1];
        f_handle.close();
        save_time = time.time();
