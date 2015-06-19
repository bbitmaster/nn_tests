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
    #images /= 255.0 
    #normalize between -1 and 1 for hyperbolic tangent
    #images = images - 0.5;
    #images = images*2.0; 
    
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
    if(p.has_key('dataset_name') and p['dataset_name'] == 'newsgroups'):
        (sample_data,class_data) = load_newsgroup_data(range(10),"training",p)
        (test_data,test_class) = load_newsgroup_data(range(10),"testing",p)
    else:
        (sample_data,class_data) = load_mnist_data(range(10),"training",p)
        (test_data,test_class) = load_mnist_data(range(10),"testing",p)

train_size = sample_data.shape[0]
input_size = sample_data.shape[1]

class_data_index = np.argmax(class_data,axis=1)
test_class_index = np.argmax(test_class,axis=1)

#logistic_regression time
logreg = linear_model.LogisticRegression()

print("Doing Logistic Regression...")

limit=train_size
if(p.has_key('limit')):
    limit = p['limit']

sample_data = sample_data[0:limit,:]
class_data_index = class_data_index[0:limit]
logtime = time.time()
logreg.fit(sample_data,class_data_index)
print("time taken: " + str(time.time() - logtime))

print("Predicting")
logtime = time.time()
test_class_prediction = logreg.predict(test_data)
miss_class = test_class_prediction != test_class_index
miss_class_percentage = float(np.sum(miss_class))/float(miss_class.shape[0])
print("missclass percentage: " + str(miss_class_percentage))
print("time taken: " + str(time.time() - logtime))
