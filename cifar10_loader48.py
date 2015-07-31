import cPickle
import gzip
import numpy as np

import sys
import h5py as h5


data_dir = '../data/cifar10_48/'

train_fname = 'cifar_train48.mat'
test_fname = 'cifar_test48.mat'
train_label_fname = 'train.txt'
test_label_fname = 'val.txt'

def load_batch(batch_fname,label_fname):
    d = {}
    f = h5.File(batch_fname,'r')
    d['data'] = np.array(f['binary_codes']).astype(np.float32)
    f.close()

    f = open(label_fname,'r')
    l = f.readlines()
    l = [a.split() for a in l]
    l = [int(a[1]) for a in l]
    l = np.array(l)
    d['labels'] = l
    return d


def load_cifar10_48(correct_target,incorrect_target):

    
    d = load_batch(data_dir + train_fname,data_dir + train_label_fname)

    full_data = d['data']
    full_class = d['labels']

    d = load_batch(data_dir + test_fname,data_dir + test_label_fname)

    test_data = d['data']
    test_class = d['labels']

    #change class labels to 1 hot encoding
    full_class_onehot = np.ones((full_class.shape[0],10))*incorrect_target
    for i in range(full_class.shape[0]):
        full_class_onehot[i,full_class[i]] = correct_target;

    test_class_onehot = np.ones((test_class.shape[0],10))*incorrect_target
    for i in range(test_class.shape[0]):
        test_class_onehot[i,test_class[i]] = correct_target;


    test_class_onehot = test_class_onehot.astype(np.float32)
    full_class_onehot = full_class_onehot.astype(np.float32)

    return (full_data,full_class_onehot,test_data,test_class_onehot)

if __name__ == '__main__':

    d = load_cifar10_48(1.0,0.0)
    print(d[0])
    print(d[1])
    print(d[2])
    print(d[3])
#    print(np.sum(face_class,axis=0))

