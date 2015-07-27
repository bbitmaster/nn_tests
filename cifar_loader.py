import cPickle
import gzip
import numpy as np

import sys


data_dir = '../data/cifar-10-batches-py-compressed/'

batch_fname_list = [
'data_batch_1.gz',
'data_batch_2.gz',
'data_batch_3.gz',
'data_batch_4.gz',
'data_batch_5.gz',
]

def load_batch(batch_fname):
    cifar_f = gzip.open(batch_fname)
    d = cPickle.load(cifar_f)
    cifar_f.close()
    return d


def load_cifar10(correct_target,incorrect_target):
    #5 batches
    full_data = []
    full_class = []
    for fname in batch_fname_list:
        file_path = data_dir + fname
        d = load_batch(file_path)
        full_data.extend(d['data'])
        full_class.extend(d['labels'])

    full_data = np.array(full_data)
    full_class = np.array(full_class)

    file_path = data_dir + fname
    d = load_batch(file_path)
    test_data = np.copy(np.array(d['data']))
    test_class = np.copy(np.array(d['labels']))

    #change class labels to 1 hot encoding
    full_class_onehot = np.ones((full_class.shape[0],10))*incorrect_target
    for i in range(full_class.shape[0]):
        full_class_onehot[i,full_class[i]] = correct_target;

    test_class_onehot = np.ones((test_class.shape[0],10))*incorrect_target
    for i in range(test_class.shape[0]):
        test_class_onehot[i,test_class[i]] = correct_target;

    #normalize data
    data_mean = np.mean(full_data,axis=0)
    data_std = np.std(full_data,axis=0)

    full_data = full_data - data_mean
    full_data = full_data/data_std

    test_data = test_data - data_mean
    test_data = test_data/data_std

    #convert to float32
    full_data = full_data.astype(np.float32)
    test_data = test_data.astype(np.float32)


    test_class_onehot = test_class_onehot.astype(np.float32)

    return (full_data,full_class_onehot,test_data,test_class_onehot)

if __name__ == '__main__':

    d = load_cifar10(1.0,0.0)
    print(d[0])
    print(d[1])
    print(d[2])
    print(d[3])
#    print(np.sum(face_class,axis=0))

