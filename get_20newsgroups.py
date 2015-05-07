#!/usr/bin/env python
import numpy as np
from nnet_toolkit import nnet
from nnet_toolkit import select_funcs as sf
from autoconvert import autoconvert
import sys
import time

#h5py used for saving results to a file
import h5py

import sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

max_features = 2000

train_data = datasets.fetch_20newsgroups(data_home="~/research/python/nn_experiments/data",subset='train',shuffle=True,remove=('headers', 'footers', 'quotes'))
test_data = datasets.fetch_20newsgroups(data_home="~/research/python/nn_experiments/data",subset='test',shuffle=True,remove=('headers', 'footers', 'quotes'))
train_data_size = len(train_data.data)
test_data_size = len(test_data.data)
data = train_data.data + test_data.data
print("vectorizing")

vectorizer = TfidfVectorizer(max_features=max_features,stop_words='english')
vectors = vectorizer.fit_transform(data)
vectors = vectors.todense() #no matrix array garbage
fnames = vectorizer.get_feature_names()

#shuffle reset training and test data
classes = np.append(train_data.target,test_data.target)
np.random.seed(1024)
rng_state = np.random.get_state();
np.random.shuffle(vectors)
np.random.set_state(rng_state)
np.random.shuffle(classes)


train_data_size = 18846 - 2048
train_vectors = vectors[0:train_data_size]
test_vectors = vectors[train_data_size:]
train_classes = classes[0:train_data_size]
test_classes = classes[train_data_size:]

f_handle = h5py.File('/home/bgoodric/research/python/nn_experiments/data/dataset_20newsgroups_'+str(max_features)+'.h5py','w')

f_handle['train_data'] = np.array(train_vectors)
f_handle['test_data'] = np.array(test_vectors)
f_handle['train_class'] = np.array(train_classes)
f_handle['test_class'] = np.array(test_classes)
f_handle['target_names'] = np.array(train_data.target_names) #tri nand test names are the same

print('train_data size: ' + str(f_handle['train_data'].shape))
print('test_data size: ' + str(f_handle['test_data'].shape))
print('train_class size: ' + str(f_handle['train_class'].shape))
print('test_class size: ' + str(f_handle['test_class'].shape))
f_handle.close()
