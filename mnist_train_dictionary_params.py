#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

use_float32=True

incorrect_target = 0.0
correct_target = 1.0

num_hidden = 1000

activation_function='linear'
activation_function_final='linear'

momentum=0.0
momentum_final = 0.0

maxnorm=None

sparse_penalty=None
sparse_target=None

minibatch_size=128

#per layer learning rates
learning_rate =       0.001
learning_rate_final = 0.001

training_epochs = 3000

num_centroids = 1000

#can be normal or weighted
cluster_func = 'cluster_func'
cluster_speed = 0.0
clusters_selected = 25

dropout=None

initialization_scheme='glorot'
initialization_scheme_final='glorot'

initialization_constant=1.0
initialization_constant_final=1.0

random_seed = 4;
save_epoch = 4

