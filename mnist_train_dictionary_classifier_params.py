#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary_classifier_tmp'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

load_centroids = False
do_cosinedistance = True
save_interval = 30*60

use_float32=True

incorrect_target=-1.0
correct_target=1.0

activation_function='tanh'
activation_function_final='tanh'

momentum=0.0
momentum_final = 0.0

maxnorm=None

sparse_penalty=None
sparse_target=None

minibatch_size=128

#per layer learning rates
learning_rate =       0.01
learning_rate_final = 0.01

training_epochs = 30000
end_type = 'no_improvement'
num_epochs = 100

num_centroids = 16

#can be normal or weighted
cluster_func = 'cluster_func'
cluster_speed = 0.0
clusters_selected = 16

dropout=None

initialization_scheme='glorot'
initialization_scheme_final='glorot'

initialization_constant=1.0
initialization_constant_final=1.0

random_seed = 4;

