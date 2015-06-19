#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary_rerun'
version = '2.0'
results_dir = '../results_2015/'

data_dir = '../data/'

use_float32=True

incorrect_target = -1.0
correct_target = 1.0

skip_pca = True
threshold_cheat = True

P1_list='L12349'
P2_list='L56780'

num_hidden = 2048
num_hidden2 = 2048
#num_hidden3 = 2048

activation_function='tanh'
activation_function2='tanh'
activation_function3='tanh'
activation_function_final='tanh'

momentum=0.0
momentum2=0.0
momentum3=0.0
momentum_final = 0.0

maxnorm=None
maxnorm2=None
maxnorm3=None

sparse_penalty=None
sparse_target=None

sparse_penalty2=None
sparse_target2=None

sparse_penalty3=None
sparse_target3=None

minibatch_size=1024

#per layer learning rates
learning_rate =       0.005
learning_rate2 =      0.005
learning_rate3 =      0.005
learning_rate_final = 0.005

training_epochs = 3000

#shuffle_type = 'shuffle_rate'
#shuffle_rate = 300

shuffle_type = 'no_improvement'
shuffle_max_epochs = 100

var_alpha = 0.95
error_difference_threshold = 2.5

#PCA Reduction factor
reduce_to = 128

#can be normal or weighted
cluster_func  =  'cluster_func'
#cluster_func2 =  'cluster_func'
#cluster_func3 =  'cluster_func'

#do_cosinedistance = True

cluster_speed = 0.000
cluster_speed2 = 0.000
cluster_speed3 = 0.000
clusters_selected = 36
number_to_replace = 152

dropout=None
dropout2=None
dropout3=None

initialization_scheme='glorot'
initialization_scheme2='glorot'
initialization_scheme3='glorot'

initialization_constant=1.0
initialization_constant2=1.0
initialization_constant3=1.0

random_seed = 4;

save_interval = 10*60; #save every _ minutes
