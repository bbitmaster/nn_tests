
#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary_rerun'
version = '2.0'
results_dir = '../gas_results_2015/'

data_dir = '../data/'

use_float32=True

incorrect_target = 0.0
correct_target = 1.0

skip_pca = False
threshold_cheat = True

num_samples = 100000

random_variance=2.0

num_hidden = 2048
num_hidden2 = 2048
#num_hidden3 = 2048

nodes_per_group=8

activation_function='tanh'
activation_function2='tanh'
activation_function3='tanh'
activation_function_final='softmax'

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
learning_rate =       0.001
learning_rate2 =      0.001
learning_rate3 =      0.001
learning_rate_final = 0.001

training_epochs = 3000

#shuffle_type = 'shuffle_rate'
#shuffle_rate = 300

shuffle_type = 'no_improvement'
shuffle_max_epochs = 10

var_alpha = 0.95
error_difference_threshold = 2.5

#PCA Reduction factor
reduce_to = 75
scale = 6

#can be normal or weighted
cluster_func  =  None
cluster_func2 =  None
#cluster_func3 =  'cluster_func'

#do_cosinedistance = True

cluster_speed = 0.000
cluster_speed2 = 0.000
cluster_speed3 = 0.000
clusters_selected = 200
number_to_replace = 300

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

activation_function = 'tanh'
activation_function2 = 'tanh'
activation_function3 = 'tanh'
activation_function_final = 'softmax'
cluster_func = 'cluster_func'
cluster_layer = 1
cluster_speed = 0.0
cluster_speed2 = 0.0
cluster_speed3 = 0.0
clusters_selected = 36.0
correct_target = 1.0
data_dir = '../data/'
error_difference_threshold = 2.5
incorrect_target = 0.0
initialization_constant = 1.0
initialization_constant2 = 1.0
initialization_constant3 = 1.0
initialization_scheme = 'glorot'
initialization_scheme2 = 'glorot'
initialization_scheme3 = 'glorot'
learning_rate = 0.02
learning_rate2 = 0.02
learning_rate3 = 0.001
learning_rate_final = 0.02
minibatch_size = 1024
momentum = 0.0
momentum2 = 0.0
momentum3 = 0.0
momentum_final = 0.0
nodes_per_group = 8
num_hidden = 512.0
num_hidden2 = 512.0
num_samples = 100000
number_to_replace = 280.0
random_seed = 4
random_variance = 1.0
reduce_to = 75
results_dir = '../gas_results_2015_lessnoise/'
save_interval = 600
scale = 6
shuffle_max_epochs = 30
shuffle_type = 'no_improvement'
training_epochs = 10000
var_alpha = 0.95
simname = 'gas_cluster_run_lownoise_GXZNKDYHEZWE'
version = 'nocov'
