#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'newsgroup_train_nonstationary_maxout_subset_multilayer'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

use_float32=True

incorrect_target = -1.0
correct_target = 1.0

skip_pca = True
threshold_cheat = True

P1_list='L0L1L2L3L4L5L6L7L8L9'
P2_list='L10L11L12L13L14L15L16L17L18L19'

num_hidden = 64
num_hidden2 = 64
#num_hidden3 = 128

#nodes_per_group=64
#nodes_per_group2=64

activation_function='tanh'
activation_function2='tanh'
#activation_function3='linear_rectifier'
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
learning_rate =       0.01
learning_rate2 =      0.01
learning_rate3 =      0.01
learning_rate_final = 0.01

training_epochs = 3000

shuffle_type = 'no_improvement'
#shuffle_rate = 30000
#shuffle_missed_percent = 0.35
shuffle_max_epochs = 100

var_alpha = 0.95
error_difference_threshold = 2.5

#PCA Reduction factor
reduce_to = 256

#can be normal or weighted
#cluster_func  =  'cluster_func'
#cluster_func2 =  'cluster_func'
#cluster_func3 =  'cluster_func'

#do_cosinedistance = True

cluster_speed = 0.00001
cluster_speed2 = 0.00001
cluster_speed3 = 0.00001
clusters_selected = 64
number_to_replace = 100

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

save_interval = 60*60; #save every _ minutes
