#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'autoassociative_train_forget_test'
version = '1.1'
results_dir = '../autoassociative_results/'

data_dir = '../data/'

#tells if the binary values are 0,1 (True) or -1,1 (False)
zerosandones = False

use_float32 = True

num_hidden = 2048
num_hidden2 = 2048
#num_hidden3 = 300

nodes_per_group=16
nodes_per_group2=16

activation_function='tanh'
activation_function2='tanh'
activation_function3='sigmoid'
activation_function_final='tanh'

sparse_penalty=None
sparse_target=None

sparse_penalty2=None
sparse_target2=None

sparse_penalty3=None
sparse_target3=None

sample_size = 100
num_old_samples = 100
num_new_samples = 100

learning_rate = .025
learning_rate2 = .025
learning_rate3 = .025
learning_rate_final = .025

training_epochs = 10000
batches_per_epoch=1

#shuffle_type='shuffle_rate'
#shuffle_rate=50

shuffle_type='no_improvement'
shuffle_max_epochs=300

#can be normal or weighted
#cluster_func  =  'cluster_func'
cluster_func2 =  'cluster_func'
#cluster_func3 =  'cluster_func'

var_alpha = 0.95
error_difference_threshold = 10.0

#do_cosinedistance = True

cluster_speed = 0.0
cluster_speed2 = 0.0
cluster_speed3 = 0.0
clusters_selected = 80
number_to_replace = 1000

dropout=None
dropout2=None
dropout3=None

momentum=None
momentum2=None
momentum3=None
momentum_final=None

maxnorm=None
maxnorm2=None
maxnorm3=None

initialization_scheme=None
initialization_scheme2=None
initialization_scheme3=None
#initialization_scheme='glorot'
#initialization_scheme2='glorot'
#initialization_scheme3='glorot'

initialization_constant=1.0
initialization_constant2=1.0
initialization_constant3=1.0

random_seed = 6;

save_interval = 5*60*60; #save every 30 minutes
