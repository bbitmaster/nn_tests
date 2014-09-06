#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'autoassociative_train_forget_test'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

#tells if the binary values are 0,1 (True) or -1,1 (False)
zerosandones = False

use_float32 = True

num_hidden = 1024
num_hidden2 = 1024
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

learning_rate = .1
learning_rate2 = .1
learning_rate3 = .1
learning_rate_final = .1

training_epochs = 100
batches_per_epoch=10

shuffle_type='shuffle_rate'
shuffle_rate=50

#shuffle_type='no_improvement'
#shuffle_max_epochs=100

#can be normal or weighted
cluster_func  =  'cluster_func'
#cluster_func2 =  'cluster_func'
#cluster_func3 =  'cluster_func'

var_alpha = 0.95
error_difference_threshold = 10.0

#do_cosinedistance = True

cluster_speed = 0.00001
cluster_speed2 = 0.00001
cluster_speed3 = 0.00001
clusters_selected = 100
number_to_replace = 500

dropout=None

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

initialization_scheme='glorot'
initialization_scheme2='glorot'
initialization_scheme3='glorot'

initialization_constant=1.0
initialization_constant2=1.0
initialization_constant3=1.0

random_seed = 6;

save_interval = 10*60; #save every 30 minutes
