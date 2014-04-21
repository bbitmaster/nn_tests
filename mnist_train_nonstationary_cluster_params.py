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
num_hidden2 = 1000
num_hidden3 = 1000

nodes_per_group=10
nodes_per_group2=10
nodes_per_group3=10

activation_function='maxout'
activation_function2='maxout'
activation_function3='maxout'
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

minibatch_size=128

#per layer learning rates
learning_rate =       0.001
learning_rate2 =      0.001
learning_rate3 =      0.001
learning_rate_final = 0.001

training_epochs = 3000

do_weight_restoration = False
fresh_value_weights = True
shuffle_rate = 500
nll_shuffle = None

#can be normal or weighted
cluster_func = None
cluster_func2 = None
cluster_func3 = None

cluster_speed = 0.001
clusters_selected = 100

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

save_interval = 10*60; #save every 30 minutes
