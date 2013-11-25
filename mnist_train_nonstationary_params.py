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

num_hidden = 500
num_hidden2 = 500
num_hidden3 = 500

activation_function='linear_rectifier'
activation_function2='linear_rectifier'
activation_function3='linear_rectifier'
activation_function_final='softmax'

momentum=0.5
momentum2=0.5
momentum3=0.5

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
shuffle_rate = 10

num_selected_neurons=100
select_func = None

num_selected_neurons2=100
select_func2 = None

num_selected_neurons3=150
select_func3 = None

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
