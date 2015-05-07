#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_rmstest'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

use_float32=True

incorrect_target = 0.0
correct_target = 1.0

num_hidden = 1000
num_hidden2 = 1000
num_hidden3 = 1000

activation_function='linear_rectifier'
activation_function2='linear_rectifier'
activation_function3='linear_rectifier'
activation_function_final='softmax'

rms_prop_rate=None

momentum=None
momentum2=None
momentum3=None

maxnorm=2.0
maxnorm2=2.0
maxnorm3=2.0

sparse_penalty=None
sparse_target=None

sparse_penalty2=None
sparse_target2=None

sparse_penalty3=None
sparse_target3=None

minibatch_size=128

#per layer learning rates
learning_rate =       0.1
learning_rate2 =      0.1
learning_rate3 =      0.1
learning_rate_final = 0.1

training_epochs = 3000

forget_epoch = 1000

dropout=None
dropout2=None
dropout3=None

select_func=None
select_func2=None
select_func3=None

initialization_scheme='glorot'
initialization_scheme2='glorot'
initialization_scheme3='glorot'

initialization_constant=1.0
initialization_constant2=1.0
initialization_constant3=1.0

random_seed = 4;

save_interval = 10*60; #save every 30 minutes
