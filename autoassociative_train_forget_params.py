#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'autoassociative_train_forget'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

#tells if the binary values are 0,1 (True) or -1,1 (False)
zerosandones = False

num_hidden = 300
num_hidden2 = 300
num_hidden3 = 300

activation_function='tanh'
activation_function2='tanh'
activation_function3='tanh'
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
minibatch_size=10

learning_rate = .001
training_epochs = 3000

forget_epoch = 1000

num_selected_neurons=50
select_func = None

num_selected_neurons2=50
select_func2 = None

num_selected_neurons3=50
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

random_seed = 5;

save_interval = 10*60; #save every 30 minutes
