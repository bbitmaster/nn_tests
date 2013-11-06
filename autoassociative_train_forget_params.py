#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_forget_2layer'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

num_hidden = 300
num_hidden2 = 300

activation_function='tanh'
activation_function2='tanh'
activation_function3='linear_rectifier'

activation_function_final='sigmoid'

learning_rate = .01
training_epochs = 3000

forget_epoch = 1500

num_selected_neurons=100
select_func = None

num_selected_neurons2=100
select_func2 = None

num_selected_neurons3=150
select_func3 = None

dropout=0.5
dropout2=0.5
dropout3=None

initialization_scheme='glorot'
initialization_scheme2='glorot'
initialization_scheme3='glorot'

initialization_constant=1.0
initialization_constant2=1.0
initialization_constant3=1.0

random_seed = 4;

save_interval = 10*60; #save every 30 minutes
