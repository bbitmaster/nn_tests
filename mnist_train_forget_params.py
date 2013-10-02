#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

num_hidden = 300
learning_rate = .01
training_epochs = 3000

forget_epoch = 10

num_selected_neurons=150
select_func = None

random_seed = 4;

save_interval = 10; #save every 30 minutes