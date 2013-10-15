#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

num_hidden = 300
num_hidden2 = 300; #use -1 for no 2nd hidden layer

activation_function='tanh'
activation_function2='tanh'

learning_rate = .01
training_epochs = 3000

forget_epoch = 1000

num_selected_neurons=100
select_func = sf.most_negative_select_func

num_selected_neurons2=100
select_func2 = sf.most_negative_select_func

random_seed = 4;

save_interval = 10; #save every 30 minutes