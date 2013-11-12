#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_forget_2layer'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

incorrect_target = 0.0
correct_target = 1.0

num_hidden = 800
num_hidden2 = 300;

activation_function='sigmoid'
activation_function2='tanh'
activation_function_final='softmax'

sparse_penalty=0.01
sparse_target=0.05

sparse_penalty2=None
sparse_target2=None


#sparse_penalty=None
#sparse_target=None

minibatch_size=128

learning_rate = .01
training_epochs = 3000

forget_epoch = 1000

num_selected_neurons=100
select_func = None

num_selected_neurons2=150
select_func2 = sf.most_positive_select_func

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
