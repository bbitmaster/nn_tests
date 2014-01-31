#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

use_float32=True

incorrect_target = -1.0
correct_target = 1.0

examples_per_class = 30
spread = 0.2
dump_to_file = True
dump_path = "/local_scratch/clustersimpletest1/"
frameskip = 25;

img_width = 720
img_height = 360

axis_x_min = -1.5
axis_y_min = -1.5

axis_x_max = 1.5
axis_y_max = 1.5

num_hidden = 64

activation_function='tanh'
activation_function_final='tanh'

momentum=0.0
momentum_final = 0.0

maxnorm=None

sparse_penalty=None
sparse_target=None

#per layer learning rates
learning_rate =       0.001
learning_rate_final = 0.001

forget_epochs = 1000
training_epochs = 3000
total_epochs = 3000

#can be normal or weighted
#cluster_func = 'cluster_func'
cluster_func = None
cluster_speed = 0.05
clusters_selected = 4

dropout=None

random_seed = 4;

save_interval = 10*60; #save every 30 minutes
