#mnist_train_params
from nnet_toolkit import select_funcs as sf;
import numpy as np;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

use_float32=True

incorrect_target = -1.0
correct_target = 1.0

#num_classes = 2;

#center_x_list1 = [0.7,  0.7]
#center_y_list1 = [0.7, -0.7]

#center_x_list2 = [-0.7, -0.7]
#center_y_list2 = [-0.7,  0.7]

num_classes = 6;

skew_angle = 180./num_classes/2.

angles1 = np.array(range(0,6))/12.0 * 360 + skew_angle

center_x_list1 = np.sin(angles1*np.pi/180.)*0.7
center_y_list1 = np.cos(angles1*np.pi/180.)*0.7

angles2 = (np.array(range(0,6)) + 6.)/12.0 * 360 + skew_angle
center_x_list2 = np.sin(angles2*np.pi/180.)*0.7
center_y_list2 = np.cos(angles2*np.pi/180.)*0.7

examples_per_class = 30
spread = 0.05
dump_to_file = False
dump_path = "/local_scratch/clustersimpletest1/"
frameskip = 20;

img_width = 720
img_height = 360

axis_x_min = -1.5
axis_y_min = -1.5

axis_x_max = 1.5
axis_y_max = 1.5

num_hidden = 256

activation_function='tanh'
activation_function_final='tanh'

momentum=0.0
momentum_final = 0.0

maxnorm=None

sparse_penalty=None
sparse_target=None

#per layer learning rates
learning_rate =       0.01
learning_rate_final = 0.01

forget_epochs = 200
training_epochs = 3000
total_epochs = 3000

#can be normal or weighted
#cluster_func = 'cluster_func'
cluster_func = 'cluster_func'
cluster_speed = 0.00
clusters_selected = 8

dropout=None

random_seed = 4;

save_interval = 10*60; #save every 30 minutes
