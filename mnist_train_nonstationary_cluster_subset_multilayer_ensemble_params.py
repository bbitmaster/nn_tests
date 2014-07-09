#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'mnist_train_nonstationary_cluster_subseti_multilayer_ensemble'
version = '1.1'
results_dir = '../results/'

data_dir = '../data/'

use_float32=True

incorrect_target = -1.0
correct_target = 1.0

skip_pca = True
threshold_cheat = True

P1_list='L12349'
P2_list='L56780'

num_hidden = [128,128,128,128,128]
num_hidden2 = [64,64,64,64,64]
num_hidden3 = [32,32,32,32,32]

nodes_per_group = [None,None,None,None,None]
nodes_per_group2 = [None,None,None,None,None]
nodes_per_group3 = [None,None,None,None,None]

activation_function=['tanh','tanh','tanh','tanh','tanh']
activation_function2=['tanh','tanh','tanh','tanh','tanh']
activation_function3=['tanh','tanh','tanh','tanh','tanh']
activation_function_final=['tanh','tanh','tanh','tanh','tanh']

momentum=[0.0,0.0,0.0,0.0,0.0]
momentum2=[0.0,0.0,0.0,0.0,0.0]
momentum3=[0.0,0.0,0.0,0.0,0.0]
momentum_final = [0.0,0.0,0.0,0.0,0.0]

maxnorm=[None,None,None,None,None]
maxnorm2=[None,None,None,None,None]
maxnorm3=[None,None,None,None,None]

sparse_penalty=[None,None,None,None,None]
sparse_target=[None,None,None,None,None]

sparse_penalty2=[None,None,None,None,None]
sparse_target2=[None,None,None,None,None]

sparse_penalty3=[None,None,None,None,None]
sparse_target3=[None,None,None,None,None]

minibatch_size=1024

#per layer learning rates
learning_rate =       [0.005,0.005,0.005,0.005,0.005]
learning_rate2 =      [0.005,0.005,0.005,0.005,0.005]
learning_rate3 =      [0.005,0.005,0.005,0.005,0.005]
learning_rate_final = [0.005,0.005,0.005,0.005,0.005]

training_epochs = 3000

#shuffle_type = 'shuffle_rate'
#shuffle_rate = 10

shuffle_type = 'no_improvement'
shuffle_max_epochs = 100

var_alpha = 0.95
error_difference_threshold = 2.5

#PCA Reduction factor
reduce_to = 128

#can be normal or weighted
cluster_func  =  ['cluster_func','cluster_func','cluster_func','cluster_func','cluster_func']
cluster_func2 =  ['cluster_func','cluster_func','cluster_func','cluster_func','cluster_func']
cluster_func3 =  [None,None,None,None,None]

do_cosinedistance = [False,False,False,False,False]

cluster_speed = [0.0,0.0,0.0,0.0,0.0]
cluster_speed2 = [0.0,0.0,0.0,0.0,0.0]
cluster_speed3 = [0.0,0.0,0.0,0.0,0.0]
clusters_selected = [16,16,16,16,16]
clusters_selected2 = [16,16,16,16,16]
clusters_selected3 = [16,16,16,16,16]
number_to_replace = [32,32,32,32,32]

dropout=[None,None,None,None,None]
dropout2=[None,None,None,None,None]
dropout3=[None,None,None,None,None]

initialization_scheme=['glorot','glorot','glorot','glorot','glorot']
initialization_scheme2=['glorot','glorot','glorot','glorot','glorot']
initialization_scheme3=['glorot','glorot','glorot','glorot','glorot']

initialization_constant=[1.0,1.0,1.0,1.0,1.0]
initialization_constant2=[1.0,1.0,1.0,1.0,1.0]
initialization_constant3=[1.0,1.0,1.0,1.0,1.0]

random_seed = 4;

save_interval = 60; #save every _ minutes
