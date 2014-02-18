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

def get_centers_from_angles(num_classes):
    skew_angle = 180./num_classes/2.

    angles1 = np.array(range(0,6))/12.0 * 360 + skew_angle

    center_x_list1 = np.sin(angles1*np.pi/180.)*0.7
    center_y_list1 = np.cos(angles1*np.pi/180.)*0.7

    angles2 = (np.array(range(0,6)) + 6.)/12.0 * 360 + skew_angle
    center_x_list2 = np.sin(angles2*np.pi/180.)*0.7
    center_y_list2 = np.cos(angles2*np.pi/180.)*0.7
    return (center_x_list1,center_y_list1,center_x_list2,center_y_list2)


def gen_random_centers(num_classes,num_tries,threshold):
    center_x_list = []
    center_y_list = []
    for i in range(num_classes*2):
        j = 0
        while j < num_tries:
            #init between -0.7 and 0.7
            (center_x_tmp,center_y_tmp) = (np.random.random(2)*2.0 - 1.0)*0.7
            dist = ((np.array(center_x_list) - center_x_tmp)**2 + (np.array(center_y_list) - center_y_tmp)**2)
            #print(str(dist))
            #print(str(dist > threshold**2))
            if(np.sum(dist < threshold**2) < 1):
                break
            j = j + 1
        print('j ' + str(j) + ' ' + str(center_x_tmp) + ' ' + str(center_y_tmp))
        center_x_list.append(center_x_tmp)
        center_y_list.append(center_y_tmp)
    center_x_list1 = center_x_list[0:num_classes]
    center_x_list2 = center_x_list[num_classes:(num_classes*2)]
    center_y_list1 = center_y_list[0:num_classes]
    center_y_list2 = center_y_list[num_classes:(num_classes*2)]

    #print(str(len(center_x_list)) + ' ' + str(len(center_x_list1)) + ' ' + str(len(center_x_list2)))
        
    return (center_x_list1,center_y_list1,center_x_list2,center_y_list2)
#try to make clusters that are spread apart
(center_x_list1,center_y_list1,center_x_list2,center_y_list2) = gen_random_centers(num_classes,1000,0.35);

examples_per_class = 30
spread = 0.05
dump_to_file = True
dump_path = "/local_scratch/clustersimpletest_6classes_nomove/"
frameskip = 1;

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
