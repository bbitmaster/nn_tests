#!/usr/bin/env python

import sys

from nnet_toolkit import nnet

import numpy as np
import scipy.stats
import matplotlib.cm as cm
import image_plotter
import cluster_select_func as csf
from autoconvert import autoconvert

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
        params_file = sys.argv[1]
else:
        params_file = 'cluster_simple_test_params.py'

p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    p[k] = v
    print(str(k) + ":" + str(v))
np.random.seed(p['random_seed'])

examples_per_class = p['examples_per_class'];
spread = p['spread']

dump_path = p['dump_path']
dump_to_file = p['dump_to_file']
frameskip = p['frameskip']
img_width = p['img_width'];
img_height = p['img_height'];

#the axis for the view
vx_axis = [p['axis_x_min'],p['axis_x_max']];
vy_axis = [p['axis_y_min'],p['axis_y_max']];

num_classes = p['num_classes']
num_hidden = p['num_hidden']

training_epochs = p['training_epochs']
total_epochs = p['total_epochs']

#minibatch_size = p['minibatch_size']

layers = [];
layers.append(nnet.layer(2))
layers.append(nnet.layer(p['num_hidden'],p['activation_function'],
              dropout=p['dropout'],sparse_penalty=p['sparse_penalty'],
              sparse_target=p['sparse_target'],use_float32=p['use_float32'],
              momentum=p['momentum'],maxnorm=p['maxnorm'],step_size=p['learning_rate']))

layers.append(nnet.layer(num_classes,p['activation_function_final'],use_float32=p['use_float32'],
              momentum=p['momentum_final'],step_size=p['learning_rate_final']))

#init net
net = nnet.net(layers)

if(p.has_key('cluster_func') and p['cluster_func'] is not None):
    #net.layer[0].centroids = np.asarray((((np.random.random((net.layer[0].weights.shape)) - 0.5)*2.0)),np.float32)
    net.layer[0].centroids = np.asarray(np.zeros(net.layer[0].weights.shape),np.float32)
#set bias to 1
    net.layer[0].centroids[:,-1] = 1.0
    net.layer[0].centroids[:,-1] = -10000.0
    net.layer[0].select_func = csf.select_names[p['cluster_func']]
    net.layer[0].centroid_speed = p['cluster_speed']
    net.layer[0].num_selected = p['clusters_selected']
    net.layer[0].do_weighted_euclidean = True

#Generate Random Classes
sample_data1 = np.zeros([2,num_classes*examples_per_class])
class_data1 = np.ones([num_classes,num_classes*examples_per_class])*p['incorrect_target']

#center_x_list1 = [0.7,0.7]
#center_y_list1 = [0.7,-0.7]

center_x_list1 = p['center_x_list1']
center_y_list1 = p['center_y_list1']

for i in range(num_classes):
    center_x = center_x_list1[i];
    center_y = center_y_list1[i];
    c = np.random.randn(2,examples_per_class)*spread
    c[0,:] += center_x;
    c[1,:] += center_y
    sample_data1[0:2,i*examples_per_class:(i+1)*examples_per_class] = c 
    class_data1[i,i*examples_per_class:(i+1)*examples_per_class] = 1.0

sample_data2 = np.zeros([2,num_classes*examples_per_class])
class_data2 = np.ones([num_classes,num_classes*examples_per_class])*p['incorrect_target']

#center_x_list2 = [-0.7,-0.7]
#center_y_list2 = [-0.7,0.7]

center_x_list2 = p['center_x_list2']
center_y_list2 = p['center_y_list2']

for i in range(num_classes):
    center_x = center_x_list2[i];
    center_y = center_y_list2[i];
    c = np.random.randn(2,examples_per_class)*spread
    c[0,:] += center_x;
    c[1,:] += center_y
    sample_data2[0:2,i*examples_per_class:(i+1)*examples_per_class] = c 
    class_data2[i,i*examples_per_class:(i+1)*examples_per_class] = 1.0

#build a pallete
pal = cm.rainbow(np.linspace(0,1,num_classes));
pal = pal[0:num_classes,0:3]

plt = image_plotter.image_plotter();
plt.initImg(img_width,img_height);
plt.setAxis(vx_axis[0],vx_axis[1],vy_axis[0],vy_axis[1])
c=pal[np.argmax(class_data1,0),:]

if(not dump_to_file):
    plt.show();



epoch = 1;

#init variables for estimating label mean and standard deviation
mean_alpha = 0.9
var_alpha = 0.9

sample_data_tmp = np.zeros((num_classes,2,30),dtype=np.float32)
sample_data_mean_tmp = np.zeros((num_classes,2),dtype=np.float32)
sample_data_var_tmp = np.zeros((num_classes,2),dtype=np.float32)
label_mean = np.zeros((2,num_classes),dtype=np.float32)
label_var = np.ones((2,num_classes),dtype=np.float32)
membership = np.ones(num_classes,dtype=np.float32)

while(epoch < p['total_epochs']):

    if(epoch < p['forget_epochs']):
        train_sample_data = sample_data1
        train_class_data = class_data1
        train_mode = 0
    else:
        train_sample_data = sample_data2
        train_class_data = class_data2
        train_mode = 1
 
    net.input = train_sample_data;
    net.feed_forward()
    
#    np.savetxt("dmp/distances_epoch" + str(epoch) + ".csv",net.layer[0].distances,delimiter=",");
#    asdf
#    print("Neuron 1 Centroid 11")
#    print(str(self.layer[0].input[:,1]);
#    print(str(self.layer[0].input[:,1]);
    for l in range(num_classes):
        mask = np.equal(l,np.argmax(train_class_data,0))
        sample_data_tmp[l] = train_sample_data[:,mask]
        sample_data_mean_tmp[l] = np.mean(sample_data_tmp[l],1)
        sample_data_var_tmp[l] = np.var(sample_data_tmp[l],1)
#compute membership
#note: this assumes imput dimension of 2-- need to change for larger inputs
        membership[l] = scipy.stats.norm.pdf(sample_data_mean_tmp[l,0],label_mean[0,l],np.sqrt(label_var[0,l]))
        membership[l] *= scipy.stats.norm.pdf(sample_data_mean_tmp[l,1],label_mean[1,l],np.sqrt(label_var[1,l]))
        label_mean[:,l] = mean_alpha*label_mean[:,l] + (1.0 - mean_alpha)*sample_data_mean_tmp[l]
        label_var[:,l] = var_alpha*label_var[:,l] + (1.0 - var_alpha)*sample_data_var_tmp[l]
        print("label: " + str(l) + " mean: " + str(label_mean[:,l]) + " var: " + str(label_var[:,l]) + " membership: " + str(membership[l]))


    number_to_replace = 16
    neuron_used_indices = net.layer[0].selected_count.argsort()
    for l in range(6):
        if(membership[l] < .2):
            print("MEMBERSHIP EXCEEDED THRESHOLD FOR LABEL " + str(l))
            #get the 8 least selected neurons
            replace_indices = neuron_used_indices[0:number_to_replace]
            neuron_used_indices = neuron_used_indices[number_to_replace:]
            #need 8 sample data points -- could use k-means -- for now sample randomly
            #samples is S x N where S is number os samples, and N is input size
            samples = sample_data_tmp[l][:,0:number_to_replace]

            #need to tack on the bias
            samples = np.append(samples,np.ones((1,samples.shape[1]),dtype=samples.dtype),axis=0)
            
            #replace centroids with new ones drawn from samples
            net.layer[0].centroids[replace_indices,:] = samples.transpose()
            net.layer[0].centroids[replace_indices,:] = net.layer[0].centroids[replace_indices,:]*net.layer[0].weights[replace_indices,:]

            #reset centroid mean
            label_mean[:,l] = sample_data_mean_tmp[l]
            label_var[:,l] = sample_data_var_tmp[l]
    print(net.layer[0].weights[replace_indices,:])



    net.error = net.output - train_class_data
    neterror = net.error
    net_classes = net.output

    net.back_propagate()
    net.update_weights()
    if(p['cluster_func'] is not None):
        csf.update_names[p['cluster_func']](net.layer[0])
    #get class 1 error rate
    net.input = sample_data1;
    net.feed_forward()
    net.error = net.output - class_data1
    neterror1 = net.error
    net_classes1 = net.output
    num_correct1 = sum(np.equal(np.argmax(net_classes1,0),np.argmax(class_data1,0)))
    percent_correct1 = num_correct1/float(num_classes*examples_per_class)
    percent_miss1 = 1.0 - percent_correct1;

    #get class 2 error rate
    net.input = sample_data2;
    net.feed_forward()
    net.error = net.output - class_data2
    neterror2 = net.error
    net_classes2 = net.output
    num_correct2 = sum(np.equal(np.argmax(net_classes2,0),np.argmax(class_data2,0)))
    percent_correct2 = num_correct2/float(num_classes*examples_per_class)
    percent_miss2 = 1.0 - percent_correct2;

    if((dump_to_file or epoch%frameskip == 0)):
        xv, yv = np.meshgrid(np.linspace(vx_axis[0],vx_axis[1],img_width),np.linspace(vy_axis[0],vy_axis[1],img_height))
        xv = np.reshape((xv),(img_height*img_width))
        yv = np.reshape((yv),(img_height*img_width))
        net.input = np.vstack((xv,yv))
        net.feed_forward()
        img_data=pal[np.argmax(net.output,0),:]
        img_data = img_data.reshape((img_height,img_width,3))
        plt.setImg(img_data)

        #class 1
        #draw incorrect as black, correct as white
        correct_pal1 = np.array([[0,0,0],[1,1,1]])
        correct = np.equal(np.argmax(net_classes1,0),np.argmax(class_data1,0))
        c = correct_pal1[np.int32(correct),:]
        plt.drawPoint(sample_data1[0,:],sample_data1[1,:],size=1,color=c)

        #class 2
        correct_pal2 = np.array([[0.1,0.1,0.0],[1.0,1.0,0.9]])
        correct = np.equal(np.argmax(net_classes2,0),np.argmax(class_data2,0))
        c = correct_pal2[np.int32(correct),:]
        plt.drawPoint(sample_data2[0,:],sample_data2[1,:],size=1,color=c)

        #draw centroids
        if(p['cluster_func'] is not None):
            plt.drawPoint(net.layer[0].centroids[:,0],net.layer[0].centroids[:,1],size=2,color=(1.0,1.0,0.8))
        
        #draw dot-product = 0 lines
        x1 = np.zeros(num_hidden);
        y1 = np.zeros(num_hidden);
        x2 = np.zeros(num_hidden);
        y2 = np.zeros(num_hidden);

        for i in range(num_hidden):
            #get m and b for y = mx + b
            m = -net.layer[0].weights[i,0]/net.layer[0].weights[i,1];
            b = net.layer[0].weights[i,2]/net.layer[0].weights[i,1]
            #if slope is large, compute x = (y - b)/m
            if(np.abs(m) > 1):
                y1[i] = vy_axis[0];
                x1[i] = (vy_axis[0] - b)/m
                y2[i] = vy_axis[1];
                x2[i] = (vy_axis[1] - b)/m
            else:
                x1[i] = vx_axis[0];
                y1[i] = m*vx_axis[0] + b;
                x2[i] = vx_axis[1];
                y2[i] = m*vx_axis[1] + b;
#        plt.drawLine(x1,x2,y1,y2,color=(0,0,0))
        plt.drawRect(0,620,0,35,color=(1,1,1),use_image_coords=True)
        plt.drawText(1,1,"epoch: " + str(epoch),color=(0,0,0),use_image_coords=True)
        plt.drawText(120,1,"P1 Miss percent: " + str(percent_miss1),color=(0,0,0),use_image_coords=True)
        plt.drawText(420,1,"P1 MSE: " + str(np.sum(neterror1**2)),color=(0,0,0),use_image_coords=True)
        plt.drawText(120,18,"P2 Miss percent: " + str(percent_miss2),color=(0,0,0),use_image_coords=True)
        plt.drawText(420,18,"P2 MSE: " + str(np.sum(neterror2**2)),color=(0,0,0),use_image_coords=True)

        if(train_mode == 0):
            plt.drawText(1,18,"P1 Training ",color=(0.5,0.5,0.0),use_image_coords=True)
        else:
            plt.drawText(1,18,"P2 Training",color=(0.5,0.5,0.0),use_image_coords=True)
        
        #plt.drawRect(100,190,340,360,color=(1,1,1),use_image_coords=True)
        #plt.drawText(100,341,"P2 Data",color=(0.0,0.0,0.0),use_image_coords=True)
        
        #plt.drawRect(450,540,340,360,color=(1,1,1),use_image_coords=True)
        #plt.drawText(450,341,"P1 Data",color=(0.0,0.0,0.0),use_image_coords=True)
    print("epoch: " + str(epoch) + 
		  " P1 percent: " + str(percent_miss1) + " P1 MSE: " + str(np.sum(neterror1**2)) +
		  " P2 percent: " + str(percent_miss2) + " P2 MSE: " + str(np.sum(neterror2**2)))
    if(dump_to_file):
        print("saving: " + dump_path + "nn_dump" + str(epoch))
        plt.save_plot(dump_path + "nn_dump" + str(epoch) + ".png")
    else:
        if(epoch%frameskip == 0):
            plt.update()
    keypress = plt.processEvents();
    if(keypress == 27):
        break
    epoch = epoch + 1;



