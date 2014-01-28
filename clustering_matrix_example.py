#!/usr/bin/env python
import numpy as np

num_inputs = 2
num_hidden = 8
num_samples = 4

alpha = 0.01
num_select = 2

X = np.random.random((num_inputs,num_samples))

W_1 = np.random.random((num_hidden,num_inputs))
C = np.random.random((num_hidden,num_inputs))

Y = np.dot(W_1,X)

count = np.ones((num_hidden,1))
for i in range(1000):
    Y_d = np.sum(C*C,1)[:,np.newaxis] - 2*np.dot(C,X) + np.sum(X*X,0)[np.newaxis,:]

    #scale them by the pseudo "starvation trace"
    Y_d = Y_d*count

    #Y_d_manual = np.zeros((num_hidden,num_samples))

    #for j in range(num_hidden):
    #    Y_d_manual[j,:] = np.sum((X.transpose() - C[j,:])**2,1)

    #select the nth smallest ones
    Y_d_sorted = np.sort(Y_d,axis=0)
    Y_s = Y_d > Y_d_sorted[num_select,:]

    #update pseudo "starvation trace"
    selected_count_this_iteration = np.sum(Y_s,axis=1)
    count = count + selected_count_this_iteration[:,np.newaxis]

    #calculate centroids for the ones that were selected
    C_prime = (np.dot(X,(~Y_s).transpose())/np.sum(~Y_s,1)).transpose()

    #note that the above matrix may contain NaNs
    #Set them to the previous centroid values
    C_prime[np.isnan(C_prime)] = C[np.isnan(C_prime)]

    #update centroids
    C = C + alpha*(C_prime - C)
