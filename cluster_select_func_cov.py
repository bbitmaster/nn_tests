import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist

def cluster_select_func(self):
    num_selected = self.num_selected
    if(hasattr(self,'do_weighted_euclidean')):
        self.distances = np.sum(self.centroids**2,1)[:,np.newaxis] \
                         - 2.0*np.dot(self.centroids*self.weights,self.input) \
                         + np.dot(self.weights**2,self.input**2)
        #temp_centroids = self.centroids/self.weights;
        #temp_distances = np.sum(temp_centroids**2,1)[:,np.newaxis] - 2*np.dot(temp_centroids,self.input) + \
        #                np.sum(self.input**2,0)[np.newaxis,:]
        #self.distances = temp_distances*(np.sum(self.weights**2,1)[:,np.newaxis])
        #print("Distance error: " + str(np.sum(np.sum((temp_distances - self.distances)**2))))
        print("Weighted Euclidean Distance")
    elif(hasattr(self,'do_cosinedistance')):
        self.distances = -np.dot(self.centroids,self.input)/(np.sqrt(np.sum(self.centroids**2.,1)[:,np.newaxis]*np.sum(self.input**2.,0)[np.newaxis,:]))
    else:
        self.distances = np.sum(self.centroids**2,1)[:,np.newaxis] - 2*np.dot(self.centroids,self.input) + \
                np.sum(self.input**2,0)[np.newaxis,:]
    distances_sorted = np.sort(self.distances,axis=0)
#    print("distances_sorted " + str(distances_sorted[0:10,0]))
    self.selected_neurons = self.distances > distances_sorted[num_selected,:]
    #keep track of this so we can count the number of times a centroid was selected
    self.saved_selected_neurons = np.copy(self.selected_neurons)
    
    #initialize selected count to 0
    if(not hasattr(self,'selected_count')):
        self.selected_count = np.zeros(self.saved_selected_neurons.shape[0])
    if(not hasattr(self,'eligibility_count')):
        self.eligibility_count = np.ones(self.saved_selected_neurons.shape[0])
    
    self.centroids_prime = (np.dot(self.input,(~self.selected_neurons).transpose())/ \
                      np.sum(~self.selected_neurons,1)).transpose()
    self.centroids_prime[np.isnan(self.centroids_prime)] = self.centroids[np.isnan(self.centroids_prime)]
    
    if(hasattr(self,'do_weighted_euclidean')):
        self.centroids_prime = self.centroids_prime*self.weights;

    self.output[self.selected_neurons] = 0;

def cluster_update_func(self):
    alpha = self.centroid_speed    
    self.centroids = self.centroids + alpha*(self.centroids_prime - self.centroids)
    
    #keep a count of the number of times a centroid was selected
    self.selected_count = self.selected_count + np.sum(~self.saved_selected_neurons,1)
    self.eligibility_count = self.eligibility_count + np.sum(~self.saved_selected_neurons,1)
    self.eligibility_count = self.eligibility_count*0.99
    #print("selected: " + str(self.selected_count))
    #print("eligibility: " + str(self.eligibility_count))

def cluster_cov_select_func(self):
    num_selected = self.num_selected
    #print("self.centroids shape: " + str(self.centroids.shape))
    #print("self.input shape: " + str(self.input.shape))
    #print("distances shape: " + str(self.distances.shape))
    #self.distances = np.sum(self.centroids**2,1)[:,np.newaxis] - 2*np.dot(self.centroids,self.input) + \
    #        np.sum(self.input**2,0)[np.newaxis,:]

    #use sklearn distance computation
    #self.distances = pairwise_distances(self.centroids,self.input.T)

    #use mahalanobis
    S = np.eye(self.centroids.shape[1])
    self.distances = pairwise_distances(self.centroids,self.input.T,metric='mahalanobis',VI=S)

    #self.distances = cdist(self.centroids,self.input.T,'euclidean')

    #print("before")
    #S = np.eye(785)
    #self.distances = cdist(self.centroids,self.input.T,'mahalanobis',VI=S)
    #print("after")


    distances_sorted = np.sort(self.distances,axis=0)
#    print("distances_sorted " + str(distances_sorted[0:10,0]))
    self.selected_neurons = self.distances > distances_sorted[num_selected,:]
    #keep track of this so we can count the number of times a centroid was selected
    self.saved_selected_neurons = np.copy(self.selected_neurons)
    
    #initialize selected count to 0
    if(not hasattr(self,'selected_count')):
        self.selected_count = np.zeros(self.saved_selected_neurons.shape[0])
    if(not hasattr(self,'eligibility_count')):
        self.eligibility_count = np.ones(self.saved_selected_neurons.shape[0])
    
    self.centroids_prime = (np.dot(self.input,(~self.selected_neurons).transpose())/ \
                      np.sum(~self.selected_neurons,1)).transpose()
    self.centroids_prime[np.isnan(self.centroids_prime)] = self.centroids[np.isnan(self.centroids_prime)]
    
    if(hasattr(self,'do_weighted_euclidean')):
        self.centroids_prime = self.centroids_prime*self.weights;

    self.output[self.selected_neurons] = 0;

def cluster_cov_update_func(self):
    alpha = self.centroid_speed    
    self.centroids = self.centroids + alpha*(self.centroids_prime - self.centroids)
    
    #keep a count of the number of times a centroid was selected
    self.selected_count = self.selected_count + np.sum(~self.saved_selected_neurons,1)
    self.eligibility_count = self.eligibility_count + np.sum(~self.saved_selected_neurons,1)
    self.eligibility_count = self.eligibility_count*0.99
    #print("selected: " + str(self.selected_count))
    #print("eligibility: " + str(self.eligibility_count))


select_names = {}
select_names['cluster_func'] = cluster_select_func
select_names['cluster_func_cov'] = cluster_cov_select_func

update_names = {}
update_names['cluster_func'] = cluster_update_func
update_names['cluster_func_cov'] = cluster_cov_update_func
