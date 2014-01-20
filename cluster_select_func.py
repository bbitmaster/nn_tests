import numpy as np

#TODO: move this to it's own package later. Test here for now.
def cluster_select_func(self,params):
    num_selected = self.num_selected
    self.distances = np.sum(self.centroids**2,1)[:,np.newaxis] - 2*np.dot(self.centroids,self.input) + \
                  np.sum(self.input**2,0)[np.newaxis,:]

    distances_sorted = np.sort(self.distances,axis=0)
    self.selected_neurons = self.distances > distances_sorted[num_selected,:]
    
    self.centroids_prime = (np.dot(self.input,(~self.selected_neurons).transpose())/ \
                      np.sum(~self.selected_neurons,1)).transpose()
    self.centroids_prime[np.isnan(self.centroids_prime)] = self.centroids[np.isnan(self.centroids_prime)]

    self.output[self.selected_neurons] = 0;

def cluster_update_func(self):
    alpha = self.centroid_speed    
    self.centroids = self.centroids + alpha*(self.centroids_prime - self.centroids)

