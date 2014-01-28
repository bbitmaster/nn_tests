import numpy as np

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
    print("def cluster_update_func(self):");
    self.centroids = self.centroids + alpha*(self.centroids_prime - self.centroids)

def cluster_select_func_starvation1(self,params):
    num_selected = self.num_selected
    #init starvation if applicable
    if(not hasattr(self,'starvation')):
        self.starvation = np.ones((self.node_count+1,1))
        print("initializing starvation count");

    self.distances = np.sum(self.centroids**2,1)[:,np.newaxis] - 2*np.dot(self.centroids,self.input) + \
                  np.sum(self.input**2,0)[np.newaxis,:]
    #scale by starvation trace
    self.distances = self.distances*self.starvation

    distances_sorted = np.sort(self.distances,axis=0)
    self.selected_neurons = self.distances > distances_sorted[num_selected,:]
    self.saved_selected_neurons = np.copy(self.selected_neurons)
     
    self.centroids_prime = (np.dot(self.input,(~self.selected_neurons).transpose())/ \
                      np.sum(~self.selected_neurons,1)).transpose()
    self.centroids_prime[np.isnan(self.centroids_prime)] = self.centroids[np.isnan(self.centroids_prime)]

    self.output[self.selected_neurons] = 0;

def cluster_update_func_starvation1(self):
    alpha = self.centroid_speed    
    self.centroids = self.centroids + alpha*(self.centroids_prime - self.centroids)
    #update starvation count
    selected_count = np.sum(self.saved_selected_neurons,1)
    self.starvation = self.starvation + selected_count[:,np.newaxis]

select_names = {}
select_names['cluster_func'] = cluster_select_func
select_names['cluster_func_starvation1'] = cluster_select_func_starvation1

update_names = {}
update_names['cluster_func'] = cluster_update_func
update_names['cluster_func_starvation1'] = cluster_update_func_starvation1
