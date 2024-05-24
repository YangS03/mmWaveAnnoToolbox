# import cupy as np
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, Normalize
from copy import deepcopy

class DBSCAN:
    def __init__(self):
        
        self.targets = []
        
    def set_matrix(self, matrix):
        
        self.matrix = matrix
        self.points = np.argwhere(matrix) # [num_points, 2]
        self.unvisited = deepcopy(self.points)
        self.num_points = self.points.shape[0]
        self.clusters = []
        
    def fit(self, matrix=None, eps=None, min_pts=None):
        
        self.eps = eps
        self.min_pts = min_pts
        self.set_matrix(matrix)
        
        while self.unvisited.shape[0] > 0:
           
            # Randomly select an unvisited point
            random_idx = np.random.choice(self.unvisited.shape[0], size=None)
            idx = self.unvisited[random_idx, :]
            # Check if the selected point is a core point
            if not self.check_if_core(idx):
                # Direct remove the point
                self.unvisited = np.delete(self.unvisited, random_idx, axis=0)
                continue
            neighbors_idx = self.get_new_neighbors(idx)
            neighbors = self.unvisited[neighbors_idx, :]
            self.unvisited = np.delete(self.unvisited, neighbors_idx, axis=0)
            # If the selected point is a core point, create a new cluster
            cluster_edge = neighbors
            cluster = np.array([idx])
            # Expand the cluster recursively
            cluster_edge, cluster = self.extend_cluster(cluster_edge, cluster)
            cluster = np.unique(cluster, axis=0)

            self.clusters.append(cluster)
            
        # Sort by cluster size
        self.clusters.sort(key=lambda x: len(x), reverse=True) 
        
        return self.clusters

    def extend_cluster(self, cluster_edge: np.array, cluster: np.array):

        while cluster_edge.shape[0] > 0:
            
            idx = cluster_edge[0, :]
            cluster_edge = np.delete(cluster_edge, 0, axis=0)
            cluster = np.append(cluster, np.array([idx]), axis=0)
            
            if self.check_if_core(idx):
                neighbors_idx = self.get_new_neighbors(idx)
                neighbors = self.unvisited[neighbors_idx, :]
                self.unvisited = np.delete(self.unvisited, neighbors_idx, axis=0)
                cluster_edge = np.concatenate((cluster_edge, neighbors), axis=0) 
            
        return cluster_edge, cluster
         
    def check_if_core(self, idx):
        neighbors = self.points[self.distance_func(self.points, idx) < self.eps]
        return neighbors.shape[0] >= self.min_pts
    
    def get_new_neighbors(self, idx):
        neighbors_idx = self.distance_func(self.unvisited, idx) < self.eps
        return neighbors_idx
    
    def distance_func(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=2, axis=-1)
    
   
        

