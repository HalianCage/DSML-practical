# 9. Write a program to do the following: You have given a collection of 8
# points. P1=[0.1,0.6] P2=[0.15,0.71] P3=[0.08,0.9] P4=[0.16, 0.85]
# P5=[0.2,0.3] P6=[0.25,0.5] P7=[0.24,0.1] P8=[0.3,0.2]. Perform the k-mean
# clustering with initial centroids as m1=P1 =Cluster#1=C1 and
# m2=P8=cluster#2=C2. Answer the following 1] Which cluster does P6
# belong to? 2] What is the population of a cluster around m2? 3] What is
# the updated value of m1 and m2? 

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import copy
import pandas as pd

# Given points
points = np.array([[0.1, 0.6], [0.15, 0.71], [0.08, 0.9], [0.16, 0.85],
                   [0.2, 0.3], [0.25, 0.5], [0.24, 0.1], [0.3, 0.2]])   
# Initial centroids
m1 = np.array([0.1, 0.6])  # P1
m2 = np.array([0.3, 0.2])  # P8
centroids = np.array([m1, m2])
# Function to assign clusters based on closest centroid
def assign_clusters(points, centroids):
    distances = cdist(points, centroids, 'euclidean')
    return np.argmin(distances, axis=1)
# Function to update centroids based on mean of assigned points
def update_centroids(points, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = points[clusters == i]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
    return np.array(new_centroids)
# K-means clustering
k = 2
clusters = assign_clusters(points, centroids)
new_centroids = update_centroids(points, clusters, k)
# Output results
print("Initial Centroids:")
print("m1 (C1):", m1)
print("m2 (C2):", m2)
print("\nCluster Assignments:", clusters)
# 1] Which cluster does P6 belong to?
p6_index = 5  # P6 is the 6th point (index 5)
print("\n1] P6 belongs to Cluster #:", clusters[p6_index] + 1)
# 2] What is the population of a cluster around m2?
m2_cluster_population = np.sum(clusters == 1)  # Cluster #2 corresponds to index 1
print("2] Population of Cluster around m2 (C2):", m2_cluster_population)
# 3] What is the updated value of m1 and m2?
print("3] Updated Centroids:")
print("Updated m1 (C1):", new_centroids[0])
print("Updated m2 (C2):", new_centroids[1])
