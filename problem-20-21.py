import pandas as pd
import numpy as np

# Step 1: Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # only numeric features

# Step 2: Set number of clusters and iterations
K = 3
iterations = 10

# Step 3: Randomly choose K data points as initial centroids
np.random.seed(42)  # for reproducibility
centroids = X[np.random.choice(range(len(X)), K, replace=False)]

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Step 4: Run K-means iterations
for it in range(iterations):
    # Step 4a: Assign each point to closest centroid
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, c) for c in centroids]
        cluster_index = np.argmin(distances)
        clusters.append(cluster_index)
    clusters = np.array(clusters)

    # Step 4b: Update centroids by taking mean of points in each cluster
    new_centroids = []
    for k in range(K):
        cluster_points = X[clusters == k]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(centroids[k])  # if cluster empty, keep old centroid

    centroids = np.array(new_centroids)

# Step 5: Print final cluster means
print("\nFinal Cluster Means (Centroids):")
for i, c in enumerate(centroids):
    print(f"Cluster {i+1} Mean: {c}")
