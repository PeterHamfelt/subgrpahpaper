"""
This is the complete implementation of the k-means++ algorithm for initializing the centroids of k-means clustering.

import numpy as np: Importing the NumPy library for array manipulation and mathematical operations.

def kmeans_plus_plus(X, K, N): Defining a function kmeans_plus_plus that takes in three arguments - the dataset X, the number of clusters K, and the number of points to remove for each iteration N.

centroids = []: Creating an empty list to store the centroids.

for k in range(K):: Looping over K clusters.

if k == 0:: For the first iteration, we need to select a random data point as the first centroid.

feature_means = np.mean(X, axis=0): Calculating the feature means of the dataset X.

distances = np.sqrt(np.sum((X - feature_means)**2, axis=1)): Calculating the distances of each data point in X from the feature means.

closest_index = np.argmin(distances): Finding the index of the data point that is closest to the feature means.

centroids.append(X[closest_index]): Adding the closest data point to the centroids list.

else:: For subsequent iterations, we need to remove the N/K nearest neighbors of the previous centroid before selecting the next centroid.

dist_to_centroid = np.sqrt(np.sum((X - centroids[k-1])**2, axis=1)): Calculating the distances of each data point in X from the previous centroid.

sorted_indices = np.argsort(dist_to_centroid): Sorting the distances in ascending order and returning the indices.

delete_indices = sorted_indices[:N//K]: Selecting the first N/K indices from the sorted indices to delete from the dataset.

X = np.delete(X, delete_indices, axis=0): Deleting the N/K nearest neighbors from the dataset X.

feature_means = np.mean(X, axis=0): Calculating the feature means of the updated dataset X.

distances = np.sqrt(np.sum((X - feature_means)**2, axis=1)): Calculating the distances of each data point in X from the feature means.

closest_index = np.argmin(distances): Finding the index of the data point that is closest to the feature means.

centroids.append(X[closest_index]): Adding the closest data point to the centroids list.

return np.array(centroids): Returning the final set of centroids as a NumPy array.

"""

import numpy as np

def kmeans_plus_plus(X, K, N):
    centroids = []
    for k in range(K):
        if k == 0:
            feature_means = np.mean(X, axis=0)
            distances = np.sqrt(np.sum((X - feature_means)**2, axis=1))
            closest_index = np.argmin(distances)
            centroids.append(X[closest_index])
        else:
            dist_to_centroid = np.sqrt(np.sum((X - centroids[k-1])**2, axis=1))
            sorted_indices = np.argsort(dist_to_centroid)
            delete_indices = sorted_indices[:N//K]
            X = np.delete(X, delete_indices, axis=0)
            feature_means = np.mean(X, axis=0)
            distances = np.sqrt(np.sum((X - feature_means)**2, axis=1))
            closest_index = np.argmin(distances)
            centroids.append(X[closest_index])
    return np.array(centroids)

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the iris dataset
iris = load_iris()
X = iris.data

# Set the parameters for kmeans_plus_plus
K = 3
N = len(X)

# Obtain the initial centroids using kmeans_plus_plus
centroids = kmeans_plus_plus(X, K, N)

# Fit KMeans with the initial centroids obtained from kmeans_plus_plus
kmeans = KMeans(n_clusters=K, init=centroids, n_init=1)
kmeans.fit(X)

# Compute the SSE (Sum of Squared Errors) for the KMeans solution
sse = kmeans.inertia_
print("SSE: ", sse)

  

# Print the sum of squared errors and the labels
print(f"SSE: {sse:.2f}")
#print(f"Labels: {labels}")

