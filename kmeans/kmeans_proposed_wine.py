from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
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

# Load wine dataset
data = load_wine()
X = data.data

# Compute centroids using kmeans++
centroids = kmeans_plus_plus(X, K=3, N=len(X))

# Compute SSE using sklearn's KMeans
kmeans = KMeans(n_clusters=3, init=centroids, max_iter=300)
kmeans.fit(X)
sse = kmeans.inertia_

print("SSE:", sse)
