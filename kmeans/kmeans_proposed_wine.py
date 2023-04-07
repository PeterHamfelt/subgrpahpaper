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
n_clusters=3
# Compute centroids using kmeans++
centroids = kmeans_plus_plus(X, K=3, N=len(X))
"""
# Compute SSE using sklearn's KMeans
kmeans = KMeans(n_clusters=3, init=centroids, max_iter=300)
kmeans.fit(X)
sse = kmeans.inertia_

print("SSE:", sse)
"""
n_runs =10
sse_values = []

for i in range(n_runs):
    kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    kmeans.fit(X)
    sse = kmeans.inertia_
    sse_values.append(sse)
    print(f"Run {i+1}: SSE = {sse:.4f}")

# Check if there are differences in SSE values
if len(set(sse_values)) > 1:
    print("\nThe SSE values differ across the runs for proposed kmeans.")
else:
    print("\nThe SSE values are the same across the runs for proposed kmeans.")


#azure data factory
#data bricks
for i in range(n_runs):
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1)
    kmeans.fit(X)
    sse = kmeans.inertia_
    sse_values.append(sse)
    print(f"Run {i+1}: SSE = {sse:.4f}")

# Check if there are differences in SSE values
if len(set(sse_values)) > 1:
    print("\nThe SSE values differ across the runs for original_kmeans.")
else:
    print("\nThe SSE values are the same across the runs for original_kmeans.")
