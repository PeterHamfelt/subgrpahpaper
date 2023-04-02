from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np

# Load the wine dataset
X, y = load_wine(return_X_y=True)

# Create a KMeans object with K=3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans model to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the coordinates of the centroids
centroids = kmeans.cluster_centers_

# Calculate the sum of squared distances to the closest centroid for each data point
sse = np.sum((X - centroids[labels])**2)

print(f"SSE: {sse}")
