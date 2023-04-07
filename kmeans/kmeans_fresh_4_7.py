"""
import numpy as np
from sklearn.cluster import KMeans

# Generate a random dataset with 2 clusters
X = np.random.rand(100, 2) * 10
y_true = np.concatenate((np.zeros(50), np.ones(50)))

# Define the number of clusters and the random state
n_clusters = 2
random_state = 1

# Initialize K-means with the same random state
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

# Fit K-means and calculate SSE
kmeans.fit(X)
sse1 = kmeans.inertia_

# Initialize K-means with a different random state
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + 1)

# Fit K-means and calculate SSE
kmeans.fit(X)
sse2 = kmeans.inertia_

print(f"SSE for run 1: {sse1}")
print(f"SSE for run 2: {sse2}")
"""
"""
import numpy as np
from sklearn.cluster import KMeans

# Generate a random dataset with 2 clusters
X = np.random.rand(100, 2) * 10
y_true = np.concatenate((np.zeros(50), np.ones(50)))

# Define the number of clusters and the random state
n_clusters = 2
random_state = 1

# Initialize K-means with the same random state
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

# Fit K-means and calculate SSE
kmeans.fit(X)
sse1 = kmeans.inertia_

# Run K-means 10 times and store the SSE values
sse_values = []
for i in range(10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=i)
    kmeans.fit(X)
    sse_values.append(kmeans.inertia_)

print(f"SSE values for 10 runs: {sse_values}")
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic dataset
n_samples = 300
n_clusters = 3
random_state = 42

X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)

# Plot the generated dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Generated Dataset with 3 Clusters')
plt.show()

from sklearn.cluster import KMeans

# Run K-means multiple times and store the SSE values
n_runs =10
sse_values = []

for i in range(n_runs):
    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1)
    kmeans.fit(X)
    sse = kmeans.inertia_
    sse_values.append(sse)
    print(f"Run {i+1}: SSE = {sse:.4f}")

# Check if there are differences in SSE values
if len(set(sse_values)) > 1:
    print("\nThe SSE values differ across the runs.")
else:
    print("\nThe SSE values are the same across the runs.")

