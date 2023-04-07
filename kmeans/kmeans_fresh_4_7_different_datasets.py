"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def compare_kmeans_runs(n_samples, n_clusters, n_runs):
    # Generate synthetic dataset
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=42)

    # Plot the generated dataset
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title(f'Generated Dataset with {n_clusters} Clusters')
    plt.show()

    # Run K-means multiple times and store the SSE values
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

# Example usage:
compare_kmeans_runs(n_samples=300, n_clusters=4, n_runs=5)
"""


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

def compare_kmeans_runs_on_standard_datasets(dataset_name, n_clusters, n_runs):
    # Load the dataset
    if dataset_name == 'iris':
        data = datasets.load_iris()
    elif dataset_name == 'digits':
        data = datasets.load_digits()
    else:
        raise ValueError("Invalid dataset name. Supported datasets: 'iris', 'digits'")

    X = data.data

    # Run K-means multiple times and store the SSE values
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

# Example usage:
compare_kmeans_runs_on_standard_datasets(dataset_name='iris', n_clusters=3, n_runs=10)
compare_kmeans_runs_on_standard_datasets(dataset_name='digits', n_clusters=10, n_runs=10)
