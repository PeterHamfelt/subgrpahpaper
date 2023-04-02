import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


if __name__ == '__main__':
    # Load the Iris dataset
    data = load_iris()
    X = data.data

    # Run the k-means algorithm with K=3
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)

    # Calculate the sum of squared errors (SSE)
    sse = np.sum((X - kmeans.cluster_centers_[kmeans.labels_])**2)

    # Print the sum of squared errors (SSE)
    print(f"SSE: {sse:.2f}")
