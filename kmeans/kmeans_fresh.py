import numpy as np


def kmeans(X, K, N):
    # Step 1: Compute the feature means of the dataset X
    means = np.mean(X, axis=0)

    # Step 2-4: Initialize the centroids using KMS
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        if k == 0:
            # Choose the first centroid randomly
            idx = np.random.choice(X.shape[0])
            centroids[k] = X[idx]
        else:
            # Choose the next centroid using KMS
            dist = np.sum((X - centroids[k-1])**2, axis=1)
            probs = dist / np.sum(dist)
            idx = np.random.choice(X.shape[0], p=probs)
            centroids[k] = X[idx]
            # Delete N/K nearest neighbors of the chosen instance
            dist = np.sum((X - centroids[k])**2, axis=1)
            nn = np.argsort(dist)[:int(N/K)]
            X = np.delete(X, nn, axis=0)
            dist = np.delete(dist, nn)

            # Update N
            N -= int(N/K)

    # Initialize the labels and SSE
    labels = np.zeros(X.shape[0], dtype=int)
    sse = np.inf

    # Run the k-means algorithm until convergence
    while True:
        # Assign each instance to the closest centroid
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - centroids)**2, axis=1)
            labels[i] = np.argmin(dist)

        # Update the centroids as the means of the instances with the same label
        for k in range(K):
            mask = labels == k
            centroids[k] = np.mean(X[mask], axis=0)

        # Compute the sum of squared errors
        sse_new = 0
        for k in range(K):
            mask = labels == k
            sse_new += np.sum((X[mask] - centroids[k])**2)

        # Check for convergence
        if np.abs(sse_new - sse) < 1e-6:
            break
        else:
            sse = sse_new

    return labels, sse


if __name__ == '__main__':
    # Generate a random dataset with 100 instances and 2 features
    np.random.seed(0)
    X = np.random.randn(100, 2)

    # Cluster the dataset using K-Means Sampling (KMS) with K=3 and N=20
    labels, sse = kmeans(X, K=3, N=20)

    # Print the sum of squared errors and the labels
    print(f"SSE: {sse:.2f}")
    print(f"Labels: {labels}")
