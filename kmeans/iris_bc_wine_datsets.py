from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np


# K-means++ algorithm
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


# Load datasets
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()
mnist = fetch_openml('mnist_784')
cifar = fetch_openml('CIFAR_10_SMALL')
#wholesale = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv', delimiter=',', skip_header=1)

# Scale the datasets
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris.data)
wine_scaled = scaler.fit_transform(wine.data)
cancer_scaled = scaler.fit_transform(cancer.data)
mnist_scaled = scaler.fit_transform(mnist.data.astype(float))
cifar_scaled = scaler.fit_transform(cifar.data.astype(float))
#wholesale_scaled = scaler.fit_transform(wholesale)

# Run K-means++ algorithm on Iris dataset
kmeans_pp_iris = KMeans(n_clusters=3, init=kmeans_plus_plus(iris_scaled, 3, len(iris_scaled)), max_iter=300, random_state=0).fit(iris_scaled)

# Run regular K-means algorithm on Iris dataset
kmeans_iris = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0).fit(iris_scaled)

# Run K-means++ algorithm on Wine dataset
kmeans_pp_wine = KMeans(n_clusters=3, init=kmeans_plus_plus(wine_scaled, 3, len(wine_scaled)), max_iter=300, random_state=0).fit(wine_scaled)

# Run regular K-means algorithm on Wine dataset
kmeans_wine = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=0).fit(wine_scaled)

# Run K-means++ algorithm on Breast cancer dataset
kmeans_pp_cancer = KMeans(n_clusters=2, init=kmeans_plus_plus(cancer_scaled, 2, len(cancer_scaled)), max_iter=300, random_state=0).fit(cancer_scaled)

# Run regular K-means algorithm on Breast cancer dataset
kmeans_cancer = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0).fit(cancer_scaled)

# Run K-means++ algorithm on MNIST dataset
X_mnist = mnist.data.astype(np.float64)
kmeans_pp_mnist = KMeans(n_clusters=10, init=kmeans_plus_plus(X_mnist, 10, len(X_mnist)), max_iter=300, random_state=0).fit(X_mnist)









#Run regular K-means algorithm on MNIST dataset
kmeans_mnist = KMeans(n_clusters=10, init='k-means++', max_iter=300, random_state=0).fit(X_mnist)

#Run K-means++ algorithm on CIFAR dataset
X_cifar = cifar.data.astype(np.float64)
kmeans_pp_cifar = KMeans(n_clusters=10, init=kmeans_plus_plus(X_cifar, 10, len(X_cifar)), max_iter=300, random_state=0).fit(X_cifar)

#Run regular K-means algorithm on CIFAR dataset
kmeans_cifar = KMeans(n_clusters=10, init='k-means++', max_iter=300, random_state=0).fit(X_cifar)



#Print SSEs
print("Iris dataset SSEs:")
print("K-means++: ", kmeans_pp_iris.inertia_)
print("Regular K-means: ", kmeans_iris.inertia_)

print("\nWine dataset SSEs:")
print("K-means++: ", kmeans_pp_wine.inertia_)
print("Regular K-means: ", kmeans_wine.inertia_)

print("\nBreast Cancer dataset SSEs:")
print("K-means++: ", kmeans_pp_cancer.inertia_)
print("Regular K-means: ", kmeans_cancer.inertia_)

print("\nMNIST dataset SSEs:")
print("K-means++: ", kmeans_pp_mnist.inertia_)
print("Regular K-means: ", kmeans_mnist.inertia_)

print("\nCIFAR dataset SSEs:")
print("K-means++: ", kmeans_pp_cifar.inertia_)
print("Regular K-means: ", kmeans_cifar.inertia_)







