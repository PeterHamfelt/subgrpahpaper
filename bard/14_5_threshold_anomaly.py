import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score
#from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


#from networkx.convert_matrix import from_scipy_sparse_matrix

import torch_geometric.transforms as T

import pandas as pd
import networkx as nx

# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]
labels = data.y

num_classes = dataset.num_classes

# ... (same as before) ...
def kmeans_plus_plus(X, K, N):
    centroids = []
    for k in range(K):
        if k == 0:
            feature_means = torch.mean(X, dim=0)
            distances = torch.sqrt(torch.sum((X - feature_means)**2, dim=1))
            closest_index = torch.argmin(distances)
            centroids.append(X[closest_index])
        else:
            dist_to_centroid = torch.sqrt(torch.sum((X - centroids[k-1])**2, dim=1))
            sorted_indices = torch.argsort(dist_to_centroid)
            delete_indices = sorted_indices[:N//K]
            remaining_indices = list(set(range(N)) - set(delete_indices.tolist()))
            X = X[remaining_indices]
            N = len(X)  # Update N after removing data points
            feature_means = torch.mean(X, dim=0)
            distances = torch.sqrt(torch.sum((X - feature_means)**2, dim=1))
            closest_index = torch.argmin(distances)
            centroids.append(X[closest_index])
    return torch.stack(centroids)

"""
# K-means++ algorithm
def d2_weighted_kmeans_plus_plus(X, K):
    centroids = []
    N = len(X)

    # Choose the data point closest to the mean of all data points as the first centroid.
    feature_means = torch.mean(X, dim=0)
    distances = torch.sqrt(torch.sum((X - feature_means) ** 2, dim=1))
    closest_index = torch.argmin(distances)
    centroids.append(X[closest_index])

    # Select remaining centroids using D2-weighting method.
    for k in range(1, K):
        # Calculate squared distances from data points to the closest existing centroid.
        dist_to_closest_centroid = torch.min(torch.cdist(X, torch.stack(centroids), p=2) ** 2, dim=1)[0]

        # Choose the next centroid with probability proportional to the squared distance.
        probabilities = dist_to_closest_centroid / torch.sum(dist_to_closest_centroid)
        next_centroid_idx = torch.multinomial(probabilities, 1).squeeze()
        centroids.append(X[next_centroid_idx])

    return torch.stack(centroids)
"""
# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a GCN model
model = GCN(data.num_features, 16, num_classes)

# ... (same as before) ...
# Create a GCN model
model = GCN(data.num_features, 16, num_classes)

# Train the model with early stopping
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()
model.train()

patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Compute validation loss
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Training stopped after {epoch} epochs.")
            break
# Train the model with early stopping
# ... (same as before) ...

# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model.conv2(model.conv1(data.x, data.edge_index), data.edge_index)

# Normalize embeddings
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

n_clusters = 7
centroids = kmeans_plus_plus(normalized_embeddings, K=7, N=len(normalized_embeddings))
def run_clustering(normalized_embeddings, centroids, true_labels, n_clusters=7):
    results = {}
    
    """
    print("************************")
    print("centroids=",centroids)
    print("************************")
    """
    
    
    
    
    
    # Run proposed K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil_score = silhouette_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    proposed_ch_score = calinski_harabasz_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_db_score = davies_bouldin_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_mi_score = adjusted_mutual_info_score(true_labels.numpy(), proposed_kmeans.labels_)
    
    
    
    
    
    """""
    
    # Run original K-means
    original_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1,random_state=42).fit(normalized_embeddings)
    original_sse = original_kmeans.inertia_
    original_sil_score = silhouette_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_ari = adjusted_rand_score(true_labels.numpy(), original_kmeans.labels_)
    original_ch_score = calinski_harabasz_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_db_score = davies_bouldin_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_mi_score = adjusted_mutual_info_score(true_labels.numpy(), original_kmeans.labels_)
   """



    
    # Store results in a dictionary
    results['proposed_sse'] = proposed_sse
    #results['original_sse'] = original_sse
    results['proposed_silhouette'] = proposed_sil_score
    #results['original_silhouette'] = original_sil_score
    results['proposed_ari'] = proposed_ari
    #results['original_ari'] = original_ari
    results['proposed_ch_score'] = proposed_ch_score
    #results['original_ch_score'] = original_ch_score
    results['proposed_db_score'] = proposed_db_score
    #results['original_db_score'] = original_db_score
    results['proposed_mi_score'] = proposed_mi_score
    #results['original_mi_score'] = original_mi_score
    return results


# Run clustering
result = run_clustering(normalized_embeddings, centroids, data.y)
#from networkx.convert_matrix import from_scipy_sparse_matrix
# Convert the edge index to a NetworkX graph
G = nx.from_edgelist(data.edge_index.t().numpy())
# Convert graph data to NetworkX format
#G = from_scipy_sparse_matrix(data.adjacency_matrix())
# You can add node attributes to the graph, for example:
for i in range(data.x.shape[0]):
    G.nodes[i]['feature'] = data.x[i].numpy()
# Compute clustering on the embeddings
kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
clusters = kmeans.labels_

# Extract subgraphs for each cluster
subgraphs = []
for c in range(n_clusters):
    nodes = [i for i, x in enumerate(clusters) if x == c]
    subgraph = G.subgraph(nodes)
    subgraphs.append(subgraph)

import networkx as nx

# ... (code to generate or load your graph G, and divide it into subgraphs)

# Calculate threshold values based on the 15% criteria
# Calculate threshold values based on the 15% criteria
n_nodes_list = [subgraph.number_of_nodes() for subgraph in subgraphs]
# Number of nodes:
# Calculate the average number of nodes
avg_n_nodes = sum(n_nodes_list) / len(n_nodes_list)

# Calculate the lower threshold as 85% of the mean
threshold_low_nodes = int(avg_n_nodes * 0.85)

# Calculate the upper threshold as 115% of the mean
threshold_high_nodes = int(avg_n_nodes * 1.15)


distances_list = [nx.average_shortest_path_length(subgraph) if nx.is_connected(subgraph) else 0 for subgraph in subgraphs]
avg_distance = sum(distances_list) / len(distances_list)
threshold_low_avg_distance = avg_distance * 0.15
threshold_high_avg_distance = avg_distance * (1 - 0.15)

densities = [nx.density(subgraph) for subgraph in subgraphs]
avg_density = sum(densities) / len(densities)
threshold_low_density = avg_density * 0.15
threshold_high_density = avg_density * (1 - 0.15)

# ... (rest of the code)


# Attribute-based anomaly detection: detect anomalies in the node embeddings
attribute_anomaly_scores = []

for subgraph in subgraphs:
    n_nodes = subgraph.number_of_nodes()
    avg_distance = 0

    if n_nodes > 0 and nx.is_connected(subgraph):
        distances = nx.average_shortest_path_length(subgraph)
        avg_distance = sum(distances.values()) / n_nodes

    attribute_anomaly_score = 0
    if n_nodes <= threshold_low_nodes or n_nodes >= threshold_high_nodes or avg_distance <= threshold_low_avg_distance or avg_distance >= threshold_high_avg_distance:
        attribute_anomaly_score = 1

    attribute_anomaly_scores.append(attribute_anomaly_score)

# Graph structure-based anomaly detection: detect anomalies in the subgraphs
structure_anomaly_scores = []

for subgraph in subgraphs:
    density = nx.density(subgraph)

    structure_anomaly_score = 0
    if density <= threshold_low_density or density >= threshold_high_density:
        structure_anomaly_score = 1

    structure_anomaly_scores.append(structure_anomaly_score)

#Combine attribute-based and graph structure-based anomaly scores
w_attr = 0.4
w_struct = 0.6

weighted_attribute_anomaly_scores = [score * w_attr for score in attribute_anomaly_scores]
weighted_structure_anomaly_scores = [score * w_struct for score in structure_anomaly_scores]

# Calculate the weighted combined anomaly scores
weighted_combined_anomaly_scores = [sum(scores) for scores in zip(weighted_attribute_anomaly_scores, weighted_structure_anomaly_scores)]

# Find the indices of clusters that are considered anomalies based on the weighted combined anomaly scores
anomaly_indices = [i for i, score in enumerate(weighted_combined_anomaly_scores) if score > 0]

print("Weighted attribute-based anomaly scores:", weighted_attribute_anomaly_scores)
print("Weighted graph structure-based anomaly scores:", weighted_structure_anomaly_scores)
print("Weighted combined anomaly scores:", weighted_combined_anomaly_scores)
print("Anomalies found in clusters:", anomaly_indices)

"""
#Combine attribute-based and graph structure-based anomaly scores
combined_anomaly_scores = [attr_score + struct_score for attr_score, struct_score in zip(attribute_anomaly_scores, structure_anomaly_scores)]

#Print anomaly scores
print("Attribute-based anomaly scores:", attribute_anomaly_scores)
print("Graph structure-based anomaly scores:", structure_anomaly_scores)
print("Combined anomaly scores:", combined_anomaly_scores)

"""
#G = from_scipy_sparse_matrix(data.adjacency_matrix())
#G = from_scipy_sparse_matrix(data.adjacency_matrix().to_scipy_sparse_matrix())
#G = nx.from_scipy_sparse_matrix(data.adjacency_matrix().to_scipy_sparse_matrix())
#G = nx.from_scipy_sparse_matrix(csr_matrix(data.adjacency_matrix().to_scipy_sparse_matrix()))
#adj_matrix_coo = coo_matrix(data.adjacency_matrix().to_scipy_sparse_matrix())
#G = nx.from_scipy_sparse_matrix(adj_matrix_coo)

import networkx as nx
import numpy as np

# Create an adjacency matrix from the edge_index tensor
adj_matrix = torch.zeros((data.num_nodes, data.num_nodes))
adj_matrix[data.edge_index[0], data.edge_index[1]] = 1
adj_matrix_np = adj_matrix.numpy()

# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(adj_matrix_np)

#Identify anomalies
anomalies = [i for i, score in enumerate(weighted_combined_anomaly_scores) if score > 0]
print("Anomalies found in clusters:", anomalies)

