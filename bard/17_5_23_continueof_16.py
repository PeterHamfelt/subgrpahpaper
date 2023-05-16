#working
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score
from scipy.sparse import coo_matrix
import torch_geometric.transforms as T
import pandas as pd
import networkx as nx

# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]
labels = data.y
num_classes = dataset.num_classes

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

# Define the kmeans_plus_plus function
def kmeans_plus_plus(X, K, N):
    centroids = []
    for k in range(K):
        if k == 0:
            feature_means = torch.mean(X, dim=0)
            distances = torch.sqrt(torch.sum((X - feature_means) ** 2, dim=1))
            closest_index = torch.argmin(distances)
            centroids.append(X[closest_index])
        else:
            dist_to_centroid = torch.sqrt(torch.sum((X - centroids[k-1]) ** 2, dim=1))
            sorted_indices = torch.argsort(dist_to_centroid)
            delete_indices = sorted_indices[:N // K]
            remaining_indices = list(set(range(N)) - set(delete_indices.tolist()))
            X = X[remaining_indices]
            N = len(X)  # Update N after removing data points
            feature_means = torch.mean(X, dim=0)
            distances = torch.sqrt(torch.sum((X - feature_means) ** 2, dim=1))
            closest_index = torch.argmin(distances)
            centroids.append(X[closest_index])
    return torch.stack(centroids)

# Define the run_clustering function
def run_clustering(normalized_embeddings, centroids, true_labels, n_clusters=7):
    results = {}
    
    # ... (previous code) ...
    
    # Run proposed K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil_score = silhouette_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    proposed_ch_score = calinski_harabasz_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_db_score = davies_bouldin_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_mi_score = adjusted_mutual_info_score(true_labels.numpy(), proposed_kmeans.labels_)
    
    results['proposed_sse'] = proposed_sse
    results['proposed_silhouette'] = proposed_sil_score
    results['proposed_ari'] = proposed_ari
    results['proposed_ch_score'] = proposed_ch_score
    results['proposed_db_score'] = proposed_db_score
    results['proposed_mi_score'] = [proposed_mi_score]  # Convert to a list or array

    return results



def impute_anomalies(graph):
    # Select nodes for anomalies (e.g., nodes 2, 5, and 10)
    anomaly_indices = [2, 5, 10]

    # Modify node features for anomalies
    for idx in anomaly_indices:
        graph.x[idx] = torch.randn(graph.x.size(1))


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

# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model.conv2(model.conv1(data.x, data.edge_index), data.edge_index)

# Normalize embeddings
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

# Perform clustering and anomaly detection
n_clusters = 7
centroids = kmeans_plus_plus(normalized_embeddings, K=7, N=len(normalized_embeddings))
result = run_clustering(normalized_embeddings, centroids, data.y)

# Convert the edge index to a NetworkX graph
G = nx.from_edgelist(data.edge_index.t().numpy())

# Impute anomalies in the graph
impute_anomalies(data)

# Identify anomalies
anomalies = [i for i, score in enumerate(result['proposed_mi_score']) if score > 0]
print("Anomalies found in clusters:", anomalies)

