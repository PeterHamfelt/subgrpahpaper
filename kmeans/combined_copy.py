import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score
import torch_geometric.transforms as T
import networkx as nx
import numpy as np

# Load the Cora dataset
dataset = Planetoid(root='/workspaces/subgraphpaper/data/Cora/', name='Cora')
data = dataset[0]

# Replace all labels other than 3 with 1
labels = torch.where(data.y == 3, data.y, torch.ones_like(data.y))
def subgraph_representation(G, X, Y, k=10, idx_map=None):
    V, E = G
    n_nodes = len(V)
    n_edges = len(E)
    
    # Create a mapping between the subgraph's node indices and indices in A and D matrices
    if idx_map is None:
        idx_map = dict(zip(V, range(n_nodes)))

    A = np.zeros((n_nodes, n_nodes))
    D = np.zeros((n_nodes, n_nodes))

    for i, j in E:
        # Use the mapping to index into A and D matrices
        i, j = idx_map.get(i, i), idx_map.get(j, j)
        A[i, j] = A[j, i] = 1
        D[i, i] += 1
        D[j, j] += 1

    W1 = np.ones((X.shape[1], X.shape[1]))
    W2 = np.ones((Y.shape[1], Y.shape[1]))

    for _ in range(k):
        a1 = []
        for i in range(n_nodes):
            neighbors = np.where(A[i] == 1)[0]
            AXW1 = X[neighbors] * A[i, neighbors][:, np.newaxis] @ W1
            a1_i = np.exp(AXW1) / np.sum(np.exp(AXW1), axis=0)
            a1.append(a1_i)
        a1 = np.vstack(a1)

        a2 = []
        for i, j in E:
            # Use the mapping to index into X and Y matrices
            i, j = idx_map.get(i, i), idx_map.get(j, j)
            a2_ij = np.exp(np.dot(X[i], X[j])) * np.exp(np.dot(Y[(i, j)], W2))
            a2_ij /= np.sum(a2_ij)
            a2.append(a2_ij)
        a2 = np.vstack(a2)

        W1 += np.sum(a1, axis=0) / n_nodes
        W2 += np.sum(a2, axis=0) / n_edges

    Z1 = np.sum(a1 * X, axis=0) / np.sum(a1)
    Z2 = np.sum(a2[:, np.newaxis] * np.array([Y[(i, j)] for i, j in E]), axis=0) / np.sum(a2)
    Z = np.hstack((Z1, Z2))

    return Z




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




num_classes = 2 # 0 indicates anomalies, 1 indicates normal



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


#Part 3: Model training and node embeddings
num_classes = 2 # 0 indicates anomalies, 1 indicates normal
#model = GCN(data.num_features, 16, num_classes)

model = GCN(data.num_features, 16, num_classes)

# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model.conv2(model.conv1(data.x, data.edge_index), data.edge_index)

# Normalize embeddings
normalized_embeddings = F.normalize(embeddings, p=3, dim=1)


#Part 4: Clustering, subgraph extraction, and subgraph representations

n_clusters = 7
centroids = kmeans_plus_plus(normalized_embeddings, K=7, N=len(normalized_embeddings))

# Run clustering
#result = run_clustering(normalized_embeddings, centroids, labels)

G = nx.from_edgelist(data.edge_index.t().numpy())

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

# Create edge features
edge_features = {}
for i, j in G.edges():
    edge_features[(i, j)] = np.hstack((G.nodes[i]['feature'], G.nodes[j]['feature']))

# Calculate subgraph representations
# Calculate subgraph representations
# Calculate subgraph representations
subgraph_representations = []
# Add this line before the loop for extracting subgraphs
inv_idx_map = {idx: i for i, idx in enumerate(G.nodes())}

# Replace the subgraph_representation function call inside the loop
#subgraph_representation_value = subgraph_representation((V, E), X, Y, idx_map=inv_idx_map)

subgraph_representations = []

# Add this line before the loop for extracting subgraphs
inv_idx_map = {idx: i for i, idx in enumerate(G.nodes())}

for subgraph in subgraphs:
    sg_nodes = list(subgraph.nodes())
    V = np.array([G.nodes[n]['feature'] for n in sg_nodes])
    E = list(subgraph.edges())
    #X = np.array([subgraph.nodes[i]['feature'] for i in range(len(sg_nodes))])
    X = np.array([subgraph.nodes[i]['feature'] for i in sg_nodes])

    Y = np.array([edge_features.get((sg_nodes.index(i), sg_nodes.index(j)), np.zeros(2 * data.num_features)) for i, j in E])
    
    # Call sub
