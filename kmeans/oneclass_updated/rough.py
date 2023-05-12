

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
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

# Load the Cora dataset
dataset = Planetoid(root='/workspaces/subgraphpaper/data/Cora/', name='Cora')
data = dataset[0]

# Preprocess the labels
labels = torch.where(data.y == 3, torch.zeros_like(data.y), torch.ones_like(data.y))






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



# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)





# Instantiate the GCN model, loss function, and optimizer
num_classes = 2
model = GCN(data.num_features, 16, num_classes)
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
train_indices = torch.where(data.train_mask)[0]
val_indices = torch.where(data.val_mask)[0]
n_epochs = 200
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_function(out[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    # Calculate the validation loss
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = loss_function(val_out[val_indices], labels[val_indices])
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
        model.train()

# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)





# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# Normalize embeddings
normalized_embeddings = F.normalize(embeddings, p=3, dim=1)
print('********************')
print(type(normalized_embeddings))
print(normalized_embeddings.size())

n_clusters = 7
centroids = kmeans_plus_plus(normalized_embeddings, K=7, N=len(normalized_embeddings))

def run_clustering(normalized_embeddings, centroids, true_labels, n_clusters=7):
    results = {}
    
    # Run proposed K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil_score = silhouette_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    proposed_ch_score = calinski_harabasz_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_db_score = davies_bouldin_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_mi_score = adjusted_mutual_info_score(true_labels.numpy(), proposed_kmeans.labels_)
    
    # Store results in a dictionary
    results['proposed_sse'] = proposed_sse
    results['proposed_silhouette'] = proposed_sil_score
    results['proposed_ari'] = proposed_ari
    results['proposed_ch_score'] = proposed_ch_score
    results['proposed_db_score'] = proposed_db_score
    results['proposed_mi_score'] = proposed_mi_score
    return results

# Run clustering
result = run_clustering(normalized_embeddings, centroids, labels)

G = nx.from_edgelist(data.edge_index.t().numpy())

# ...
# Convert graph data to NetworkX format
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

# ... (continue with the rest of the code)
# Analyze each subgraph and compute some properties
subgraph_properties = []

for subgraph in subgraphs:
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()
    avg_degree = sum(dict(subgraph.degree()).values()) / num_nodes

    # Calculate connected components
    connected_components = list(nx.connected_components(subgraph))

    # Calculate the size of the largest connected component
    largest_cc = max(connected_components, key=len)
    largest_cc_size = len(largest_cc)

    # Compute the average clustering coefficient for the subgraph
    avg_clustering_coefficient = nx.average_clustering(subgraph)

    subgraph_properties.append({
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'largest_cc_size': largest_cc_size,
        'avg_clustering_coefficient': avg_clustering_coefficient,
    })

# Print subgraph properties
for idx, properties in enumerate(subgraph_properties):
    print(f"Cluster {idx}:")
    print(f"  Number of nodes: {properties['num_nodes']}")
    print(f"  Number of edges: {properties['num_edges']}")
    print(f"  Average degree: {properties['avg_degree']:.2f}")
    print(f"  Largest connected component size: {properties['largest_cc_size']}")
    print(f"  Average clustering coefficient: {properties['avg_clustering_coefficient']:.2f}")
    print()

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

# ... (rest of the code)


# Attribute-based anomaly detection: detect anomalies in the node embeddings
attribute_anomaly_scores = []
# Calculate accuracy
# We consider the class with the majority of nodes in a cluster to be the label of that cluster
# For the anomalies (class 0), we reverse the labels, since we consider the neural network class to be normal
cluster_labels = []
for subgraph in subgraphs:
    if nx.is_connected(subgraph):
        node_labels = [labels[i].item() for i in subgraph.nodes()]
        if sum(node_labels) > len(node_labels) / 2:
            cluster_labels.append(1)
        else:
            cluster_labels.append(0)
    else:
        cluster_labels.append(0)

pred_labels = np.array([cluster_labels[label] for label in kmeans.labels_])
accuracy = sum(pred_labels == (labels == 3).numpy()) / len(labels)
print("accuracy",accuracy)


# Plot node embeddings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = normalized_embeddings[:, 0].numpy()
ys = normalized_embeddings[:, 1].numpy()
colors = [int(label) for label in labels.numpy()]
ax.scatter(xs, ys, c=colors)
plt.savefig('scatter.png')
plt.show()

# Plot subgraphs
for i, subgraph in enumerate(subgraphs):
    nodes = list(subgraph.nodes())
    subgraph_colors = ['b' if labels[node] == 1 else 'r' for node in nodes]
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx(subgraph, pos=pos, node_color=subgraph_colors, with_labels=False)
    plt.savefig(f"subgraph_{i}.png")
    plt.clf()


# Display anomalies in the subgraphs
for i, subgraph in enumerate(subgraphs):
    if i in anomalies:
        nodes = list(subgraph.nodes())
        subgraph_colors = ['b' if labels[node] == 1 else 'r' for node in nodes]
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw_networkx(subgraph, pos=pos, node_color=subgraph_colors, with_labels=False)
        plt.savefig(f"anomaly_subgraph_{i}.png")
        plt.clf()

