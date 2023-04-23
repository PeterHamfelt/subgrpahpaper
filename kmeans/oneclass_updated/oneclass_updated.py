"""

Load the Cora dataset and preprocess the labels.
Define a Graph Convolutional Network (GCN) model.
Get node embeddings from the trained model.
Normalize the embeddings.
Cluster the embeddings using k-means++.
Convert the graph data to NetworkX format and divide it into subgraphs based on clustering.
Perform attribute-based anomaly detection on node embeddings.
Calculate the accuracy of the classification.
Plot node embeddings and subgraphs.
=========training not done
"""





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

# Load the Cora dataset
dataset = Planetoid(root='/workspaces/subgraphpaper/data/Cora/', name='Cora')
data = dataset[0]

# Replace all labels other than 3 with 1
labels = torch.where(data.y == 3, data.y, torch.ones_like(data.y))


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










num_classes = 2 # 0 indicates anomalies, 1 indicates normal

model = GCN(data.num_features, 16, num_classes)

# Get node embeddings from the trained model
with torch.no_grad():
    #embeddings = model.conv2(model.conv1(data.x, data.edge_index), data.edge_index)
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
    #proposed_kmeans = kmeans_plus_plus(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
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
# Convert graph data to NetworkX format
#G = from_scipy_sparse_matrix(data.adjacency_matrix())
# You can add node attributes to the graph, for example:
for i in range(data.x.shape[0]):
    G.nodes[i]['feature'] = data.x[i].numpy()
# Compute clustering on the embeddings
kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
#kmeans = kmeans_plus_plus(normalized_embeddings, K=7, N=len(normalized_embeddings))

clusters = kmeans.labels_

# Extract subgraphs for each cluster
subgraphs = []
for c in range(n_clusters):
    nodes = [i for i, x in enumerate(clusters) if x == c]
    subgraph = G.subgraph(nodes)
    subgraphs.append(subgraph)



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

print("Accuracy:", accuracy)
# Plot node embeddings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = normalized_embeddings[:, 0].numpy()
ys = normalized_embeddings[:, 1].numpy()
#zs = normalized_embeddings[:, 2].numpy()
colors = [int(label) for label in labels.numpy()]
ax.scatter(xs, ys, c=colors)
plt.savefig('scatter.png')
plt.show()

# ... (previous code)

# Plot subgraphs
# Plot subgraphs
# Plot subgraphs
for i, subgraph in enumerate(subgraphs):
    nodes = list(subgraph.nodes())
    subgraph_colors = ['b' if labels[node] == 1 else 'r' for node in nodes]
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx(subgraph, pos=pos, node_color=subgraph_colors, with_labels=False)
    #plt.savefig(f"subgraph_{i}.png")
    plt.clf()



    

