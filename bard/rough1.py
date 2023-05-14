import networkx as nx
import torch
import community
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from torch_geometric.datasets import KarateClub

# Load the Karate Club dataset
dataset = KarateClub()

# Convert the data to a NetworkX graph object
G = nx.from_edgelist(dataset.data.edge_index.T)

# Perform community detection using Greedy Modularity
greedy_communities = greedy_modularity_communities(G)

# Perform community detection using Louvain
louvain_communities = community.best_partition(G)

# Step 4: Anomalous Subgraph Detection using Methodology 1 (Greedy Modularity)
anomalous_subgraphs_greedy = []
for community in greedy_communities:
    # Perform clustering and anomaly detection within each community

    # Step 4a: Ego-net analysis
    ego_net = G.subgraph(community)

    # Step 4b: Non-overlapping clustering (K-means)
    kmeans = KMeans(n_clusters=2)
    node_features = nx.to_numpy_array(ego_net)
    clustering_results = kmeans.fit_predict(node_features)

    # Step 4c: Anomaly detection within clusters (Isolation Forest)
    anomaly_detector = IsolationForest(contamination=0.1)
    anomalous_nodes = []
    for cluster_id in set(clustering_results):
        cluster = [node for node, c in enumerate(clustering_results) if c == cluster_id]
        cluster_features = node_features[cluster]
        anomaly_scores = anomaly_detector.fit_predict(cluster_features)
        anomalous_nodes.extend([node for node, score in zip(cluster, anomaly_scores) if score == -1])

    # Step 4d: Store anomalous subgraphs
    if len(anomalous_nodes) > 0:
        anomalous_subgraphs_greedy.append(ego_net.subgraph(anomalous_nodes))

# Step 5: Anomalous Subgraph Detection using Methodology 2 (Louvain)
anomalous_subgraphs_louvain = []
for community_id in set(louvain_communities.values()):
    community_nodes = [node for node, c in louvain_communities.items() if c == community_id]
    # Perform clustering and anomaly detection within each community

    # Step 5a: Ego-net analysis
    ego_net = G.subgraph(community_nodes)

    # Step 5b: Non-overlapping clustering (K-means)
    kmeans = KMeans(n_clusters=2)
    node_features = nx.to_numpy_array(ego_net)
    clustering_results = kmeans.fit_predict(node_features)

    # Step 5c: Anomaly detection within clusters (Isolation Forest)
    anomaly_detector = IsolationForest(contamination=0.1)
    anomalous_nodes = []
    for cluster_id in set(clustering_results):
        cluster = [node for node, c in enumerate(clustering_results) if c == cluster_id]
        cluster_features = node_features[cluster]
        anomaly_scores = anomaly_detector.fit_predict(cluster_features)
        anomalous_nodes.extend([node for node, score in zip(cluster, anomaly_scores) if score == -1])

    # Step 5d: Store anomalous subgraphs
    if len(anomalous_nodes) > 0:
        anomalous_subgraphs_louvain.append(ego_net.subgraph(anomalous_nodes))

# Step 6: Compare the performance (accuracy) of the two methodologies
# Step 6: Compare the performance (accuracy) of the two methodologies
accuracy_greedy = len(anomalous_subgraphs_greedy) / len(greedy_communities)
accuracy_louvain = len(anomalous_subgraphs_louvain) / len(louvain_communities)

# Print the results
print("Performance Comparison:")
print("Accuracy (Greedy Modularity): {:.2f}".format(accuracy_greedy))
print("Accuracy (Louvain): {:.2f}".format(accuracy_louvain))
