import torch
import community
from torch_geometric.datasets import KarateClub

# Load the datasets
datasets = [KarateClub]

# Perform community detection using Greedy Modularity
greedy_communities = []
for dataset in datasets:
    # Get the graph data
    graph = dataset

    # Perform community detection
    greedy_communities.append(community.best_partition(graph,
                                                         directed=False))

# Perform anomaly detection using Isolation Forest
anomaly_detector = IsolationForest(contamination=0.1)
anomalous_subgraphs = []
for greedy_community in greedy_communities:
    # Perform clustering and anomaly detection within each community

    # Step 4a: Ego-net analysis
    ego_net = graph.subgraph(greedy_community)

    # Step 4b: Non-overlapping clustering (K-means)
    kmeans = KMeans(n_clusters=2)
    node_features = ego_net.x
    clustering_results = kmeans.fit_predict(node_features)

    # Step 4c: Anomaly detection within clusters (Isolation Forest)
    anomalous_nodes = []
    for cluster_id in set(clustering_results):
        cluster = [node for node, c in enumerate(clustering_results) if c == cluster_id]
        cluster_features = node_features[cluster]
        anomaly_scores = anomaly_detector.fit_predict(cluster_features)
        anomalous_nodes.extend([node for node, score in zip(cluster, anomaly_scores) if score == -1])

    # Step 4d: Store anomalous subgraphs
    if len(anomalous_nodes) > 0:
        anomalous_subgraphs.append(ego_net.subgraph(anomalous_nodes))

# Print the results
for dataset, anomalous_subgraph in zip(datasets, anomalous_subgraphs):
    print("Dataset:", dataset)
    print("Number of anomalous subgraphs:", len(anomalous_subgraph))
