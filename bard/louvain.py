import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
import community
from sklearn.ensemble import IsolationForest

# Step 1: Load the social network dataset
G = nx.karate_club_graph()

# Step 2: Perform community detection using a suitable algorithm
communities = greedy_modularity_communities(G)

# Step 3: Anomalous Subgraph Detection (Proposed Methodology)
anomalous_subgraphs = []
for community in communities:
    # Step 3a: Ego-net analysis
    ego_net = G.subgraph(community)

    # Step 3b: Non-overlapping clustering (K-means)
    kmeans = KMeans(n_clusters=2)
    node_features = nx.to_numpy_array(ego_net)
    clustering_results = kmeans.fit_predict(node_features)

    # Step 3c: Global graph partitioning (Louvain)
    #partition = community.best_partition(ego_net)
    partition = greedy_modularity_communities(ego_net)

    # Step 3d: Anomaly detection within clusters (Isolation Forest)
    anomaly_detector = IsolationForest(contamination=0.1)
    anomalous_nodes = []
    for cluster_id, nodes in enumerate(partition):
        if len(nodes) <= 1:
            continue
        cluster = [node for node in nodes]
        cluster_features = node_features[:, :len(cluster)]
        anomaly_scores = anomaly_detector.fit_predict(cluster_features)
        anomalous_nodes.extend([node for node, score in zip(cluster, anomaly_scores) if score == -1])

    # Step 3e: Store anomalous subgraphs
    if len(anomalous_nodes) > 0:
        anomalous_subgraphs.append(ego_net.subgraph(anomalous_nodes))

# Print the anomalous subgraphs
print("Anomalous Subgraphs:")
for subgraph in anomalous_subgraphs:
    print(subgraph.nodes())
