import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
#from community import best_partition as louvain_partition
#from community import louvain_partition
from networkx.algorithms.community import louvain




import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import louvain

# Step 1: Load the social network dataset
G = nx.karate_club_graph()

# Step 2: Perform community detection using Greedy Modularity
greedy_communities = greedy_modularity_communities(G)

# Step 3: Perform community detection using Louvain
louvain_communities = community.best_partition(G)


# Rest of the code...







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
for nodes in louvain_partition.values():
    # Perform clustering and anomaly detection within each community

    # Step 5a: Ego-net analysis
    ego_net = G.subgraph(nodes)

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
accuracy_greedy = len(anomalous_subgraphs_greedy) / len(greedy_communities)
accuracy_louvain = len(anomalous_subgraphs_louvain) / len(louvain_partition)

# Print the results
# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
tp = len(set(anomalous_communities) & set(anomalous_subgraphs))
fp = len(anomalous_subgraphs) - tp
fn = len(anomalous_communities) - tp

# Calculate Precision, Recall, and F1-score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

# Calculate Accuracy
accuracy = tp / len(anomalous_communities)

# Print the performance metrics
print("Performance Comparison:")
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1-score: {:.2f}".format(f1_score))
print("Accuracy: {:.2f}".format(accuracy))
