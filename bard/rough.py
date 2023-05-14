"""
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import community
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

# Print the detected communities
print("Greedy Modularity Communities:")
for i, comm in enumerate(greedy_communities):
    print(f"Community {i+1}: {comm}")

print("Louvain Communities:")
for node, comm in louvain_communities.items():
    print(f"Node {node}: Community {comm}")
"""
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import community
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

# Create a list of communities
communities = [[] for _ in range(len(louvain_communities))]
for node, community in enumerate(louvain_communities):
    communities[community].append(node)

# Calculate the modularity of each community
greedy_modularity = nx.algorithms.community.modularity(G, greedy_communities)
louvain_modularity = nx.algorithms.community.modularity(G, communities)

# Print the results
print("Greedy Modularity:", greedy_modularity)
print("Louvain Modularity:", louvain_modularity)

# Plot the graphs
nx.draw(G, with_labels=True)
nx.draw_networkx_communities(G, greedy_communities, node_color='red', alpha=0.7)
nx.draw_networkx_communities(G, communities, node_color='blue', alpha=0.7)
