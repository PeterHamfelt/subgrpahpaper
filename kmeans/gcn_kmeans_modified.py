import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import pandas as  pd

# Load the Cora dataset
dataset = Planetoid(root='/home/victor/pytorch_geometric/graphgym/datasets/Cora/', name='Cora')
data = dataset[0]
labels=data.y
print(labels)

num_classes = dataset.num_classes

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

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model.conv2(model.conv1(data.x, data.edge_index), data.edge_index)



n_clusters=3
# Compute centroids using kmeans++
#centroids = kmeans_plus_plus(embeddings, K=7, N=len(embeddings))
# Compute centroids using d2_weighted_kmeans_plus_plus
centroids = d2_weighted_kmeans_plus_plus(embeddings, K=7)

# Perform K-means clustering on the embeddings
#kmeans = KMeans(n_clusters=7, init=centroids, n_init=1).fit(embeddings)




def run_clustering(embeddings, centroids, true_labels, n_clusters=7):
    results = {}
    
    """
    print("************************")
    print("centroids=",centroids)
    print("************************")
    """
    
    
    
    
    
    # Run proposed K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(embeddings)
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil_score = silhouette_score(embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    
    
    """
    print("************************")
    print("centroids=",centroids)
    print("************************")
    """
    
    
    
    
    
    # Run original K-means
    original_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1).fit(embeddings)
    original_sse = original_kmeans.inertia_
    original_sil_score = silhouette_score(embeddings.numpy(), original_kmeans.labels_)
    original_ari = adjusted_rand_score(true_labels.numpy(), original_kmeans.labels_)
    
    # Store results in a dictionary
    results['proposed_sse'] = proposed_sse
    results['original_sse'] = original_sse
    results['proposed_silhouette'] = proposed_sil_score
    results['original_silhouette'] = original_sil_score
    results['proposed_ari'] = proposed_ari
    results['original_ari'] = original_ari
    
    return results

n_runs = 10
results_list = []

for i in range(n_runs):
    centroids = d2_weighted_kmeans_plus_plus(embeddings, K=7)
    result = run_clustering(embeddings, centroids, data.y)
    results_list.append(result)

# Convert the results_list to a pandas DataFrame and print the table
results_df = pd.DataFrame(results_list)
print(results_df)
mean_results = results_df.mean()
print(mean_results)












