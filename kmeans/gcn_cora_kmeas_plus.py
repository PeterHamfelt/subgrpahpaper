import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

# Load the Cora dataset
dataset = Planetoid(root='/home/victor/pytorch_geometric/graphgym/datasets/Cora/', name='Cora')
data = dataset[0]
labels=data.y
print(labels)

num_classes = dataset.num_classes

# K-means++ algorithm
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
centroids = kmeans_plus_plus(embeddings, K=7, N=len(embeddings))


"""

# Perform K-means clustering on the embeddings
#centroids = kmeans_plus_plus(embeddings, 7, len(embeddings))
kmeans = KMeans(n_clusters=7, init='centroid', random_state=None).fit(embeddings)

# Print the cluster labels for the first 10 nodes
print(kmeans.labels_[:10])

# Compute the Silhouette coefficient for the clustering result
silhouette_score = silhouette_score(embeddings.numpy(), kmeans.labels_)
print("Silhouette coefficient: {:.4f}".format(silhouette_score))

# Compute the ARI between the true class labels and the predicted cluster labels
ari = adjusted_rand_score(data.y.numpy(), kmeans.labels_)
print("Adjusted Rand index: {:.4f}".format(ari))
"""










n_runs =10
sse_values = []



for i in range(n_runs):
    kmeans = KMeans(n_clusters=7, init='random').fit(embeddings)
    #kmeans.fit(X)
    sse = kmeans.inertia_
    sse_values.append(sse)
    print(f"Run {i+1}: SSE = {sse:.4f}")
print("sse_values:",set(sse_values))
# Check if there are differences in SSE values
if len(set(sse_values)) > 1:
    print("\nThe SSE values differ across the runs for original_kmeans.")
else:
    print("\nThe SSE values are the same across the runs for original_kmeans.")  


# Compute the Silhouette coefficient for the clustering result
sil_score = silhouette_score(embeddings.numpy(), kmeans.labels_)
print("Silhouette coefficient: {:.4f}".format(sil_score))

# Compute the ARI between the true class labels and the predicted cluster labels
ari = adjusted_rand_score(data.y.numpy(), kmeans.labels_)
print("Adjusted Rand index for original: {:.4f}".format(ari))

















for i in range(n_runs):
    #kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1)
    kmeans = KMeans(n_clusters=7, init=centroids).fit(embeddings)
    #kmeans.fit(X)
    sse = kmeans.inertia_
    sse_values.append(sse)
    print(f"Run {i+1}: SSE = {sse:.4f}")
    
print("sse_values:",set(sse_values))

# Check if there are differences in SSE values
if len(set(sse_values)) > 1:
    print("\nThe SSE values differ across the runs for proposed kmeans.")
else:
    print("\nThe SSE values are the same across the runs for proposed kmeans.")

# Compute the Silhouette coefficient for the clustering result
sil_score = silhouette_score(embeddings.numpy(), kmeans.labels_)
print("Silhouette coefficient: {:.4f}".format(sil_score))

# Compute the ARI between the true class labels and the predicted cluster labels
ari = adjusted_rand_score(data.y.numpy(), kmeans.labels_)
print("Adjusted Rand index for proposed: {:.4f}".format(ari))


#azure data factory
#data bricks


    
    
    
    

