"""
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
    
    
    print("************************")
    print("centroids=",centroids)
    print("************************")
  
    
    
    
    
    
    # Run proposed K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(embeddings)
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil_score = silhouette_score(embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    
    
  
    print("************************")
    print("centroids=",centroids)
    print("************************")
    
    
    
    
    
    
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
import pandas as  pd

# Load the Cora dataset
dataset = Planetoid(root='/home/victor/pytorch_geometric/graphgym/datasets/Cora/', name='Cora')
data = dataset[0]
labels = data.y
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

# ... (same as before) ...

# Define the GCN model
# ... (same as before) ...
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

# Train the model with early stopping
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()
model.train()

patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Compute validation loss
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Training stopped after {epoch} epochs.")
            break

# Get node embeddings from the trained model
with torch.no_grad():
    embeddings = model.conv2(model.conv1(data.x, data.edge_index), data.edge_index)
# Get node embeddings from the trained model

# Normalize embeddings
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

n_clusters = 7

# Compute centroids using kmeans++
# ... (same as before) ...
# Compute centroids using d2_weighted_kmeans_plus_plus
centroids = d2_weighted_kmeans_plus_plus(normalized_embeddings, K=7)

# Compute centroids using d2_weighted_kmeans_plus_plus
#centroids = d2_weighted_kmeans_plus_plus(embeddings, K=7)

# Perform K-means clustering on the embeddings
# ... (same as before) ...
def run_clustering(normalized_embeddings, centroids, true_labels, n_clusters=7):
    results = {}
    
    """
    print("************************")
    print("centroids=",centroids)
    print("************************")
    """
    
    
    
    
    
    # Run proposed K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=1).fit(normalized_embeddings)
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil_score = silhouette_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    proposed_ch_score = calinski_harabasz_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_db_score = davies_bouldin_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_mi_score = adjusted_mutual_info_score(true_labels.numpy(), proposed_kmeans.labels_)
    
    
    
    
    
    
    
    # Run original K-means
    original_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1).fit(normalized_embeddings)
    original_sse = original_kmeans.inertia_
    original_sil_score = silhouette_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_ari = adjusted_rand_score(true_labels.numpy(), original_kmeans.labels_)
    original_ch_score = calinski_harabasz_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_db_score = davies_bouldin_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_mi_score = adjusted_mutual_info_score(true_labels.numpy(), original_kmeans.labels_)




    
    # Store results in a dictionary
    results['proposed_sse'] = proposed_sse
    results['original_sse'] = original_sse
    results['proposed_silhouette'] = proposed_sil_score
    results['original_silhouette'] = original_sil_score
    results['proposed_ari'] = proposed_ari
    results['original_ari'] = original_ari
    results['proposed_ch_score'] = proposed_ch_score
    results['original_ch_score'] = original_ch_score
    results['proposed_db_score'] = proposed_db_score
    results['original_db_score'] = original_db_score
    results['proposed_mi_score'] = proposed_mi_score
    results['original_mi_score'] = original_mi_score
    return results



p1=pd.DataFrame()
for i in range(10):
    #columns = ['proposed_sse', 'original_sse', 'proposed_silhouette', 'original_silhouette', 'proposed_ari', 'original_ari', 'proposed_ch_score', 'original_ch_score', 'proposed_db_score', 'original_db_score', 'proposed_mi_score', 'original_mi_score', 'proposed_sse_percentage_difference', 'proposed_silhouette_percentage_difference', 'proposed_ari_percentage_difference', 'proposed_ch_score_percentage_difference', 'proposed_db_score_percentage_difference', 'proposed_mi_score_percentage_difference']
    #def run_clustering(embeddings, centroids, true_labels, n_clusters=7):
        # ... (same as before) ...
    #final_table = pd.DataFrame(columns=columns)
    n_runs = 10
    results_list = []

    for i in range(n_runs):
        centroids = d2_weighted_kmeans_plus_plus(normalized_embeddings, K=7)
        result = run_clustering(normalized_embeddings, centroids, data.y)
        results_list.append(result)

    # Convert the results_list to a pandas DataFrame and print the table
    results_df = pd.DataFrame(results_list)
    #print(results_df)
    mean_results = results_df.mean()
    #print(mean_results.index)
    #print((mean_results))
    percent_diff_sse=((mean_results['proposed_sse']-mean_results['original_sse'])/mean_results['original_sse'])*100
    percent_diff_silhouette=((mean_results['proposed_silhouette']-mean_results['original_silhouette'])/mean_results['original_silhouette'])*100
    percent_diff_ari=((mean_results['proposed_ari']-mean_results['original_ari'])/mean_results['original_ari'])*100
    percent_diff_ch_score=((mean_results['proposed_ch_score']-mean_results['original_ch_score'])/mean_results['original_ch_score'])*100
    percent_diff_db_score=((mean_results['proposed_db_score']-mean_results['original_db_score'])/mean_results['original_db_score'])*100
    percent_diff_mi_score=((mean_results['proposed_mi_score']-mean_results['original_mi_score'])/mean_results['original_mi_score'])*100

    dict={'sse':[],'silohette':[],'ari':[],'ch_score':[],'db_score':[],'mi_score':[]}
    #dict={'sse':percent_diff_sse,'silohette':percent_diff_silhouette,'ari':percent_diff_ari,'ch_score':percent_diff_ch_score,'db_score':percent_diff_db_score,'mi_score':percent_diff_mi_score}
    dict['sse'].append(percent_diff_sse)

    dict['silohette'].append(percent_diff_silhouette)
    dict['ari'].append(percent_diff_ari)
    dict['ch_score'].append(percent_diff_ch_score)
    dict['db_score'].append(percent_diff_db_score)
    dict['mi_score'].append(percent_diff_mi_score)

    #print(dict)
    p2=pd.DataFrame(dict)
    p1=p1.append (p2)

print(p1)
"""
    percentage_differences = {}
    for key in mean_results.keys():
        if 'original' in key:
            proposed_key = key.replace('original', 'proposed')
            percentage_difference = ((mean_results[proposed_key] - mean_results[key]) / mean_results[key]) * 100
            percentage_differences[key.replace('original', '')] = percentage_difference

    final_table.loc[i] = {**mean_results, **percentage_differences}
print(final_table)
"""






