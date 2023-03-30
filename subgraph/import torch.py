import torch
print("Hi")
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.datasets import Planetoid

# Define the Graph Neural Network (GNN) model
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=2):
        super(GNN, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)
        
        self.lin1 = torch.nn.Linear(out_channels, hidden_channels)  # Add this line
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)  # Add this line

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        batch = torch.zeros(x.size(0), dtype=torch.long)  # Create a batch tensor indicating that all nodes belong to the same graph
        x, _ = to_dense_batch(x, batch, fill_value=0)
        x = global_mean_pool(x, batch)
        
        return x


# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')

dataset_length = len(dataset)
train_split = int(len(dataset) * 0.8)
train_dataset = dataset[:train_split]
train_loader = DataLoader(dataset[:train_split], batch_size=64, shuffle=True)
print('*******************')
print('dataset:',dataset)
print("dataset_length",dataset_length)
print('train_split:',train_split)
print("train_dataset:",train_dataset)
print("train_loader:",train_loader)
print('***************')



#train_split = int(dataset_length * 0.8)  # 80% of the dataset for training
test_split = dataset_length - train_split  # 20% of the dataset for testing

# Replace 800 with train_split

test_dataset = dataset[train_split:]


# Split the dataset into training and testing subsets

test_loader = DataLoader(dataset[test_split:], batch_size=64, shuffle=False)

# Instantiate the GNN model and set the optimizer and loss function
in_channels = dataset.num_features
hidden_channels = 16
out_channels = 2

model = GNN(in_channels, hidden_channels, out_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CosineSimilarity()


def extract_subgraph(x, edge_index, nodes):
    nodes = torch.tensor(nodes)
    mask = torch.zeros(x.size(0), dtype=torch.bool)
    mask[nodes] = True
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    edge_index_subgraph = edge_index[:, edge_mask]

    # Remap node indices
    node_mapping = torch.full((x.size(0),), -1, dtype=torch.long)
    node_mapping[nodes] = torch.arange(nodes.size(0), dtype=torch.long)
    edge_index_subgraph = node_mapping[edge_index_subgraph]

    x_subgraph = x[nodes]
    return x_subgraph, edge_index_subgraph

# Train the model
model.train()
for epoch in range(200):
    loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_index, y = data.x, data.edge_index, data.y
        subgraph_nodes = [1, 2, 3]
        # Example of extracting a subgraph of interest
        x_subgraph, edge_index_subgraph = extract_subgraph(x, edge_index, subgraph_nodes)
        out = model(x_subgraph, edge_index_subgraph)
        normal_representation = model(dataset[0].x, dataset[0].edge_index)  # Example of using the first graph in the dataset as normal representation
        anomaly_score = 1 - criterion(out, normal_representation)
        loss += anomaly_score
        anomaly_score_mean = anomaly_score.mean()
        anomaly_score_mean.backward()

        optimizer.step()
    if epoch % 10 == 0:
        print('Epoch {}, Loss {}'.format(epoch, loss))

# Test the model
# Test the model
model.eval()
correct = 0
for data in test_loader:
    x, edge_index, y = data.x, data.edge_index, data.y
    subgraph_nodes = [1, 2, 3]
    x_subgraph, edge_index_subgraph = extract_subgraph(x, edge_index, subgraph_nodes)
    out = model(x_subgraph, edge_index_subgraph)
    normal_representation = model(dataset[0].x, dataset[0].edge_index)  # Example of using the first graph in the dataset as normal representation
    anomaly_score = 1 - criterion(out, normal_representation)
    pred = int(anomaly_score > 0.5)  # Threshold the anomaly score to classify the subgraph as anomalous or not
    correct += int(pred == y[subgraph_nodes])
if len(dataset[800:]) > 0:
    accuracy = correct / len(dataset[800:])
else:
    accuracy = 0
    print("The dataset is empty or has fewer than 800 elements.")

accuracy = correct / len(dataset[800:])
print('Accuracy {:.2f}'.format(accuracy))


