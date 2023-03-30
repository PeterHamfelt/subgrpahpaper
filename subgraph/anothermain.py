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


# Load the dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Instantiate the GNN model and set the optimizer and loss function
in_channels = dataset.num_features
hidden_channels = 16
out_channels = 2

model = GNN(in_channels, hidden_channels, out_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CosineSimilarity()

# Train the model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    x, edge_index, y = data.x, data.edge_index, data.y
    subgraph_nodes = [1, 2, 3]
    # Example of extracting a subgraph of interest
    x_subgraph, edge_index_subgraph = extract_subgraph(x, edge_index, subgraph_nodes)
    out = model(x_subgraph, edge_index_subgraph)
    normal_representation = model(dataset[0].x, dataset[0].edge_index)  # Example of using the first graph in the dataset as normal representation
    anomaly_score = 1 - criterion(out, normal_representation)
    loss = anomaly_score.mean()
    loss.backward()

    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch {}, Loss {}'.format(epoch, loss))

# Test the model
# Test the model
model.eval()
correct=0
x, edge_index, y = data.x, data.edge_index, data.y
subgraph_nodes = [1, 2, 3]
#subgraph_nodes = subgraph.tolist()
y_subgraph = y[subgraph_nodes]
x_subgraph, edge_index_subgraph = extract_subgraph(x, edge_index, subgraph_nodes)
out = model(x_subgraph, edge_index_subgraph)
normal_representation = model(dataset[0].x, dataset[0].edge_index)  # Example of using the first graph in the dataset as normal representation
anomaly_score = 1 - criterion(out, normal_representation)
pred = (anomaly_score > 0.5).int()  # Threshold the anomaly score to classify the subgraph as anomalous or not
y_subgraph = y[subgraph_nodes]
"""
y_subgraph_binary = torch.zeros_like(pred)
y_subgraph_binary[y_subgraph] = 1
"""
num_classes = y_subgraph.max().item() + 1
y_subgraph_binary = torch.zeros_like(data.y)
y_subgraph_binary[subgraph_nodes] = 1
y_subgraph = y_subgraph_binary[subgraph_nodes].bool()


#correct = int(pred == y[subgraph_nodes])
#correct += (pred == y[subgraph_nodes]).sum().item()
pred = (anomaly_score > 0.5).flatten().bool()  # Flatten pred and y_subgraph
correct += (pred == y_subgraph).sum().item()

accuracy = correct / 1  # As we only have one graph in the dataset
print('Accuracy {:.2f}'.format(accuracy))