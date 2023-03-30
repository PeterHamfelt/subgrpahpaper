# Import necessary libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch


# Define the Graph Neural Network (GNN) model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 16, heads=8, dropout=0.6)
        self.conv2 = GATConv(16*8, 2, heads=1, concat=False, dropout=0.6)


    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x, _ = to_dense_batch(x, edge_index, fill_value=0)
        x = F.elu(self.conv2(x, None))
        return x.mean(dim=0)


# Load the dataset
dataset = YourDataset(root='/path/to/your/dataset')


# Split the dataset into training and testing subsets
train_loader = DataLoader(dataset[:800], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset[800:], batch_size=64, shuffle=False)


# Instantiate the GNN model and set the optimizer and loss function
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CosineSimilarity()


# Train the model
model.train()
for epoch in range(200):
    loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        x, edge_index, y = data.x, data.edge_index, data.y
        subgraph = torch.tensor([1, 2, 3])  # Example of extracting a subgraph of interest
        x_subgraph, edge_index_subgraph = x[subgraph], edge_index[:, subgraph]
        out = model(x_subgraph, edge_index_subgraph)
        normal_representation = model(dataset[0].x, dataset[0].edge_index)  # Example of using the first graph in the dataset as normal representation
        anomaly_score = 1 - criterion(out, normal_representation)
        loss += anomaly_score
        anomaly_score.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Epoch {}, Loss {}'.format(epoch, loss))


# Test the model
model.eval()
correct = 0
for data in test_loader:
    x, edge_index, y = data.x, data.edge_index, data.y
    subgraph = torch.tensor([1, 2, 3])  # Example of extracting a subgraph of interest
    x_subgraph, edge_index_subgraph = x[subgraph], edge_index[:, subgraph]
    out = model(x_subgraph, edge_index_subgraph)
    normal_representation = model(dataset[0].x, dataset[0].edge_index)  # Example of using the first graph in the dataset as normal representation
    anomaly_score = 1 - criterion(out, normal_representation)
    pred = int(anomaly_score > 0.5)  # Threshold the anomaly score to classify the subgraph as anomalous or not
    correct += int(pred == y[subgraph])
accuracy = correct / len(dataset[800:])
print('Accuracy {:.2f}'.format(accuracy))
