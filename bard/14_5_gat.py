import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

def main():
    # Load dataset
    dataset = Planetoid(root='data/Cora', name='Cora')
    graph = dataset[0]

    # Define labels and number of classes
    labels = graph.y
    num_classes = dataset.num_classes

    # Define model and optimizer
    gnn = GAT(num_features=graph.num_node_features, hidden_size=8, num_classes=num_classes)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

    # Train model
    train_gnn(gnn, graph, labels, optimizer)

def train_gnn(gnn, graph, labels, optimizer, num_epochs=200):
    gnn.train()

    # Train for num_epochs epochs
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        x = gnn(graph.x, graph.edge_index)
        loss = F.nll_loss(x[graph.train_mask], labels[graph.train_mask])

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    # Evaluate on test set
    gnn.eval()
    with torch.no_grad():
        x = gnn(graph.x, graph.edge_index)
        test_loss = F.nll_loss(x[graph.test_mask], labels[graph.test_mask])
        test_acc = (x.argmax(dim=-1)[graph.test_mask] == labels[graph.test_mask]).sum() / graph.test_mask.sum()
    print(f"Test Loss {test_loss:.4f}, Test Accuracy {test_acc:.4f}")


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, num_heads=8):
        super(GAT, self).__init__()

        self.conv1 = GATConv(num_features, hidden_size, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_size * num_heads, num_classes, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    main()
