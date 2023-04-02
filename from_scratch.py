import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score, average_precision_score

class Encoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, hidden_dim)
        self.conv_logvar = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

def main():
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(Encoder(dataset.num_features, 16)).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        z, _, _ = model(data.x, data.edge_index)
        loss = model.recon_loss(data.edge_index, z) + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

    # Anomaly detection
    edge_scores = model.decode_all(z)
    auc_score = roc_auc_score(data.test_pos_edge_label, edge_scores[data.test_pos_edge_index].detach().cpu())
    ap_score = average_precision_score(data.test_pos_edge_label, edge_scores[data.test_pos_edge_index].detach().cpu())

    print(f'AUC Score: {auc_score:.4f}, Average Precision Score: {ap_score:.4f}')

if __name__ == '__main__':
    main()
