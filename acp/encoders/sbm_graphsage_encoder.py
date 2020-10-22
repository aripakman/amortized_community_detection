import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm


def get_sbm_graph_sage_encoder(params):
    return SBM_GraphSageEncoder(
        in_dim=params['enc_in_dim'],
        out_dim=params['enc_out_dim'],
        hidden_dim=params['enc_hidden_dim'],
        layers=params['enc_layers'])


class SBM_GraphSageEncoder(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, layers, dropout=0):
        super().__init__()
        assert layers >= 2
        self.dropout = dropout
        self.conv_layers = nn.ModuleList(
            [SAGEConv(in_dim, hidden_dim, normalize=True)])
        for _ in range(layers - 2):
            self.conv_layers.append(
                SAGEConv(hidden_dim, hidden_dim, normalize=True))
        self.conv_layers.append(SAGEConv(hidden_dim, out_dim, normalize=True))
        self.bn_layers = nn.ModuleList(
            [BatchNorm(hidden_dim) for _ in range(layers-1)])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x, edge_index)
            x = self.bn_layers[i](x)
            x = x.relu()
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_layers[-1](x, edge_index)
        return x
