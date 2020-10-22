
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sbm_gated_gcn_dgl_encoder(params):
    return GatedGcnDglEncoder(net_params=params)


class GatedGcnDglEncoder(nn.Module):
    """Residual GatedGCN encoder
    Adapted from https://github.com/graphdeeplearning/benchmarking-gnns

    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf

    """

    def __init__(self, net_params):
        super().__init__()

        # onehot and/or dense input features
        in_dim_node = net_params['enc_in_dim']
        in_dim_edge = 1
        hidden_dim = net_params['enc_hidden_dim']
        out_dim = net_params.get('enc_out_dim', hidden_dim)
        n_layers = net_params['enc_layers']
        dropout = net_params['enc_dropout']

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([
            GatedGCNLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                batch_norm=True,
                residual=True)
            for _ in range(n_layers)])
        self.fc_out = None
        if out_dim != hidden_dim:
            self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, g):
        h = g.ndata['feat']
        e = g.edata['feat']

        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        # residual gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        if self.fc_out is not None:
            h = self.fc_out(h)

        return h


class GatedGCNLayer(nn.Module):
    """GatedGCN Layer
    From https://github.com/graphdeeplearning/benchmarking-gnns

    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf

    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / \
            (torch.sum(sigma_ij, dim=1) + 1e-6)
        return {'h': h}

    def forward(self, g, h, e):

        h_in = h
        e_in = e

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        e = g.edata['e']

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        h = F.relu(h)
        e = F.relu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels)
