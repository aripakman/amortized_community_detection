import torch
import numpy as np
from torch_geometric.data import Data as GeomData
from torch_geometric.data import Batch as GeomBatch
from ..data_generator.utils import relabel
import dgl

def shuffle_adj_matrix_batch_and_labels(adj_matrix, labels):
    # shuffle the assignment order
    arr = np.arange(len(labels))
    np.random.shuffle(arr)
    labels = labels[arr]

    adj_matrix = adj_matrix[:, arr, :]
    adj_matrix = adj_matrix[:, :, arr]

    # relabel cluster numbers so that they appear in order
    labels = relabel(labels)
    return adj_matrix, labels

def adj_matrix_to_edge_list(adj_matrix):
    return torch.nonzero(adj_matrix == 1, as_tuple=False).transpose(0,1)

def edge_list_to_adj_matrix(edge_index, N):
    adj_matrix = torch.zeros(N, N)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    return adj_matrix


def create_torch_geom_batch(adj_matrix_batch, node_features, device):
    batch_size = adj_matrix_batch.shape[0]
    batch = []
    for b in range(batch_size):
        geom_data = create_torch_geom_single_graph(
            adj_matrix_batch[b], node_features[b], device)
        batch.append(geom_data)
    batch = GeomBatch.from_data_list(batch)
    batch.device = device
    batch.shape = node_features.shape
    return batch

def create_torch_geom_single_graph(adj_matrix, node_features, device):
    edge_index = adj_matrix_to_edge_list(adj_matrix)
    geom_data = GeomData(x=node_features, edge_index=edge_index)
    return geom_data.to(device)

def create_dgl_batch(adj_matrix_batch, node_features, device):
    batch_size = adj_matrix_batch.shape[0]
    batch = []
    for b in range(batch_size):
        graph = create_dgl_single_graph(adj_matrix_batch[b], node_features[b], device=device)
        batch.append(graph)
    batch = dgl.batch(batch)
    batch.device = device
    return batch

def create_dgl_single_graph(adj_matrix, node_features, device):
    graph = dgl.DGLGraph()
    N = adj_matrix.shape[0]
    graph.add_nodes(N)
    edge_index = adj_matrix_to_edge_list(adj_matrix)
    graph.add_edges(edge_index[0], edge_index[1])
    graph.ndata['feat'] = node_features
    edge_feat_dim = 1
    graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)
    return graph.to(device)

def create_dgl_single_graph_from_edge_list(edge_index, node_features, device):
    graph = dgl.DGLGraph()
    N = edge_index.max() + 1
    graph.add_nodes(N)
    graph.add_edges(edge_index[0], edge_index[1])
    graph.ndata['feat'] = node_features
    edge_feat_dim = 1
    graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)
    return graph.to(device)

def adj_matrix_to_geom_data(adj_matrix, labels):
    edge_index = adj_matrix_to_edge_list(adj_matrix)
    geom_data = GeomData(edge_index=edge_index, y=labels)
    geom_data.adj_matrix = adj_matrix

def batch_geom_data_to_tensor(geom_data):
    return torch.stack([geom.x for geom in geom_data])