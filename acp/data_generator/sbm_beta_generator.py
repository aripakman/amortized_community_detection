import numpy as np
import torch
from .distributions import CRP_Generator
from ..utils.graph_utils import shuffle_adj_matrix_batch_and_labels
from ..utils.graph_utils import create_torch_geom_batch, create_dgl_batch
from ..utils.graph_utils import create_torch_geom_single_graph, create_dgl_single_graph
from ..utils.graph_positional_encoding import laplacian_positional_encoding


def get_sbm_beta_crp_generator(params):
    partition_generator = CRP_Generator(
        alpha=params['alpha'], maxK=params['maxK'])
    sbm_generator = SBM_BetaGenerator(
        params, partition_generator=partition_generator)
    return sbm_generator


class SBM_BetaGenerator():
    """Generate graphs using stochastic block model (SBM) parameterized by Beta distributions.
    """

    def __init__(self, params, partition_generator):
        self.partition_generator = partition_generator
        self.params = params
        self.random_embed_dim = params['random_embed_dim']
        self.pos_enc_dim = params['pos_enc_dim']

    def generate(self, N=None, batch_size=1):

        if N is None:
            N = np.random.randint(self.params['Nmin'], self.params['Nmax'])

        alpha0 = self.params['between_alpha']
        beta0 = self.params['between_beta']
        alpha1 = self.params['within_alpha']
        beta1 = self.params['within_beta']

        clusters = self.partition_generator.generate(N=N)
        assert(np.all(clusters > 0))
        N = np.sum(clusters)
        K = len(clusters)

        cumsum = np.cumsum(np.insert(clusters, 0, [0]))
        adj_matrix = np.zeros([batch_size, N, N])
        labels = np.empty(N, dtype=np.int32)

        for i in range(K):
            for j in range(i, K):

                if j == i:
                    p = np.random.beta(alpha1, beta1, batch_size).reshape(
                        [batch_size, 1])
                else:
                    p = np.random.beta(alpha0, beta0, batch_size).reshape(
                        [batch_size, 1])

                p = np.repeat(p, clusters[i]*clusters[j], axis=1).reshape(
                    [batch_size, clusters[i], clusters[j]])

                rands = np.random.rand(batch_size, clusters[i], clusters[j])
                adj_matrix[:, cumsum[i]:cumsum[i+1],
                           cumsum[j]:cumsum[j+1]] = (rands < p)

            labels[cumsum[i]:cumsum[i+1]] = i

        # make the matrix symmetric
        i_lower = np.tril_indices(N, -1)
        idd = (np.arange(N), np.arange(N))
        for b in range(batch_size):
            adj_matrix[b, :, :][i_lower] = adj_matrix[b, :, :].T[i_lower]
            adj_matrix[b, :, :][idd] = 0

        # shuffle the assignment order and relabel clusters so that they appear in order
        adj_matrix, labels = shuffle_adj_matrix_batch_and_labels(
            adj_matrix, labels)

        adj_matrix = torch.from_numpy(adj_matrix).float()
        labels = torch.from_numpy(labels).int()

        # node features
        node_features = self.build_node_features(adj_matrix)

        return adj_matrix, labels, node_features

    def build_node_features(self, adj_matrix):
        batch_size, N = adj_matrix.shape[0], adj_matrix.shape[1]

        node_features = []
        if self.random_embed_dim:
            node_features.append(
                torch.normal(0, 1, size=(batch_size, N, self.random_embed_dim)))

        if self.pos_enc_dim:
            pos_enc = torch.zeros(batch_size, N, self.pos_enc_dim)
            for b in range(batch_size):
                pos_enc[b] = laplacian_positional_encoding(
                    adj_matrix[b], self.pos_enc_dim)
            node_features.append(pos_enc)
        node_features = torch.cat(node_features, dim=-1)

        # append extra features if needed
        return node_features

    def generate_batch(self, batch_size, data_lib, device):
        # nodes are not sorted by labels, as we can sort the encoder output in the model
        adj_matrix, labels, node_features = self.generate(
            batch_size=batch_size)
        if data_lib == "torch_geom":
            batch = create_torch_geom_batch(adj_matrix, node_features, device)
        elif data_lib == "dgl":
            batch = create_dgl_batch(adj_matrix, node_features, device)
        else:
            raise ValueError("data_lib should be 'torch_geom' or 'dgl'")
        return batch, labels

    def generate_single(self, data_lib, device):
        adj_matrix, labels, node_features = self.generate(batch_size=1)
        shape = adj_matrix.shape
        adj_matrix, node_features = adj_matrix[0], node_features[0]

        if data_lib == "torch_geom":
            data = create_torch_geom_single_graph(
                adj_matrix, node_features, device)
        elif data_lib == "dgl":
            data = create_dgl_single_graph(adj_matrix, node_features, device)
        else:
            raise ValueError("data_lib should be 'torch_geom' or 'dgl'")

        labels = labels.to(device)
        data.shape = shape
        return data, labels
