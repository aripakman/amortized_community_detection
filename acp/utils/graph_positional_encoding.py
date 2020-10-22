
import dgl
import torch
import torch.nn as nn
from scipy import sparse
import numpy as np

def laplacian_positional_encoding(adj_matrix, pos_enc_dim):
    return scipy_laplacian_positional_encoding(adj_matrix, pos_enc_dim, tol=1e-2)
    # return numpy_laplacian_positional_encoding(adj_matrix, pos_enc_dim)  # this might be more stable

def scipy_laplacian_positional_encoding(adj_matrix, pos_enc_dim, tol=1e-2):
    """Graph positional encoding with Laplacian eigenvectors
    
    Adapted from https://github.com/graphdeeplearning/benchmarking-gnns

    Args:
        adj_matrix: 2d numpy matrix
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    # Laplacian
    A = adj_matrix.astype(np.float)
    N = sparse.diags(A.sum(0).clip(1) ** -0.5, dtype=np.float)
    L = sparse.eye(A.shape[0]) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sparse.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sparse.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=tol) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    return torch.from_numpy(EigVec[:,1:pos_enc_dim+1].real).float()


def numpy_laplacian_positional_encoding(adj_matrix, pos_enc_dim):
    """Graph positional encoding v/ Laplacian eigenvectors
    
    Adapted from https://github.com/graphdeeplearning/benchmarking-gnns
    
    Args:
        adj_matrix: 2d numpy matrix
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()
    # Laplacian
    A = adj_matrix.astype(np.float)
    N = sparse.diags(A.sum(0).clip(1) ** -0.5, dtype=np.float)
    L = sparse.eye(A.shape[0]) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(np.array(L))
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    return torch.from_numpy(EigVec[:,1:pos_enc_dim+1] * np.random.choice([-1, 1])).float()


