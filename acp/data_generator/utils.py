
import torch
import numpy as np


def get_ATUS(clusters, device=None):
    """Get the anchors, targets, unsassigned and assigned indices at each of the k steps during training.

    Args:
        clusters: list[int] -- sizes of all clusters

    Returns:
        anchors: Tensor(k) -- indices of anchors at each k.
        targets: list[Tensor(N-len(assigned))] -- each tensor is the prediction target at each 
            training step k, i.e. a binary mask of points in cluster k except 
            the anchor point. Shortened to start at the anchor and excluded already 
            assigned points.
        unassigned: list[Tensor(N)] -- each tensor is the indices of unassigned
            points at step k.
        assigned: list[Tensor(N)] -- each tensor is the indices of assigned points
            in cluster k. assigned[k-1] is the indices of the last assigned points 
            at training step k.
    """
    assert(torch.all(clusters > 0))
    N = torch.sum(clusters)
    K = len(clusters)
    clusters = clusters.to(device)

    anchors = torch.zeros(K, dtype=torch.int32, device=device)
    targets = []
    unassigned = []
    assigned = []

    cumsum = torch.cumsum(torch.cat([torch.LongTensor([0]).to(device), clusters]), dim=0)

    for k in range(K):
        anchors[k] = cumsum[k]
        available = torch.arange(cumsum[k]+1, N).to(device)
        unassigned.append(available)
        # indices of the elements in cluster k
        assigned_in_k = torch.arange(cumsum[k], cumsum[k+1]).to(device)
        assigned.append(assigned_in_k)
        # (shortened) binary masks of elements in cluster k except the anchor
        target_k = torch.zeros(len(available)).to(device)
        target_k[:len(assigned_in_k)-1] = 1
        targets.append(target_k)

    return anchors, targets, unassigned, assigned


def get_ATUS_batch(batch_clusters, device=None):
    """Get the anchors, targets, unsassigned and assigned indices at each of the k steps during training.
        This is the mini-batch version of get_ATUS

    Args:
        clusters: list[int] -- sizes of all clusters

    Returns:
        anchors: Tensor(b, max_K, max_N) -- binary mask of anchor indices
        targets: Tensor(b, max_K, max_N) -- each tensor along dim=1 is the prediction target at each 
            training step k, i.e. a binary mask of points in cluster k except 
            the anchor point.
        unassigned: Tensor(b, max_K, max_N) -- each tensor is the binary mask of unassigned
            points at step k.
        assigned: Tensor(b, max_K, max_N) -- each tensor is the binary mask of assigned points
            in cluster k. assigned[k-1] is the binary mask of the last assigned points 
            at training step k.
    """
    batch_size = len(batch_clusters)
    max_N = max([torch.sum(clusters) for clusters in batch_clusters])
    max_K = max([len(clusters) for clusters in batch_clusters])
    batch_anchors, batch_targets, batch_unassigned, batch_assigned = \
        [torch.zeros(batch_size, max_K, max_N, dtype=torch.bool, device=device)
            for _ in range(4)]

    for b, clusters in enumerate(batch_clusters):
        N = torch.sum(clusters)
        K = len(clusters)
        cumsum = torch.cumsum(
            torch.cat([torch.LongTensor([0]).to(device), clusters.to(device)]), dim=0)
        for k in range(K):
            batch_anchors[b, k, cumsum[k]] = 1
            batch_targets[b, k, cumsum[k]+1:cumsum[k+1]] = 1
            batch_unassigned[b, k, cumsum[k]+1:N] = 1
            batch_assigned[b, k, cumsum[k]:cumsum[k+1]] = 1
    return batch_anchors, batch_targets, batch_unassigned, batch_assigned


def relabel(labels):
    """Relabel the cluster labels so that they appear in order
    """
    labels = labels.copy()
    d = {}
    k = 0
    for i in range(len(labels)):
        j = labels[i]
        if j not in d:
            d[j] = k
            k += 1
        labels[i] = d[j]

    return labels

def remap_labels_by_cluster_size(labels):
    classes_to, counts = torch.unique(labels, return_counts=True, sorted=True)
    classes_from = classes_to[counts.argsort(descending=True)]
    
    indices = (classes_from.unsqueeze(1) == labels)

    new_labels = torch.zeros(len(labels)).to(labels.dtype).to(labels.device)
    for i, cl in enumerate(classes_to):
        new_labels[indices[i]] = cl
    return new_labels
