import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from ..data_generator.utils import get_ATUS, get_ATUS_batch


class CCP_BaseModel(nn.Module):

    # the conditional prior
    def get_pz(self, anchor, U, G):

        mu_logstd = self.pz_mu_log_sigma(torch.cat((anchor, U, G), dim=1))
        mu = mu_logstd[:, :self.z_dim]
        log_sigma = mu_logstd[:, self.z_dim:]

        return mu, log_sigma

    # the conditional posterior
    def get_qz(self, anchor, U_in, U_out, G):

        mu_logstd = self.qz_mu_log_sigma(
            torch.cat((anchor, U_in, U_out, G), dim=1))

        mu = mu_logstd[:, :self.z_dim]
        log_sigma = mu_logstd[:, self.z_dim:]

        return mu, log_sigma

    # this function is not used during training, it is auxiliary for samplers
    def sample_z(self, anchor, U, G):

        mu, log_sigma = self.get_pz(anchor, U, G)
        std = log_sigma.exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward_train(self, data, labels, w):
        """"Forward training iteration. Labels are shared across the mini-batch.
            Note: data and labels do not need to be pre-sorted. They will be sorted after encoder.
        Args:
            Data: torch.Tensor | torch_geometric.data.Batch | batched dgl.DGLGraph
                The data format needs to be consistent with what the encoder takes as input.
                The encoder output will always be a torch.Tensor of shape (batch_size, N, e_dim)
            labels: torch.Tensor of shape (N,)
            w: the number of VAE samples
        Return:
            loss, elbo
        """
        device = data.device
        N = len(labels)

        _, cluster_counts = torch.unique(
            labels, sorted=True, return_counts=True)
        K = len(cluster_counts)
        all_anchors, all_targets, all_unassigned, all_last_assigned = get_ATUS(
            cluster_counts, device=device)
        loss = 0
        elbo = 0

        # iterate k times and compute final loss
        enc_data = self.encoder(data).view([-1, N, self.e_dim])

        # sort the nodes by labels for CCP
        sorted_ind = torch.argsort(labels)
        labels = labels[sorted_ind]
        enc_data = enc_data[:, sorted_ind]

        G = None
        for k in range(K):
            # when the last cluster has only one element
            if k == K-1 and len(all_unassigned[k]) == 0:
                break

            ind_anchor = all_anchors[k]
            ind_unassigned = all_unassigned[k]
            ind_last_assigned = all_last_assigned[k-1] if k > 0 else None
            targets = all_targets[k].to(device)

            loss_term, elbo_term, G, _ = self.forward_k(
                k, enc_data, G, ind_anchor, ind_unassigned, ind_last_assigned, w, targets)
            loss += loss_term
            elbo += elbo_term
        return loss, elbo

    def forward_train_mask(self, data, batch_labels, w):
        """"Forward training iteration using masks. Labels are different across the mini-batch.
            Note: data and labels do not need to be pre-sorted. They will be sorted after encoder.
        Args:
            Data: torch.Tensor | torch_geometric.data.Batch | batched dgl.DGLGraph
                The data format needs to be consistent with what the encoder takes as input.
                The encoder output will always be a torch.Tensor of shape (batch_size, N, e_dim)
            labels: List[torch.Tensor]
            w: the number of VAE samples
        Return:
            loss, elbo
        """
        device = data.device
        batch_size = len(batch_labels)

        Ns = []
        Ks = []
        max_N = 0
        max_K = 0
        batch_cluster_counts = []

        sorted_ind_batch = []
        sorted_batch_labels = []
        for i, labels in enumerate(batch_labels):
            sorted_ind = torch.argsort(labels)
            labels = labels[sorted_ind]
            sorted_ind_batch.append(sorted_ind)
            sorted_batch_labels.append(labels)
        batch_labels = sorted_batch_labels

        for labels in batch_labels:
            Ns.append(len(labels))
            max_N = max(len(labels), max_N)

            _, cluster_counts = torch.unique(
                labels, sorted=True, return_counts=True)
            Ks.append(len(cluster_counts))
            max_K = max(len(cluster_counts), max_K)
            batch_cluster_counts.append(cluster_counts)

        # print(max_K, max_N)
        batch_label_tensor = torch.zeros(batch_size, max_N) - 1
        # -1 is used for padding
        for i, labels in enumerate(batch_labels):
            batch_label_tensor[i, :Ns[i]] = labels

        batch_anchors, batch_targets, batch_unassigned, batch_assigned = \
            get_ATUS_batch(batch_cluster_counts, device=device)
        # (batch_size, max_K, max_N)

        enc_data_raw = self.encoder(data)  # [total_N, e_dim]
        enc_data_raw = enc_data_raw.split(Ns, dim=0)  # split the batch

        # stack the encoded data into the same batch tensor with padding
        enc_data = torch.zeros(batch_size, max_N, self.e_dim).to(device)
        # the mask of indices corresponding to actual data in each batch
        mask = torch.zeros([batch_size, max_N]).to(device)
        for i in range(batch_size):
            sorted_ind = sorted_ind_batch[i]
            enc_data[i, :Ns[i]] = enc_data_raw[i][sorted_ind]
            mask[i, :Ns[i]] = 1

        # the number of unfinished training examples in each mini-batch
        t = batch_size
        G = None

        loss = 0
        elbo = 0
        # iterate k times and compute final loss
        for k in range(max_K):

            i = 0
            while i < t:
                # when no cluster left or the last cluster has only one element
                finished = (k == Ks[i]) \
                    or (k == Ks[i] - 1 and len(batch_unassigned[i][k]) == 0)
                if finished:
                    new_order = torch.cat(
                        [torch.arange(i), torch.arange(i+1, t)])
                    enc_data = enc_data[new_order, :]
                    mask = mask[new_order, :]
                    G = G[new_order, :]
                    Ks = Ks[new_order]
                    Ns = Ns[new_order]
                    batch_anchors = batch_anchors[new_order]
                    batch_targets = batch_targets[new_order]
                    batch_unassigned = batch_unassigned[new_order]
                    batch_assigned = batch_assigned[new_order]
                    t -= 1  # i stays the same
                else:
                    i += 1
                # all finished
                if t == 0:
                    break

            anchor = batch_anchors[:, k]
            unassigned = batch_unassigned[:, k]
            last_assigned = batch_assigned[:, k-1] if k > 0 else None
            targets = batch_targets[:, k].to(data.device)

            loss_term, elbo_term, G, _ = self.forward_k_mask(
                k, enc_data, G, anchor, unassigned, last_assigned, w, mask, targets)

            loss += loss_term
            elbo += elbo_term

        return loss, elbo

    def forward_k(self, k, enc_data, G, ind_anchor, ind_unassigned, ind_last_assigned,
                  w, targets=None):
        """
        k: the cluster number to sample in this call
        data: [batch_size, N, ...]
        anchor: the anchor point
        ind_unassigned: numpy integer array with the indices of the unassigned points
        ind_in_clusters: list, where ind_in_clusters[k] is a numpy integer array with the indices of the points in cluster k. 
        w: number of VAE samples

        Output:
            logits for sampling the binary variables to join cluster k   
        """
        # print(k)

        G = self.update_global(enc_data, ind_last_assigned, G, k)

        anch, data_unassigned, us_unassigned, U = self.encode_unassigned(
            enc_data, ind_anchor, ind_unassigned)

        pz_mu, pz_log_sigma = self.get_pz(anch, U, G)

        if targets is None:
            z = pz_mu  # + torch.randn_like(pz_mu)*torch.exp(pz_log_sigma)
            z = z.unsqueeze(0)
            logits = self.vae_likelihood(z, U, G, anch, data_unassigned)
            return logits.squeeze(-1)

        qz_mu, qz_log_sigma, z = self.conditional_posterior(
            us_unassigned, G, anch, targets, w)

        logits = self.vae_likelihood(z, U, G, anch, data_unassigned)

        loss, elbo = self.kl_loss(
            qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets)

        return loss, elbo, G, logits

    def forward_k_mask(self, k, enc_data, G, anchor, unassigned, last_assigned,
                       w, mask, targets=None):
        """
        k: the cluster number to sample in this call
        data: [batch_size, N, ...]
        anchor: the anchor point
        unassigned: numpy integer array with the indices of the unassigned points
        assigned: list, where assigned[k] is a numpy integer array with the indices of the points in cluster k. 
        w: number of VAE samples

        Output:
            logits for sampling the binary variables to join cluster k   
        """
        # print(k)

        G = self.update_global_mask(enc_data, last_assigned, G, k)

        anch, data_unassigned, us_unassigned, U = self.encode_unassigned_mask(
            enc_data, anchor, unassigned)

        pz_mu, pz_log_sigma = self.get_pz(anch, U, G)

        if targets is None:
            z = pz_mu  # + torch.randn_like(pz_mu)*torch.exp(pz_log_sigma)
            z = z.unsqueeze(0)
            logits = self.vae_likelihood(z, U, G, anch, data_unassigned)
            return logits.squeeze(-1)

        qz_mu, qz_log_sigma, z = self.conditional_posterior_mask(
            us_unassigned, G, anch, targets, unassigned, w)

        logits = self.vae_likelihood(z, U, G, anch, data_unassigned)

        loss, elbo = self.kl_loss_mask(
            qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets, unassigned)

        return loss, elbo, G, logits