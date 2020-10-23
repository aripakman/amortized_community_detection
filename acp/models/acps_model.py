
import torch
import torch.nn as nn
import torch.distributions as dist
from ..data_generator.utils import get_ATUS, get_ATUS_batch
from ..modules.attention import ISAB, PMA


class ACP_Model(nn.Module):
    """The ACP-S model
    """

    def __init__(self, params, encoder):

        super().__init__()

        self.params = params
        self.encoder = encoder
        self.previous_k = -1

        self.g_dim = params['g_dim']
        self.h_dim = params['h_dim']
        self.u_dim = params['h_dim']
        self.z_dim = params['z_dim']
        self.e_dim = params['e_dim']

        H = params['H_dim']

        self.use_attn = self.params['use_attn']

        if self.use_attn:
            self.n_heads = params['n_heads']
            self.n_inds = params['n_inds']

            self.isab_enc = ISAB(dim_X=self.e_dim, dim=self.e_dim, num_inds=self.n_inds,
                                 num_heads=params['n_heads'], ln=True)

            self.pma_h = PMA(dim_X=self.h_dim, dim=self.h_dim,
                             num_inds=1, num_heads=params['n_heads'])
            self.pma_u = PMA(dim_X=self.u_dim, dim=self.u_dim,
                             num_inds=1, num_heads=params['n_heads'])
            self.pma_u_in = PMA(dim_X=self.u_dim, dim=self.u_dim,
                                num_inds=1, num_heads=params['n_heads'])
            self.pma_u_out = PMA(
                dim_X=self.u_dim, dim=self.u_dim, num_inds=1, num_heads=params['n_heads'])

        self.u = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.u_dim),
        )

        self.h = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.h_dim),
        )

        self.g = torch.nn.Sequential(
            torch.nn.Linear(self.h_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, self.g_dim),
        )

        self.phi_input_dim = self.e_dim + self.z_dim + \
            self.e_dim + self.u_dim + self.g_dim

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(self.phi_input_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 1, bias=False),
        )

        self.pz_mu_log_sigma = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim + self.u_dim + self.g_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 2*self.z_dim),
        )

        self.qz_mu_log_sigma = torch.nn.Sequential(
            torch.nn.Linear(self.e_dim + 2*self.u_dim + self.g_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 2*self.z_dim),
        )

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
        Return:
            loss, elbo
        """
        device = data.device
        N = len(labels)
        # no longer need input data to be sorted. Now we sort the encoder output.
        # assert (torch.all(torch.sort(labels)[0] == labels)), "data and labels need to be sorted"
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

        if self.use_attn:
            enc_data = self.isab_enc(enc_data)

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
        enc_data = self.isab_enc(enc_data, mask)

        Ks = torch.tensor(Ks)
        Ns = torch.tensor(Ns)

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

    def update_global(self, enc_data, ind_last_assigned, G, k):

        if k == 0:
            G = torch.zeros([enc_data.shape[0], self.g_dim]
                            ).to(enc_data.device)
        else:
            hs_last_cluster = self.h(enc_data[:, ind_last_assigned, :])
            if self.use_attn:
                G += self.g(self.pma_h(hs_last_cluster).squeeze(dim=1))
            else:
                G += self.g(hs_last_cluster.mean(dim=1))
        return G

    def update_global_mask(self, enc_data, last_assigned, G, k):

        if k == 0:
            G = torch.zeros([enc_data.shape[0], self.g_dim]
                            ).to(enc_data.device)
        else:
            hs = self.h(enc_data)
            if self.use_attn:
                hs_mean = self.pma_h(hs, last_assigned).squeeze(dim=1)
            else:
                hs_mean = self.masked_mean(hs, last_assigned)
            G += self.g(hs_mean)
        return G

    def encode_unassigned(self, enc_data, ind_anchor, ind_unassigned):
        anch = enc_data[:, ind_anchor, :]
        data_unassigned = enc_data[:, ind_unassigned, :]
        us_unassigned = self.u(enc_data[:, ind_unassigned, :])
        if self.use_attn:
            U = self.pma_u(us_unassigned).squeeze(dim=1)
        else:
            U = us_unassigned.mean(dim=1)    # [batch_size, u_dim]
        return anch, data_unassigned, us_unassigned, U

    def encode_unassigned_mask(self, enc_data, anchor, unassigned):
        anch = enc_data[anchor]
        data_unassigned = enc_data
        us_unassigned = self.u(enc_data)
        if self.use_attn:
            U = self.pma_u(us_unassigned, unassigned).squeeze(dim=1)
        else:
            U = self.masked_mean(us_unassigned, unassigned)
        return anch, data_unassigned, us_unassigned, U

    def masked_mean(self, tensor, mask):
        mask_sum = mask.sum(dim=1, keepdim=True)
        positives = mask_sum > 0
        mask_mean = torch.zeros_like(mask_sum).float()
        mask_mean[positives] = 1 / mask_sum[positives].float()
        return (tensor * mask.unsqueeze(-1) * mask_mean.unsqueeze(-1)).sum(dim=1)

    def conditional_posterior(self, us_unassigned, G, anch, targets, w):
        device = G.device
        t_in = targets.type(torch.BoolTensor)
        reduced_shape = (us_unassigned.shape[0], us_unassigned.shape[2])

        if torch.all(~t_in):  # all False, U_in should be zero
            U_in = torch.zeros(reduced_shape).to(device)
        else:
            if self.use_attn:
                U_in = self.pma_u_in(us_unassigned[:, t_in, :]).squeeze(1)
            else:
                U_in = us_unassigned[:, t_in, :].mean(dim=1)

        if torch.all(t_in):  # all True, U_out should be zero
            U_out = torch.zeros(reduced_shape).to(device)
        else:
            if self.use_attn:
                U_out = self.pma_u_out(
                    us_unassigned[:, ~t_in, :]).squeeze(1)
            else:
                U_out = us_unassigned[:, ~t_in, :].mean(dim=1)

        # [batch_size, z_dim], [batch_size, z_dim]
        qz_mu, qz_log_sigma = self.get_qz(anch, U_in, U_out, G)

        qz_b = dist.Normal(qz_mu, qz_log_sigma.exp())
        z = qz_b.rsample(torch.Size([w]))      # [w,batch_size, z_dim]
        return qz_mu, qz_log_sigma, z

    def conditional_posterior_mask(self, us_unassigned, G, anch, targets, unassigned, w):
        device = G.device
        reduced_shape = (us_unassigned.shape[0], us_unassigned.shape[2])
        invert_targets = unassigned & ~targets

        U_in = torch.zeros(reduced_shape).to(device)
        zero_mask = torch.all(~(targets & unassigned), dim=1)
        if self.use_attn:
            U_in[~zero_mask] = self.pma_u_in(
                us_unassigned[~zero_mask], targets[~zero_mask]).squeeze(1)
        else:
            U_in[~zero_mask] = self.masked_mean(
                us_unassigned[~zero_mask], targets[~zero_mask])

        U_out = torch.zeros(reduced_shape).to(device)
        zero_mask = torch.all(targets == unassigned, dim=1)
        if self.use_attn:
            U_out[~zero_mask] = self.pma_u_out(
                us_unassigned[~zero_mask], invert_targets[~zero_mask]).squeeze(1)
        else:
            U_out[~zero_mask] = self.masked_mean(
                us_unassigned[~zero_mask], invert_targets[~zero_mask])

        # [batch_size, z_dim], [batch_size, z_dim]
        qz_mu, qz_log_sigma = self.get_qz(anch, U_in, U_out, G)

        qz_b = dist.Normal(qz_mu, qz_log_sigma.exp())
        z = qz_b.rsample(torch.Size([w]))      # [w,batch_size, z_dim]
        return qz_mu, qz_log_sigma, z

    def vae_likelihood(self, z, U, G, anch, data_unassigned):
        w, batch_size = z.shape[0], z.shape[1]
        Lk = data_unassigned.shape[1]
        expand_shape = (-1, Lk, w, -1)

        zz = z.transpose(0, 1)   # [batch_size, w, z_dim]
        zz = zz.view(batch_size, 1, w, -1).expand(expand_shape)
        dd = data_unassigned.view(batch_size, Lk, 1, -1).expand(expand_shape)
        aa = anch.view(batch_size, 1, 1, -1).expand(expand_shape)
        UU = U.view(batch_size, 1, 1, -1).expand(expand_shape)
        GG = G.view(batch_size, 1, 1, -1).expand(expand_shape)

        ddzz = torch.cat([dd, zz, aa, UU, GG], dim=3).view(
            [batch_size*Lk*w, self.phi_input_dim])
        logits = self.phi(ddzz).view([batch_size, Lk, w])
        return logits

    def kl_loss(self, qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets):
        # For the loss function we use below the doubly-reparametrized gradient estimator from https://arxiv.org/abs/1810.04152
        # as implemented in https://github.com/iffsid/DReG-PyTorch

        pb_z = dist.Bernoulli(logits=logits)
        batch_size, Lk, w = logits.shape
        targets = targets.view(1, Lk, 1).expand(batch_size, Lk, w)
        lpb_z = pb_z.log_prob(targets).sum(dim=1)   # [batch_size,w]
        lpb_z.transpose_(0, 1)

        # in dreg qz_b is not differentiated
        qz_b_ = dist.Normal(qz_mu.detach(), qz_log_sigma.detach().exp())
        lqz_b = qz_b_.log_prob(z).sum(-1)  # [w,batch_size]

        lpz = dist.Normal(pz_mu, pz_log_sigma.exp()).log_prob(
            z).sum(-1)     # [w,batch_size]

        lw = lpz + lpb_z - lqz_b     # [w,batch_size]

        with torch.no_grad():
            reweight = torch.exp(lw - torch.logsumexp(lw + 1e-30, 0))
            # reweight = F.softmax(lw, dim=0)
            if self.training:
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

        loss = -(reweight * lw).sum(0).mean(0)

        le = torch.exp(lw).mean(dim=0) + 1e-30     # mean over w terms
        elbo = torch.log(le).mean()               # mean over minibatch

        return loss, elbo

    def kl_loss_mask(self, qz_mu, qz_log_sigma, pz_mu, pz_log_sigma, z, logits, targets, unassigned):
        # For the loss function we use below the doubly-reparametrized gradient estimator from https://arxiv.org/abs/1810.04152
        # as implemented in https://github.com/iffsid/DReG-PyTorch

        pb_z = dist.Bernoulli(logits=logits)
        batch_size, Lk, w = logits.shape
        targets = targets.unsqueeze(-1).expand(batch_size, Lk, w)
        lpb_z = (pb_z.log_prob(targets.float()) *
                 unassigned.unsqueeze(-1)).sum(dim=1)   # [batch_size,w]
        lpb_z.transpose_(0, 1)

        # in dreg qz_b is not differentiated
        qz_b_ = dist.Normal(qz_mu.detach(), qz_log_sigma.detach().exp())
        lqz_b = qz_b_.log_prob(z).sum(-1)  # [w,batch_size]

        lpz = dist.Normal(pz_mu, pz_log_sigma.exp()).log_prob(
            z).sum(-1)     # [w,batch_size]

        lw = lpz + lpb_z - lqz_b     # [w,batch_size]

        with torch.no_grad():
            reweight = torch.exp(lw - torch.logsumexp(lw + 1e-30, 0))
            # reweight = F.softmax(lw, dim=0)
            if self.training:
                z.register_hook(lambda grad: reweight.unsqueeze(-1) * grad)

        loss = -(reweight * lw).sum(0).mean(0)

        le = torch.exp(lw).mean(dim=0) + 1e-30     # mean over w terms
        elbo = torch.log(le).mean()               # mean over minibatch

        return loss, elbo
