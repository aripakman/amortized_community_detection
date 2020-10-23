import torch
import numpy as np


class ACP_Sampler():
    """Posterior sampler of the ACP-S model
    """

    def __init__(self, model, data, device=None):

        if not torch.cuda.is_available():
            print('Warning: CUDA is not available')

        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.device = device

        # use the batch dim
        # data = data.view([N, *data.shape[2:]]).to(device)

        with torch.no_grad():
            self.enc_data = model.encoder(data)  # [1,N, e_dim]
            # the batch dim is important for ISAB to work properly
            if len(self.enc_data.shape) == 2:
                self.enc_data = self.enc_data.view([1, -1, model.e_dim])
            if self.model.use_attn:
                self.enc_data = self.model.isab_enc(self.enc_data)
            self.enc_data = self.enc_data.view([-1, model.e_dim])

            self.hs = self.model.h(self.enc_data)  # [N,h_dim]
            self.us = self.model.u(self.enc_data)  # [N,u_dim]

    def _sample_Z(self, A, U, G, nZ):

        mu, log_sigma = self.model.get_pz(A, U, G)  # [t,z_dim]
        std = log_sigma.exp().unsqueeze(0)
        eps = torch.randn([nZ, *mu.shape]).to(self.device)
        return mu.unsqueeze(0) + eps*std

    def sample(self, S, sample_Z=False, sample_B=False, prob_nZ=10, prob_nA=None):
        """
        S: number of parallel sampling threads 
        nZ = number of samples of the latent variable z when estimating the probability of each sample

        Output:
            cs: tensor of samples, shape = [Ss,N], where Ss <= S, because duplicates are eliminated
            probs: estimated probabilities of the Ss samples 
        """
        ccs = self._sample(S=S, sample_Z=sample_Z, sample_B=sample_B)
        cs, probs = self._estimate_probs(ccs, nZ=prob_nZ, nA=prob_nA)

        return cs, probs

    def _sample(self, S, sample_Z=False, sample_B=False):

        device = self.device
        N = self.enc_data.shape[0]

        G = torch.zeros([S, self.model.g_dim]).to(device)
        cs = -torch.ones([S, N]).long().to(device)

        big_us = self.us.view([1, N, self.model.u_dim]
                              ).expand(S, N, self.model.u_dim)
        big_hs = self.hs.view([1, N, self.model.h_dim]
                              ).expand(S, N, self.model.h_dim)

        with torch.no_grad():

            # this matrix keeps track of available unassigned indices in each thread
            mask = torch.ones([S, N]).to(device)

            k = -1
            t = S  # t counts how many threads have not completed their sampling
            while t > 0:

                k += 1

                # sample the anchor element in a new cluster for each thread
                anchs = torch.multinomial(mask[:t, :], 1)  # [t,1]
                # assign label k to anchor elements
                cs[:t, :].scatter_(1, anchs, k)

                # eliminate selected anchors from the mask
                mask[:t, :].scatter_(1, anchs, 0)
                if self.model.use_attn:
                    U = self.model.pma_u(big_us[:t], mask[:t, :]).squeeze(1)
                else:
                    # this is used when U is agregated from us with the mean
                    normalized_mask = mask[:t, :] / \
                        mask[:t, :].sum(1, keepdims=True)
                    U = torch.mm(normalized_mask, self.us)  # [t, u_dim]

                A = self.enc_data[anchs[:, 0], :]  # [t, u_dim]

                if sample_Z:
                    Z = self._sample_Z(A, U, G[:t, :], 1)  # [1,t,z_dim]
                else:
                    # or use the mu without sampling
                    Z, _ = self.model.get_pz(A, U, G[:t, :])
                    Z = Z.unsqueeze(0)

                Ur = U.view([t, 1, self.model.u_dim]).expand(
                    t, N, self.model.u_dim)
                Ar = A.view([t, 1, self.model.e_dim]).expand(
                    t, N, self.model.e_dim)
                Zr = Z.view([t, 1, self.model.z_dim]).expand(
                    t, N, self.model.z_dim)
                Gr = G[:t, :].view([t, 1, self.model.g_dim]).expand(
                    t, N, self.model.g_dim)
                Dr = self.enc_data.view([1, N, self.model.e_dim]).expand(
                    t, N, self.model.e_dim)

                phi_arg = torch.cat([Dr, Zr, Ar, Ur, Gr], dim=2).view(
                    [t*N, self.model.phi_input_dim])

                logits = self.model.phi(phi_arg).view([t, N])
                prob_one = 1/(1+torch.exp(-logits))

                if sample_B:
                    inds = torch.rand([t, N]).to(device) < prob_one[:, :]
                else:
                    inds = .5 < prob_one[:, :]
                sampled = inds.long()

                # these are the points that were available (1 in mask) AND sampled (1 in sampled)
                sampled_new = mask[:t, :].long()*sampled

                # find the flattened indices of new points for cluster k
                new_points = torch.nonzero(
                    sampled_new.view(t*N), as_tuple=True)
                # assign points to cluster k
                cs[:t, :].view(t*N)[new_points] = k

                mask_update = 1-sampled
                # the matrix mask_update has a 1 on those points that survived the last sampling
                # so if a point was available before sampling and survived, it should be available in mask
                mask[:t, :] = mask[:t, :]*mask_update

                new_cluster = (cs[:t, :] == k).float()
                if self.model.use_attn:
                    new_Hs = self.model.pma_h(
                        big_hs[:t], new_cluster).squeeze(1)
                else:
                    # this is used when H is agregated from hs with the mean
                    new_cluster = new_cluster/new_cluster.sum(1, keepdims=True)
                    new_Hs = torch.mm(new_cluster, self.hs)

                G[:t] = G[:t] + self.model.g(new_Hs)

                msum = mask[:t, :].sum(dim=1)
                if (msum == 0).any():                # if any thread was fully sampled
                    msumfull = mask.sum(dim=1)

                    # reorder the threads so that those already completed are at the end
                    mm = torch.argsort(msumfull, descending=True)
                    mask = mask[mm, :]
                    cs = cs[mm, :]
                    G = G[mm, :]
                    # recompute the number of threads where there are still points to assign
                    t = (mask.sum(dim=1) > 0).sum()

        cs = cs.cpu().numpy()
        for i in range(S):
            cs[i, :] = relabel(cs[i, :])

        # eliminate duplicates
        lcs = list(set([tuple(cs[i, :]) for i in range(S)]))
        Ss = len(lcs)
        ccs = np.zeros([Ss, N], dtype=np.int32)
        for s in range(Ss):
            ccs[s, :] = lcs[s]

        return ccs

    def _estimate_probs(self, cs, nZ, nA=None):

        device = self.device

        with torch.no_grad():

            S = cs.shape[0]
            N = cs.shape[1]
            probs = np.ones(S)

            for s in range(S):

                K = cs[s, :].max()+1
                G = torch.zeros(self.model.g_dim).to(self.device)

                # array of available indices before sampling cluster k
                Ik = np.arange(N)

                for k in range(K):

                    # all these points are possible anchors
                    Sk = cs[s, :] == k

                    nk = len(Ik)

                    ind_in = np.where(cs[s, Ik] == k)[0]
                    ind_out = np.where(cs[s, Ik] != k)[0]

                    if nA is None or Sk.sum() < nA:
                        sk = Sk.sum()
                        anchors = ind_in
                    else:
                        sk = nA
                        anchors = np.random.choice(ind_in, sk, replace=False)

                    d1 = list(range(sk))

                    A = self.enc_data[Ik, :][anchors, :]

                    if self.model.use_attn:
                        U = self.us[Ik, :].view([1, nk, self.model.u_dim]).expand(
                            [sk, nk, self.model.u_dim])
                        mask = torch.ones([sk, nk]).to(device)
                        mask[d1, anchors] = 0
                        U = self.model.pma_u(U, mask).squeeze(1)
                    else:
                        U = self.us[Ik, :].sum(0)
                        U = U.view([1, self.model.u_dim]).expand(
                            sk, self.model.u_dim)
                        U = U-self.us[Ik, :][anchors, :]
                        U /= nk-1

                    Ge = G.view([1, self.model.u_dim]).expand(
                        sk, self.model.u_dim)

                    Z = self._sample_Z(A, U, Ge, nZ)  # [nZ,sk,z_dim]

                    Ar = A.view([1, 1, sk, self.model.e_dim]).expand(
                        [nZ, nk, sk, self.model.e_dim])
                    Dr = self.enc_data[Ik, :]
                    Dr = Dr.view([1, nk, 1, self.model.e_dim]).expand(
                        [nZ, nk, sk, self.model.e_dim])
                    Ur = U.view([1, 1, sk, self.model.u_dim]).expand(
                        [nZ, nk, sk, self.model.u_dim])
                    Gr = G.view([1, 1, 1, self.model.g_dim]).expand(
                        [nZ, nk, sk, self.model.g_dim])
                    Zr = Z.view([nZ, 1, sk, self.model.z_dim]).expand(
                        [nZ, nk, sk, self.model.g_dim])

                    phi_arg = torch.cat([Dr, Zr, Ar, Ur, Gr], dim=3).view(
                        [nZ*nk*sk, self.model.phi_input_dim])

                    logits = self.model.phi(phi_arg).view([nZ, nk, sk])
                    prob_one = 1/(1+torch.exp(-logits))
                    prob_one = prob_one.cpu().detach().numpy()

                    prob_one[:, anchors, d1] = 1

                    pp = prob_one[:, ind_in, :].prod(1)
                    if len(ind_out) > 0:
                        pp *= (1-prob_one)[:, ind_out, :].prod(1)

                    pp = pp.mean(0).sum()*(Sk.sum()/sk)/nk

                    probs[s] *= pp

                    # prepare for next iteration
                    Hs = self.hs[Sk, :]
                    if self.model.use_attn:
                        Hs = Hs.unsqueeze(0)
                        Hs = self.model.pma_h(Hs).view(self.model.h_dim)
                    else:
                        Hs = Hs.mean(dim=0)
                    G += self.model.g(Hs)

                    # update the set of available indices
                    Ik = np.setdiff1d(Ik, np.where(Sk)[0], assume_unique=True)

        # sort in decreasing order of probability
        inds = np.argsort(-probs)
        probs = probs[inds]
        cs = cs[inds, :]

        return cs, probs


def relabel(cs):
    cs = cs.copy()
    d = {}
    k = 0
    for i in range(len(cs)):
        j = cs[i]
        if j not in d:
            d[j] = k
            k += 1
        cs[i] = d[j]

    return cs
