import torch
import numpy as np


class ACP_Single_Sampler():
    """Fast sampling without probability estimation.
    """

    def __init__(self, model, data, device=None):

        self.data = data
        self.model = model
        
        if device is not None:
            self.data = self.data.to(device)
            self.model = self.model.to(device)


    def sample(self, S=1, stochastic=False):
        
        assert(S == 1)  # only support one sample
        logprob = 0

        N = self.data.shape[1]
        unassigned = np.arange(N)
        k = 0
        Ss = []
        self.model.previous_k = -1

        cs = np.zeros(N, dtype=np.int32)

        L = 0
        with torch.no_grad():

            while len(unassigned) > 0:

                # logprob -= np.log(len(unassigned))
                anchor = np.random.choice(unassigned)
                cs[anchor] = k+1
                unassigned = unassigned[unassigned != anchor]
                L += len(unassigned)

                if len(unassigned) > 0:

                    logits = self.model(
                        k, self.data, anchor, unassigned, Ss, train=False, w=1)
                    logits = logits.detach().cpu().numpy()
                    sigmoids = 1/(1 + np.exp(-logits))

                    if stochastic:
                        inds = np.random.rand(len(unassigned)) < sigmoids[0, :]
                    else:
                        inds = .5 < sigmoids[0, :]
                    not_inds = np.invert(inds)

                    logprob += np.log(sigmoids[0, inds]).sum()
                    logprob += np.log(1-sigmoids[0, not_inds]).sum()

                    sk = unassigned[inds]
                    cs[sk] = k+1

                    Ss.append(np.concatenate(
                        [sk, np.arange(anchor, anchor+1)]))

                    unassigned = unassigned[not_inds]
                    k += 1

        cs = relabel(cs)
        logprob = logprob / L

        clusters = np.expand_dims(cs, 0)
        logprob = np.array([logprob])
        probs = np.exp(logprob)

        return clusters, probs


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



