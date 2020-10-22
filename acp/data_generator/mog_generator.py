
import numpy as np
import torch
from .distributions import CRP_Generator
from .utils import relabel


def get_mog_crp_generator(params):
    partition_generator = CRP_Generator(
        alpha=params['alpha'], maxK=params['maxK'])
    mog_generator = MOG_Generator(
        x_dim=params['x_dim'],
        lambd=params['lambda'],
        sigma=params['sigma'],
        partition_generator=partition_generator,
        Nmin=params['Nmin'],
        Nmax=params['Nmax'])
    return mog_generator


class MOG_Generator():
    """Generate Gausian Mixtures
    """

    def __init__(self, x_dim, lambd, sigma, partition_generator, Nmin=None, Nmax=None):
        self.partition_generator = partition_generator
        self.x_dim = x_dim
        self.lambd = lambd
        self.sigma = sigma
        self.Nmin = Nmin
        self.Nmax = Nmax

    def generate(self, N=None, batch_size=1):

        if N is None:
            N = np.random.randint(self.Nmin, self.Nmax)

        clusters = self.partition_generator.generate(N=N)
        assert(np.all(clusters > 0))
        N = np.sum(clusters)
        K = len(clusters)

        cumsum = np.cumsum(np.insert(clusters, 0, [0]))
        data = np.empty([batch_size, N, self.x_dim])
        labels = np.empty(N, dtype=np.int32)

        for k in range(K):

            mu = np.random.normal(0, self.lambd, size=[
                                  self.x_dim * batch_size, 1])

            if type(self.sigma) == tuple:
                sig = np.random.uniform(self.sigma[0], self.sigma[1], size=[
                                        self.x_dim * batch_size, 1])
            else:
                sig = self.sigma

            samples = np.random.normal(
                mu, sig, size=[self.x_dim * batch_size, clusters[k]])
            samples = np.swapaxes(samples.reshape(
                [batch_size, self.x_dim, clusters[k]]), 1, 2)
            data[:, cumsum[k]:cumsum[k+1], :] = samples
            labels[cumsum[k]:cumsum[k+1]] = k

        # shuffle the data
        indices = np.arange(N)
        np.random.shuffle(indices)
        labels = labels[indices]
        data = data[:, indices, :]
        labels = relabel(labels)

        # normalize data
        #means = np.expand_dims(data.mean(axis=1),1 )
        medians = np.expand_dims(np.median(data, axis=1), 1)
        data = data - medians

        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).int()

        return data, labels

    def generate_batch(self, batch_size, device, data_lib=None):
        data, labels = self.generate(N=None, batch_size=batch_size)
        sorted_ind = torch.argsort(labels)
        labels = labels[sorted_ind].to(device)
        data = data[:, sorted_ind]
        data = data.to(device)
        return data, labels
