
import numpy as np


class CRP_Generator():
    """Generate synthetic cluster distributions using the Chinese Restaurant Process (CRP) generative model

    The number of data points ~ Uniform(Nmin, Nmax)
    The cluster assignments is produced by a CRP generative model: 
    """

    def __init__(self, alpha=0.7, maxK=None):
        """Set CRP parameters
        Args:
            alpha: the dispersion parameter of CRP
            maxK: the maximum number of synthetic clusters (clip the long tail of large K)
        """
        self.alpha = alpha
        self.maxK = maxK

    def generate(self, N):
        """Generate synthetic cluster assignments

        Args:
            N: int
        Returns:
            clusters: a numpy array of shape (N,) of cluster assignments
            N: total number of data points
            K: total number of distinct clusters 
        """
        keep = True
        while keep:

            # 0...N,N+1  how many data points in each cluster
            clusters = np.zeros(N+2)
            clusters[0] = 0  # placeholder
            # first cluster. After this, fill the array with cumsum
            clusters[1] = 1
            clusters[2] = self.alpha  # dispersion
            index_new = 2  # the next unassigned cluster ID

            # loop over N-1 data points because the first one was assigned already to cluster 1
            for n in range(N-1):
                p = clusters / clusters.sum()
                # random draw from 0...n_clust+1
                z = np.argmax(np.random.multinomial(1, p))
                if z < index_new:  # existing clusters
                    clusters[z] += 1
                else:  # new cluster
                    clusters[index_new] = 1
                    index_new += 1
                    # the next new cluster, with prob = alpha/(n_samples + alpha)
                    clusters[index_new] = self.alpha

            clusters[index_new] = 0
            clusters = clusters.astype(np.int32)

            K = np.sum(clusters > 0)
            keep = (self.maxK is not None and K > self.maxK)

        clusters = clusters[clusters > 0]

        return clusters


