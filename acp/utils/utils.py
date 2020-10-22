
import numpy as np

def get_topn_clusters(clusters, nll, topn=1):
    """Get the top-n cluster assignments sorted by probability 
    """
    sorted_nll = np.sort(list(set(nll)))
    topn_clusters = []
    topn_nll = []
    for i in range(topn):
        snll = sorted_nll[i]
        r = np.nonzero(nll == snll)[0][0]
        cs = clusters[r, :]
        topn_clusters.append(cs)
        topn_nll.append(snll)
    topn_clusters = np.array(topn_clusters)
    topn_nll = np.array(topn_nll)
    return topn_clusters, topn_nll

