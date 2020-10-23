import numpy as np
import matplotlib.pyplot as plt

DEFAULT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9', 'C7',
                  'black', 'blue', 'red', 'green', 'magenta', 'brown', 'orange'] * 3


def plot_stats(stats, index=None, ylabel="Loss", log_y=False, w=50, save_name=None):

    m = np.ones(w)/w    
    avg_loss = np.convolve(stats,m,'valid')
    plt.figure(22, figsize=(13, 10))
    plt.clf()
    if log_y:
        if index is not None:
            plt.semilogy(index, avg_loss)
        else:
            plt.semilogy(avg_loss)
    else:
        if index is not None:
            plt.plot(index, avg_loss)
        else:
            plt.plot(avg_loss)
    plt.ylabel(ylabel)
    plt.xlabel('Iteration')
    plt.grid()

    if save_name:
        plt.savefig(save_name)
        plt.close()


