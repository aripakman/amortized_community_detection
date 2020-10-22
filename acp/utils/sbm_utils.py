import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def plot_heatmap(adj_matrix, labels, ax, colors=['black', 'white']):
    sorted_idx = np.argsort(labels)
    adj_matrix = adj_matrix[sorted_idx, :][:, sorted_idx]
    cmap = mpl.colors.ListedColormap(colors)
    bounds = [0, 1, 2]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(adj_matrix, cmap=cmap, norm=norm)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return


def plot_colored_heatmap(adj_matrix, labels, ax, colors, bg_colors=['black', 'white']):
    assert(len(colors) >= len(set(labels)))
    colors = colors[:len(set(labels))]
    cluster_ids = sorted(set(labels))
    sorted_idx = np.argsort(labels)
    labels = labels[sorted_idx]
    adj_matrix = adj_matrix[sorted_idx, :][:, sorted_idx]
    for k in cluster_ids:
        mask = (labels == k)
        adj_matrix[np.ix_(mask, mask)] *= k + 2  # start from 2
    cmap = mpl.colors.ListedColormap(bg_colors + colors)
    bounds = [0, 1] + [x + 2 for x in cluster_ids] + [max(cluster_ids) + 3]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(adj_matrix, cmap=cmap, norm=norm)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    return


def plot_colored_adj_matrix_with_prediction(
        adj_matrix, labels, predicted, colors, title=None, fontsize=12, save_name=None, bg_colors=['black', 'white']):
    sns.set()
    sns.set_style('white')
    assert(len(colors) >= max(len(set(labels)), len(set(predicted))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_colored_heatmap(adj_matrix, labels, axes[0], colors, bg_colors=bg_colors)
    axes[0].set_title("Ground Truth", fontsize=fontsize)

    plot_colored_heatmap(adj_matrix, predicted, axes[1], colors, bg_colors=bg_colors)
    axes[1].set_title("Inferred", fontsize=fontsize)

    if title:
        plt.suptitle(title, fontsize=fontsize+2)

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    if save_name:
        plt.savefig(save_name)
        plt.close()
    return



