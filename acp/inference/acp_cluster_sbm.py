"""Clusters SBM models with random input node features with SAGE Conv + CCP.
"""

import torch
import numpy as np
import os
import argparse
import time
import importlib

import torch_geometric
import dgl

from sklearn.metrics import adjusted_mutual_info_score
from ..utils.sbm_utils import plot_colored_adj_matrix_with_prediction
from ..utils.plotting import DEFAULT_COLORS

from ..data_config.sbm_data_beta_crp_config import sbm_data_beta_crp_params

from ..data_generator.sbm_beta_generator import get_sbm_beta_crp_generator
from ..data_generator.utils import remap_labels_by_cluster_size

from ..encoders.sbm_graphsage_encoder import get_sbm_graph_sage_encoder
from ..encoders.sbm_gatedgcn_dgl_encoder import get_sbm_gated_gcn_dgl_encoder

from ..models.acp_model import ACP_Model
from ..models.acp_sampler import ACP_Sampler

from ..utils.graph_utils import edge_list_to_adj_matrix

parser = argparse.ArgumentParser(
    description='Run ACP inference for SBM graphs.')
parser.add_argument('--data_type', type=str, default=None,
                    help="(required) data generator type.")
parser.add_argument('--encoder_type', type=str, default=None,
                    help="(required) encoder type.")
parser.add_argument('--load_model_def', type=str, default="",
                    help="(optional) alternative .py file of acp_model definition")
parser.add_argument('--model_file', type=str, default=None,
                    help="(required) the file path of the trained model checkpoint.")
parser.add_argument('--load_sampler_def', type=str, default="",
                    help="(optional) alternative .py file of acp_sampler definition")
parser.add_argument('--n_graphs', type=int, default=50,
                    help="number of n_graphs to cluster.")
parser.add_argument('--out_dir', type=str, default="outputs",
                    help="output dir.")
parser.add_argument('--gpu', type=int, default=0,
                    help="gpu id.")
parser.add_argument('--S', type=int, default=10,
                    help="how many parallel samples.")
parser.add_argument('--prob_nZ', type=int, default=1,
                    help="how many Z to sample when estimating probability.")
parser.add_argument('--prob_nA', type=int, default=10,
                    help="how many anchors to sample when estimating probability.")
parser.add_argument('--use_train_data_params', action="store_true",
                    help="whether to use the data_params saved in model checkpoint.")


def main():
    print("\n")
    args = parser.parse_args()
    model_file = args.model_file
    device = torch.device('cuda:{}'.format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_file, map_location=device)
    params = checkpoint['params']
    train_data_params = checkpoint['data_params']

    # data generator
    if args.data_type == "sbm_beta_crp":
        data_params = sbm_data_beta_crp_params if not args.use_train_data_params else train_data_params
        data_generator = get_sbm_beta_crp_generator(data_params)

    else:
        raise ValueError("Unknown data type: " + args.data_type)

    # encoder
    if args.encoder_type == "graphsage": 
        encoder = get_sbm_graph_sage_encoder(params)
        data_lib = "torch_geom"

    elif args.encoder_type == "gatedgcn_dgl":
        encoder = get_sbm_gated_gcn_dgl_encoder(params)
        data_lib = "dgl"

    else:
        raise ValueError("Unknown encoder type: " + args.encoder_type)

    print("model params:", params)
    print("training data params:", train_data_params)

    # model
    if args.load_model_def:
        print("load model definition from:", args.load_model_def)
        model_module, model_module_name = load_module(args.load_model_def)
        model = model_module.ACP_Model(params, encoder).to(device)
    else:
        model = ACP_Model(params, encoder)

    # sampler class
    if args.load_sampler_def:
        sampler_module, sampler_module_name = load_module(
            args.load_sampler_def)
        SelectedSampler = sampler_module.ACP_Sampler
    else:
        SelectedSampler = ACP_Sampler

    # load model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    model.eval()

    n_data = args.n_graphs

    fname_prefix = "{}_{}".format(args.data_type, args.encoder_type)
    fname_postfix = "_N-{}_S-{}_pA-{}_pZ-{}".format(
        args.n_graphs, args.S, args.prob_nA, args.prob_nZ)
    out_dir = os.path.join(
        args.out_dir, args.data_type + "__" + os.path.basename(model_file).strip(".pt") + fname_postfix)
    fig_dir = os.path.join(out_dir, "figures")
    data_dir = os.path.join(out_dir, "data")
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    experiment_name = "Community Detection in SBM"

    all_metrics = {}
    all_amis = []
    all_times = []
    for i in range(n_data):
        print(i, end=' ')
        np.random.seed(i)
        data, labels = data_generator.generate_single(
            data_lib=data_lib, device=device)

        labels_np = labels.cpu().numpy()
        N = labels_np.shape[0]

        if isinstance(data, torch_geometric.data.Data):
            features = data.x
            edge_index = data.edge_index
        elif isinstance(data, dgl.DGLGraph):
            features = data.ndata['feat']
            edge_index = torch.stack(data.all_edges())
        else:
            raise ValueError("Unknown data type")

        features_np = features.cpu().numpy()
        adj_matrix_np = edge_list_to_adj_matrix(edge_index, N).cpu().numpy()

        t = time.time()
        try:
            sampler = SelectedSampler(model, data, device=device)

            clusters, probs = sampler.sample(
                S=args.S, sample_Z=False, sample_B=False,
                prob_nZ=args.prob_nZ, prob_nA=args.prob_nA)
        except Exception as e:
            print(e)
            continue
        predicted = clusters[np.argmax(probs)]

        inference_time = time.time() - t
        all_times.append(inference_time)

        fname = fname_prefix + "_example{}_N{}".format(i, N)

        np.savez_compressed(
            file=os.path.join(data_dir, fname + ".npz"),
            adj_matrix=adj_matrix_np, node2vec=features_np, labels=labels_np, predicted=predicted,
            inference_time=inference_time)

        ami = adjusted_mutual_info_score(labels.cpu().numpy(), predicted)
        all_amis.append(ami)

        title = experiment_name + " (i={}, N={}, AMI={:.3f})".format(i, N, ami)

        labels_sorted = remap_labels_by_cluster_size(labels.cpu()).numpy()
        predicted_sorted = remap_labels_by_cluster_size(
            torch.from_numpy(predicted)).numpy()
        plot_colored_adj_matrix_with_prediction(
            adj_matrix_np, labels_sorted, predicted_sorted, DEFAULT_COLORS, title=title,
            fontsize=16,
            bg_colors=['white', 'dimgray'],
            save_name=os.path.join(fig_dir, fname + ".png")
        )

    all_metrics["AMI"] = all_amis
    all_metrics["time"] = all_times
    print("\n")
    for metric, metric_array in all_metrics.items():
        mean_score = np.mean(metric_array)
        median_score = np.median(metric_array)
        print("{} -- mean: {:.4f}, median: {:.4f}".format(metric,
                                                          mean_score, median_score))


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    module = importlib.import_module(filename.replace("/", ".").strip(".py"))
    return module, module_name


if __name__ == "__main__":
    main()
