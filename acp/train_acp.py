
import time
import os
import pickle
import argparse
import importlib
import torch
import numpy as np

from .data_config.mog_data_config import mog_data_params
from .config.mog_acp_config import mog_acp_params
from .data_config.sbm_data_beta_crp_config import sbm_data_beta_crp_params
from .config.sbm_acp_config import sbm_acp_params
from .models.acp_model import ACP_Model

from .data_generator.mog_generator import get_mog_crp_generator
from .data_generator.sbm_beta_generator import get_sbm_beta_crp_generator

from .encoders.mog_encoder import get_mog_encoder
from .encoders.sbm_graphsage_encoder import get_sbm_graph_sage_encoder
from .encoders.sbm_gatedgcn_dgl_encoder import get_sbm_gated_gcn_dgl_encoder

from .utils.plotting import plot_stats

parser = argparse.ArgumentParser(description="Train ACP model.")
parser.add_argument('--model_name', type=str, default="",
                    help="Model name")
parser.add_argument('--data_type', type=str, default="",
                    help="type of data for training")
parser.add_argument('--encoder_type', type=str, default="",
                    help="type of encoder")
parser.add_argument('--load_model_def', type=str, default="",
                    help="optional alternative .py file of acp_model definition")
parser.add_argument('--n_iter', type=int, default=10000,
                    help="number of training iterations.")
parser.add_argument('--saved_checkpoint', type=str, default=None,
                    help="if provided (as file path), continue training from the checkpoint.")
parser.add_argument('--save_every', type=int, default=1000,
                    help="save every.")
parser.add_argument('--print_every', type=int, default=1,
                    help="print every.")
parser.add_argument('--gpu', type=int, default=0,
                    help="gpu id.")
args = parser.parse_args()

device = torch.device(
    "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")


def train_acp():

    # data type
    if args.data_type == "mog":
        data_params = mog_data_params
        params = mog_acp_params
        get_data_generator = get_mog_crp_generator

    elif args.data_type == "sbm_beta_crp":
        data_params = sbm_data_beta_crp_params
        params = sbm_acp_params
        get_data_generator = get_sbm_beta_crp_generator

    else:
        raise ValueError("Unknown data type: " + args.data_type)

    # encoder type
    get_encoder = None  # function to build the encoder: get_encoder(params)

    if args.encoder_type == "mog":
        get_encoder = get_mog_encoder
        data_lib = None

    elif args.encoder_type == "graphsage":
        encoder_params = {
            "enc_in_dim": 20,  # orignal_feature + random_feature or pos_enc dim
            "enc_out_dim": params['e_dim'],
            "enc_hidden_dim": 128,
            "enc_layers": 4,
            "enc_dropout": 0}
        params.update(encoder_params)
        get_encoder = get_sbm_graph_sage_encoder
        data_lib = "torch_geom"

    elif args.encoder_type == "gatedgcn":
        encoder_params = {
            "enc_in_dim": 20,  # orignal_feature + random_feature or pos_enc dim
            "enc_hidden_dim": 128,
            "enc_out_dim": params['e_dim'],
            "enc_layers": 4,
            "enc_dropout": 0}
        params.update(encoder_params)
        get_encoder = get_sbm_gated_gcn_dgl_encoder
        data_lib = "dgl"
    else:
        raise ValueError("Unknown encoder type: " + args.encoder_type)

    #######################################################

    # load params if resuming from partially trained model
    if args.saved_checkpoint is not None:
        params = torch.load(args.saved_checkpoint)['params']
        data_params = torch.load(args.saved_checkpoint)['data_params']

    print("data params: ", data_params)
    print("model params: ", params)
    # build data generator

    data_generator = get_data_generator(data_params)

    # build encoder
    encoder = get_encoder(params)

    # build model
    model_name = args.data_type
    if args.load_model_def:
        module, module_name = load_module(args.load_model_def)
        model = module.ACP_Model(params, encoder).to(device)
        model_name += "_" + module_name
    else:
        model = ACP_Model(params, encoder).to(device)
        model_name += "_acp_model"

    if args.encoder_type:
        model_name += "_" + args.encoder_type + "_encoder"
    if args.model_name:
        model_name += "_" + args.model_name

    it = -1

    # containers to collect statistics
    losses = []      # Loss
    elbos = []       # ELBO

    # training parameters;
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #########################
    # resume from partially trained model
    if args.saved_checkpoint is not None:
        checkpoint = torch.load(args.saved_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        it = checkpoint['it']

        with open(args.saved_checkpoint[:-3] + "_stats.pkl", 'rb') as f:
            losses, elbos = pickle.load(f)
    #########################

    # total number of iterations
    n_iter = args.n_iter

    batch_size = params['batch_size']
    optim_update_every = params['optim_update_every']
    acp_vae_samples = params['vae_samples']

    model.train()
    optimizer.zero_grad()

    loss, elbo = None, None
    while True:
        t_start = time.time()
        it += 1

        # generate a new batch of data
        data, labels = data_generator.generate_batch(
            batch_size=batch_size, data_lib=data_lib, device=device)

        try:
            # if args.data_type in ["benchmarking_gnn"]:
            if isinstance(labels, list) and isinstance(labels[0], torch.Tensor):
                # batch data with different labels, need masked training loop
                N = max(len(x) for x in labels)
                K = max(len(torch.unique(x)) for x in labels)
                loss, elbo = model.forward_train_mask(
                    data, labels, w=acp_vae_samples)
            else:
                N = len(labels)
                K = len(torch.unique(labels))
                loss, elbo = model.forward_train(
                    data, labels, w=acp_vae_samples)

            loss.backward()

            losses.append(loss.item())
            elbos.append(elbo.item())

            if it % optim_update_every == 0:
                optimizer.step()
                optimizer.zero_grad()

            lr_curr = optimizer.param_groups[0]['lr']
            if it % args.print_every == 0:
                print('{0}  N:{1:3d}  K:{2}  ELBO:{3:4.3f}  Loss:{4:.3f}  Time/Iter: {5:.1f}  lr: {6}'
                      .format(it, N, K, np.mean(elbos[-50:]), np.mean(losses[-50:]), (time.time()-t_start), lr_curr))

        except RuntimeError as error:
            del data, labels, loss, elbo
            torch.cuda.empty_cache()
            print(error)  # check if it's OOM error
            print('RuntimeError handled  ', 'N:', N,
                  ' K:', K, 'Skip this iteration')
            continue

        if (it % args.save_every == 0 and it > 0) or it == n_iter+1:
            plot_metrics(
                metrics=[losses, elbos],
                names=["Loss", "ELBO"],
                model_name=model_name)

        if it % args.save_every == 0:
            if not os.path.isdir('saved_models'):
                os.mkdir('saved_models')
            # remove previous checkpoints
            if 'fname' in vars():
                os.remove(fname)
            fname = 'saved_models/{}_{}.pt'.format(model_name, it)
            torch.save({
                'it': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'params': params,
                'data_params': data_params
            }, fname)

            if 'pickle_fname' in vars():
                os.remove(pickle_fname)
            pickle_fname = 'saved_models/{}_{}_stats.pkl'.format(
                model_name, it)
            metrics_to_save = [losses, elbos]
            with open(pickle_fname, 'wb') as f:
                pickle.dump(metrics_to_save, f)

        # terminate training
        if it == n_iter:
            break


def plot_metrics(metrics, names, model_name, index=None, w=50):
    if not os.path.isdir('train_log'):
        os.mkdir('train_log')
    for data, name in zip(metrics, names):
        plot_stats(data, index=index, ylabel=name, log_y=False, w=w,
                   save_name='./train_log/train_{}_{}.pdf'.format(name.lower(), model_name))


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    module = importlib.import_module(filename.replace("/", ".").strip(".py"))
    return module, module_name


if __name__ == "__main__":
    train_acp()
