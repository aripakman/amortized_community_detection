# Attentive Clustering Processes

This repo contains the PyTorch implementation of Attentive Clustering Process (ACP).

<!-- a normal html comment  
- Ari Pakman*, Yueqi Wang*, Yoonho Lee, Pallab Basu, Juho Lee, Yee Whye Teh, Liam Paninski, [Attentive Clustering Processes](https://arxiv.org/abs/2010.15727), arXiv preprint 2020
- Ari Pakman, Yueqi Wang, Catalin Mitelut, JinHyung Lee, Liam Paninski, [Neural Clustering Processes](https://arxiv.org/abs/1901.00409), ICML 2020
-->

### Graph community detection using ACP + Graph ConvNets (GCN)

Here we use synthetic graphs generated by stochastic block models (SBM) as an example.
#### Training
```
cd attentive_clustering_processes

# ACP with GraphSAGE encoder
python -m acp.train_acp --model_name acp --data_type sbm_beta_crp --encoder_type graphsage

# ACP with GatedGCN encoder
python -m acp.train_acp --model_name acp --data_type sbm_beta_crp --encoder_type gatedgcn

# Use an alternative ACP model definition, e.g. ACP-S
python -m acp.train_acp --model_name test_code --data_type sbm_beta_crp --encoder_type graphsage \
  --load_model_def acp/models/acps_model.py

```

#### Inference -- probablistic clustering
```
# Cluster SBM graphs using a saved checkpoint
python -m acp.inference.acp_cluster_sbm --data_type sbm_beta_crp --encoder_type graphsage \
  --model_file ./saved_models/xxxx.pt

# Use an alternative ACP model and sampler, e.g. ACP-S
python -m acp.inference.acp_cluster_sbm --data_type sbm_beta_crp --encoder_type gatedgcn \
  --model_file ./saved_models/xxxx.pt \
  --load_model_def acp/models/acps_model.py --load_sampler_def acp/models/acps_sampler.py

```

### Other types of data

#### Mixture of Gaussian (MOG)
```
python -m acp.train_acp --model_name mog --data_type mog --encoder_type mog
```

