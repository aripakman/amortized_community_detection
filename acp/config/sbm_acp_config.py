
sbm_acp_params = {
    # training
    'batch_size': 16,
    'optim_update_every': 2,  # increase effective batch size to reduce loss variance
    'vae_samples': 40,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,

    # neural net architecture for CCP
    'h_dim': 128,  # 256
    'g_dim': 128,  # 256
    'H_dim': 128,  # 128
    'e_dim': 128,  # 256
    'z_dim': 128,  # 256

    'use_attn': True,
    'n_heads': 4,
    'n_inds': 32,
}
