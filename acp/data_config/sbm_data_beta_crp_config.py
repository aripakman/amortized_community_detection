
sbm_data_beta_crp_params = {
    # number of data points (for training), N ~ unif(Nmin, Nmax)
    'Nmin': 50,
    'Nmax': 300,

    # CRP
    'alpha': .7,
    'maxK': 12,  # max number of clusters to generate
    
    # SBM 
    'within_alpha': 6, 
    'within_beta': 3,      
    'between_alpha': 1,
    'between_beta': 5,

    # extra node features
    'random_embed_dim': 0,
    'pos_enc_dim': 20
}





