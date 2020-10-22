
mog_data_params = {
    # number of data points for training, N ~ unif(Nmin, Nmax)
    'Nmin': 50,
    'Nmax': 200,

    # data shape
    'x_dim': 2,

    # CRP
    'alpha': .7,

    'maxK': 12,  # max number of clusters to generate

    # Gaussian Mixtures
    "sigma": 1,
    "lambda": 10,
}