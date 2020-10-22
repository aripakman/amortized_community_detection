
import torch
import torch.nn as nn


def get_mog_encoder(params):
    return MOG_Encoder(
        in_dim=params['x_dim'],
        out_dim=params['e_dim'],
        H_dim=params['H_dim'])


class MOG_Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, H_dim):
        super().__init__()

        H = H_dim
        self.h = torch.nn.Sequential(
            torch.nn.Linear(in_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, out_dim),
        )

    def forward(self, x):
        return self.h(x)
