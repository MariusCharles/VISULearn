import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """
    Projection head as defined in 
    'A Simple Framework for Contrastive Learning of Visual Representations, Chen et al. (2020)'.
    """
    def __init__(self, d_in, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(inplace=True),
            nn.Linear(d_in, d_model)
        )


    def forward(self, x: Tensor):
        return self.net(x)