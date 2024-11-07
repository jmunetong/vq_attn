import torch

import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, d_k=None, gain=True, bias=True, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) if gain else None
        self.beta = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normed = (x - mean) / (std + self.eps)
        if self.gamma is not None:
            normed = normed * self.gamma
        if self.beta is not None:
            normed = normed + self.beta
        return normed