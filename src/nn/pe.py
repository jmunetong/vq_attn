import dataclasses

import torch
import torch.nn as nn

from transformer_vq.nn.types import TransformerConfig


def get_sinusoid_embs(length, width, lam, flip, start=0):
    pos_seq = start + torch.arange(length)
    inv_lams = 1 / (lam ** (torch.arange(0, width, 2) / width))
    pre = pos_seq[..., None] * inv_lams[None, ...]
    sin = torch.sin(pre)
    cos = torch.cos(pre)
    cat = torch.cat([sin, cos], dim=-1)
    if not flip:
        return cat
    return torch.flip(cat, dims=[0])


class ScaledSin(nn.Module):
    # see w. hua et al., 2022
    def __init__(self, config: TransformerConfig):
        super(ScaledSin, self).__init__()
        self.config = config
        self.apply_config()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def forward(self, length, offset):
        embs = get_sinusoid_embs(
            length=length, start=offset, width=self.d_model, lam=self.pe_lam, flip=False
        )
        return (self.scale * embs).type(self.dtype)
