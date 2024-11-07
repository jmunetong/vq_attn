"""
Helper class for VQ Attention.

Contains mostly static methods (for ease of unit testing).
"""
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from transformer_vq.nn.grad import sg
from transformer_vq.nn.grad import st
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import get_sinusoid_embs
from transformer_vq.nn.config_spec import TransformerConfig
from index import IndexSearcher # TODO: Fix this import 


def codebook_loss(vecs, short_codes, c_sum, c_count, c_gamma, vq_spec, loss_mask=None):
    """
    Function that computes the codebook loss for a VQ-VAE.

    Args:
        vecs (torch.Tensor): Tensor of vectors, shape [B, H, S, D].
        short_codes (torch.Tensor): Tensor of quantized indices, shape [B, H, S].
        c_sum (torch.Tensor): Accumulated sum of vectors for each codebook vector, shape [H, L, D].
        c_count (torch.Tensor): Count of vectors for each codebook vector, shape [H, L].
        c_gamma (float): Momentum for EMA.
        vq_spec: Configuration for the VQ-VAE, not detailed in the original.
        loss_mask (torch.Tensor): Optional mask to apply to the loss.

    Returns:
        float: The codebook loss.
    """
    # Assume get_ema_targets is defined elsewhere to handle EMA updates
    c_sum_tgt, c_count_tgt = get_ema_targets(vecs, short_codes, c_sum, c_count, c_gamma, vq_spec, loss_mask)

    # Compute the loss components
    l_sum = torch.sum((c_sum - c_sum_tgt) ** 2)
    l_count = torch.sum((c_count - c_count_tgt) ** 2)

    # Total codebook loss
    l_codebook = l_count + l_sum
    return l_codebook

def get_ema_targets(vecs: torch.Tensor, short_codes: torch.Tensor,
                    c_count: torch.Tensor, c_sum: torch.Tensor,
                    gamma: float, vq_spec: dict, loss_mask: torch.Tensor, 
                    dtype):
    d = vq_spec.get('n_device')
    p = vq_spec.get('n_block_per_update')
    momentum = gamma
    num_code_vectors = c_sum.shape[1]
    r = F.one_hot(short_codes.long(), num_classes=num_code_vectors)  # [B, S]

    r = r * loss_mask.unsqueeze(-1)
    c_sum_hat = d * p * torch.einsum("mbhts,mhbtsd->mbhtd", r, vecs)
    
    c_count_hat = d * p * torch.sum(r, dim=(0, 1, 2))  # TODO: CHECK THIS WHETHER IT IS ACCURATE IN BLOOM
    c_sum_tgt = (1 - momentum) * c_sum + momentum * c_sum_hat
    c_count_tgt = (1 - momentum) * c_count + momentum * c_count_hat

    return c_sum_tgt, c_count_tgt

def get_shortcodes(vecs: torch.Tensor, codebook: torch.Tensor,
                   training=True, flaiss_searcher: IndexSearcher=None): # TODO: CREATE INDEX SEARCHER
    """
    Function to get shortcodes for the given vectors and codebook.
    In addition, it also computes the commitment loss.
    
    Args:
        vecs: torch.Tensor, shape [B, H, S, D]
        codebook: torch.Tensor, shape [H, L, D]
        training: bool, whether the model is in training mode. This check whether
                  we should compute arg-min
                  of l-2 distances or then we could compute with FLAISS

    Returns:
        z: torch.Tensor, shape [B, H, S]
        errs2: torch.Tensor, shape [B, H, S] (Commitment loss)
    """
    assert not codebook.requires_grad, "Codebook should not require gradients. \
    This is to compute commitment loss"

    if training:
        diffS2 = (
            torch.unsqueeze(torch.sum(torch.square(vecs), axis=-1), -1)
            - 2.0 * torch.einsum("tbhlk,hsd->tbhls", vecs, codebook)
            + torch.unsqueeze(torch.unsqueeze(torch.sum(torch.square(codebook), axis=-1), 0), 0)
        )  # B, H, L, S
        assert diffS2.shape == (vecs.shape[:-1] + codebook.shape[1:])
        errs2, z = torch.min(diffS2, axis=-1)
    else:
        if flaiss_searcher is None:
            raise ValueError("FLAISS searcher is required for inference")
        else:
            z, errs2 = flaiss_searcher.get_closest(vecs, k=1)
        errs2 = nn.ReLU()(errs2)  # this is a no-op if using infinite precision

    return z, errs2

class LearnableVQ(nn.Module):
    n_head: int
    n_code: int
    d_model: int
    loss_mask: torch.Tensor
    c_gamma: float
    param_dtype: torch.dtype
    device: torch.device
    index_name: str
    n_probe: int
    n_list: int
    n_bits: int
    M: int
    ef_search: int
    ef_construction: int

    def __init__(self, config: dict):
        self.param_fields = [
            'n_head',
            'n_code',
            'd_model',
            'c_gamma',
            'param_dtype',
            'device',
            'index_name',
            'n_probe',
            'n_list',
            'n_bits',
            'M',
            'ef_search',
            'ef_construction'
        ]
        super(LearnableVQ, self).__init__()
        self.config = config
        self.apply_config()
        self.d_type = self.param_dtype
        self.n_head = int(self.n_head)
        self.n_code = int(self.n_code)
        self.d_model = int(self.d_model)
        self.w = nn.Parameter(torch.empty((self.n_head, self.n_code, self.d_model)).to(device=self.device, dtype=self.d_type))

    def apply_config(self):
        for k, v in self.config.items():
            if k in list(self.param_fields):
                setattr(self, k, v)

    def _build_faiss_config(self):
        return dict(index_name=self.index_name,
                    n_probe=self.n_probe,
                    n_list=self.n_list,
                    n_bits=self.n_bits,
                    M=self.M,
                    ef_search=self.ef_search,
                    ef_construction=self.ef_construction)

    def update_index(self, codebook: torch.Tensor=None):
        codebook = self.w if codebook is None else codebook
        self.index_searcher.mount_codebook(codebook, faiss_configs=
                                        self._build_faiss_config())

    def forward(self, vecs: torch.Tensor, loss_mask=torch.Tensor([1]), return_vecs_hat: bool=True) -> torch.Tensor:
        assert vecs.shape[-1] == self.d_model
        assert vecs.shape[-3] == self.n_head
        codebook = self.get_codebook(epsilon=0.01)
        if not self.training and not self.index_searcher.is_codebook_ready():
            self.update_index()
        z, errs2 = get_shortcodes(vecs, codebook,
                                training=self.training,
                                flaiss_searcher=self.index_searcher if not self.training else None)
        if return_vecs_hat:
            cz = self.get_codevectors(z, codebook)
            vecs_hat = sg(cz) + st(vecs)
        else:
            vecs_hat = None
        
        if self.training:
            loss_mask = loss_mask.unsqueeze(1)
            l_commit = torch.mean(torch.sum(loss_mask.unsqueeze(1) * errs2, dim=1)) ## TODO: CHECK THIS LOSSS
            l_codebook = codebook_loss(vecs, z, self.w, self.c_count, self.c_gamma, self, loss_mask)
        else:
            l_commit = torch.tensor(0)
            l_codebook = torch.tensor(0)
        
        out = dict(quantized_vecs_hat=vecs_hat, shortcodes=z, l_commit=l_commit, l_codebook=l_codebook)
        return out

    @staticmethod
    def get_codevectors(shortcodes: torch.Tensor, codebooks: torch.Tensor):
        shortcodes = shortcodes.unsqueeze(-1).long() # [B, H, S, i]
        codebooks = codebooks.unsqueeze(0) # [i, H, S, d]
        n = shortcodes.ndim - codebooks.ndim
        codebooks = codebooks.view(*[1 for _ in range(n)], *codebooks.shape)
        cz = torch.take_along_dim(codebooks, indices=shortcodes, dim=-2)
        assert cz.shape == (*shortcodes.shape[:-1], codebooks.shape[-1])
        return cz









