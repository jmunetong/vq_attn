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
from transformer_vq.nn.types import TransformerConfig
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
        shortcodes = shortcodes.unsqueeze(-1).long()
        codebooks = codebooks.unsqueeze(0)
        n = shortcodes.ndim - codebooks.ndim
        codebooks = codebooks.view(*[1 for _ in range(n)], *codebooks.shape)
        cz = torch.take_along_dim(codebooks, indices=shortcodes, dim=-2)
        assert cz.shape == (*shortcodes.shape[:-1], codebooks.shape[-1])
        return cz









##### LINGLEEEEEEEE ####################

class LearnableVQ(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.apply_config()
        self.c_sum = nn.Parameter(torch.zeros(self.n_head, self.n_code, self.d_k))
        self.c_count = nn.Parameter(torch.ones(self.n_head, self.n_code))

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def _get_codebook(c_sum, c_count):
        c = c_sum / torch.clamp(c_count[..., None], min=0.01)
        return sg(c)

    def get_codebook(self):
        return LearnableVQ._get_codebook(self.c_sum, self.c_count)

    @staticmethod
    def get_codebook_ema_targets(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec):
        n_code = c_sum.shape[1]
        B, H, L, d = vecs.shape
        chex.assert_shape(vecs, (B, H, L, d))
        chex.assert_shape(shortcodes, (B, H, L))
        chex.assert_shape(c_sum, (H, S, d))
        chex.assert_shape(c_count, (H, S))
        chex.assert_shape(vq_spec.loss_mask, (B, L))
        g = c_gamma
        d = vq_spec.n_device
        p = vq_spec.n_block_per_update
        chex.assert_shape(d, (1,))
        chex.assert_shape(p, (1,))
        r = F.one_hot(shortcodes, num_classes=n_code).type(vecs.dtype)
        r *= vq_spec.loss_mask.unsqueeze(1).unsqueeze(-1)
        c_sum_hat = d * p * einsum("bhts,bhtd->hsd", r, vecs)
        c_count_hat = d * p * torch.sum(r, dim=(0, 2))
        c_sum_tgt = (1 - g) * c_sum_hat + g * c_sum
        c_count_tgt = (1 - g) * c_count_hat + g * c_count
        chex.assert_shape(c_sum_tgt, (H, S, d))
        chex.assert_shape(c_count_tgt, (H, S))
        return c_sum_tgt, c_count_tgt

    @staticmethod
    def get_codebook_loss(
        vecs,
        shortcodes,
        c_sum,
        c_count,
        c_gamma,
        vq_spec,
    ):
        batch_size, n_head, block_len, d_k = vecs.shape
        n_code = c_count.shape[1]
        c_sum_tgt, c_count_tgt = LearnableVQ.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            vq_spec=vq_spec,
        )
        l_codebook_sum = torch.sum(sg(c_sum - c_sum_tgt) * st(c_sum))
        l_codebook_count = torch.sum(sg(c_count - c_count_tgt) * st(c_count))
        l_codebook = l_codebook_count + l_codebook_sum
        return l_codebook

    @staticmethod
    def get_quantization_metrics(vecs, vecs_hat, errs2, c_sum, c_count, dtype):
        n_head, n_code = c_count.shape
        eps, errmin, errmax, maskval = 1e-2, 0e1, 1e1, 1e30
        c_count = torch.clamp(c_count, min=eps)
        c = c_sum / c_count[..., None]  # HSd
        c_norms = torch.clamp(torch.norm(c, dim=-1), min=eps)  # HS
        c_normed = c / c_norms[..., None]  # HSd
        c_sims = einsum("hsd,hzd->hsz", c_normed, c_normed)  # HSS
        c_dists = torch.norm(
            c.unsqueeze(2) - c.unsqueeze(1), dim=-1
        )  # HSS
        vec_norms = torch.clamp(torch.norm(vecs, dim=-1), min=eps)  # BHL
        vec_hat_norms = torch.clamp(torch.norm(vecs_hat, dim=-1), min=eps)  # BHL
        errs = torch.sqrt(errs2)  # BHL
        relative_errs = torch.clamp(errs / vec_norms, min=errmin, max=errmax)  # BHL
        probs = c_count / torch.sum(c_count, dim=-1, keepdim=True)  # HS
        c_thresh_oob = torch.logical_or(c_count < 1.0, c_count > 1_000_000).float()

        ones = torch.ones([1, n_code, n_code], dtype=torch.float32)
        up = torch.triu(ones)  # upper triangular ones mask
        low = torch.tril(ones, diagonal=-1)  # strict lower triangular ones mask
        metrics = dict(
            c_sim_min=torch.min(low * c_sims + maskval * up, dim=(1, 2)).values,  # [H]
            c_sim_mean=torch.sum(low * c_sims, dim=(1, 2)) / torch.sum(low, dim=(1, 2)),
            c_sim_max=torch.max(low * c_sims - maskval * up, dim=(1, 2)).values,  # [H]
            c_dist_min=torch.min(low * c_dists + maskval * up, dim=(1, 2)).values,  # [H]
            c_dist_mean=torch.sum(low * c_dists, dim=(1, 2)) / torch.sum(low, dim=(1, 2)),
            c_dist_max=torch.max(low * c_dists - maskval * up, dim=(1, 2)).values,  # [H]
            c_norm_min=torch.min(c_norms, dim=1).values,  # [H]
            c_norm_mean=torch.mean(c_norms, dim=1),  # [H]
            c_norm_max=torch.max(c_norms, dim=1).values,  # [H]
            c_usage_min=torch.min(c_count, dim=1).values,  # [H]
            c_usage_mean=torch.mean(c_count, dim=1),  # [H]
            c_usage_max=torch.max(c_count, dim=1).values,  # [H]
            c_thresh_oob=torch.sum(c_thresh_oob, dim=1),  # [H]
            c_entropy=torch.sum(F.kl_div(probs, reduction='none'), dim=-1),  # [H]
            vec_norm_mean=torch.mean(vec_norms, dim=2),  # [B, H]
            vec_hat_norm_mean=torch.mean(vec_hat_norms, dim=2),  # [B, H]
            relative_err_min=torch.min(relative_errs, dim=2).values,  # [B, H]
            relative_err_mean=torch.mean(relative_errs, dim=2),  # [B, H]
            relative_err_max=torch.max(relative_errs, dim=2).values,  # [B, H]
        )
        return {k: sg(v).mean().type(dtype) for k, v in metrics.items()}

    def forward(self, vecs, vq_spec):
        orig_dtype = vecs.dtype
        vecs_hp = vecs.to(self.param_dtype)
        c = LearnableVQ._get_codebook(self.c_sum, self.c_count)
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        errs2 = errs2.to(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.to(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)
        if self.is_train:
            loss_mask = vq_spec.loss_mask
            l_commit = torch.mean(torch.sum(loss_mask.unsqueeze(1) * errs2, dim=1))
            l_codebook = LearnableVQ.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                vq_spec=vq_spec,
            ).to(self.dtype)
        else:
            l_commit = torch.zeros(dtype=self.dtype)
            l_codebook = torch.zeros(dtype=self.dtype)
        if self.is_train:
            metrics = LearnableVQ.get_quantization_metrics(
                vecs=sg(vecs),
                vecs_hat=sg(vecs_hat),
                errs2=sg(errs2),
                c_sum=sg(self.c_sum),
                c_count=sg(self.c_count),
                dtype=self.dtype,
            )
        else:
            metrics = dict()
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )


class SimpleVQ(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.apply_config()
        self.tau = self.d_k**0.5
        self.norm = LayerNorm(
            input_dim=self.d_k,
            param_dtype=self.param_dtype,
            center=False,
            norm=True,
            gain=False,
            bias=False,
        )

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def get_codebook(self):
        c = get_sinusoid_embs(
            length=self.n_code, width=self.d_k, start=0, lam=self.pe_lam, flip=False
        )
        return (self.tau**-0.5) * sg(self.norm(c))[None, ...]

    def forward(self, vecs, vq_spec):
        orig_dtype = vecs.dtype
        vecs_hp = vecs.to(self.param_dtype)
        c = self.get_codebook()
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        errs2 = errs2.to(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.to(orig_dtype)
        vecs_hat = sg(cz) + st(vecs)
        if self.is_train:
            loss_mask = vq_spec.loss_mask
            l_commit = torch.mean(torch.sum(loss_mask.unsqueeze(1) * errs2, dim=1))
            l_codebook = torch.zeros(dtype=self.dtype)
        else:
            l_commit = torch.zeros(dtype=self.dtype)
            l_codebook = torch.zeros(dtype=self.dtype)
        metrics = dict()
        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )
