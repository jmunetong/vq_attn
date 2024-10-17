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


@dataclasses.dataclass
class VQSpec:
    n_device: torch.Tensor
    n_block_per_update: torch.Tensor
    loss_mask: torch.Tensor

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in dataclasses.fields(VQSpec)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        return cls(**filtered)


def get_shortcodes(vecs, codebook):
    B, H, L, K = vecs.shape
    S = codebook.shape[1]
    chex.assert_shape(vecs, (B, H, L, K))
    chex.assert_shape(codebook, (H, S, K))
    diffs2 = (
        torch.sum(vecs**2, dim=-1, keepdim=True)
        - 2.0 * einsum("bhlk,hsk->bhls", vecs, codebook)
        + torch.sum(codebook**2, dim=-1).unsqueeze(0).unsqueeze(2)
    )  # B, H, L, S
    z = torch.argmin(diffs2, dim=-1)
    chex.assert_shape(z, (B, H, L))
    errs2 = torch.min(diffs2, dim=-1).values
    errs2 = F.relu(errs2)  # this is a no-op if using infinite precision
    chex.assert_shape(errs2, (B, H, L))
    return z.int(), errs2


def get_codewords(shortcodes, codebook):
    B, H, L = shortcodes.shape
    S, d = codebook.shape[1], codebook.shape[2]
    shortcodes = shortcodes.unsqueeze(-1)
    codebook = codebook.unsqueeze(0)
    chex.assert_shape(shortcodes, (B, H, L, 1))
    chex.assert_shape(codebook, (1, H, S, d))
    cz = torch.gather(codebook, 2, shortcodes.expand(-1, -1, -1, d))
    return cz


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
