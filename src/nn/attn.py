# python standard library
import dataclasses
from dataclasses import asdict
import logging

# third party
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange

# local
from transformer_vq_nn.config import TransformerConfig, VQSpec
from transformer_vq_nn.norm import LayerNorm
from transformer_vq_nn.pe import SinusoidPE
from transformer_vq_nn.vq import LearnableVQ


MASK_INFTY_APPROX = 1e30  # mask value approximating infinity

class VQAttention(nn.Module):
    config: TransformerConfig
    n_head: int
    d_k: int
    d_v: int
    n_code_k: int
    n_code_v: int
    p_dropout: float
    device: str
    dtype: torch.dtype
    param_dtype: torch.dtype
    mem_len: int
    block_len: int
    pe_lam: float
    agg_cache: bool
    grad_thru_cache: bool
    quantize_q: bool
    tau: float

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.apply_config()

        self.tau = self.d_k**0.5
        self.n_head = int(self.n.head)
        self.input_ln = LayerNorm(self.d_model)
        self.q_in = LayerNorm(self.d_model)
        self.k_ch = int(self.n_head * self.d_k)
        self.v_ch = int(self.n_head * self.d_v)
        self.k_ln = LayerNorm(self.d_model, self.d_k, gain=False, bias=False)
        self.q_ln = LayerNorm(self.d_model, self.d_k, gain=False, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.k_ch + self.v_ch, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.k_ch + self.v_ch, bias=False)
        self.r_proj = nn.Linear(self.v_ch, self.d_model, bias=False)
        self.res_proj = nn.Linear(self.v_ch, self.d_model, bias=False)
        self.x_proj = nn.Parameter(torch.zeros(self.n_head, self.d_k))
        self.x_u = nn.Parameter(torch.zeros(self.n_head, self.d_k))
        self.x_v = nn.Parameter(torch.zeros(self.n_head, self.d_k))
        self.n_code_k = int(self.n_code_k)
        config_k = asdict(self.config)
        config_k.update({'d_model': self.d_k})
        config_k.update({'n_code': int(self.n_code_k)})
        self.n_code = self.n_code_k
        self.quantizer_k = LearnableVQ(config_k)
        self.quantizer_q = None
        self.dropSin = nn.Dropout(self.p_dropsin)
        self.dropres = nn.Dropout(self.p_dropres)
        self.d_type = config.param_dtype
        self.SiLU = nn.SiLU()
        self._apply_initializers()

    def apply_config(self):
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)

    def mount_codebook(self, codebook_k: torch.Tensor=None, codebook_q: torch.Tensor=None):
        if not self.training:
            self.quantizer_k.update_index(codebook_k)
            if self.config.quantize_v:
                self.quantizer_q.update_index(codebook_q)
        else:
            logging.warning("Codebook not mounted during training")

    def forward(self, state: dict, input_features: torch.Tensor, doc_ids: torch.Tensor, vq_spec: VQSpec, **kwargs):
        """
        Method responsible for running the forward pass of the transformer model

        Args:
            state (dict): Dictionary containing the state of the transformer model
            input_features (torch.Tensor): Input features
            doc_ids (torch.Tensor): Document ids
            vq_spec (VQSpec): Specs for VQAttention Module
            kwargs: Additional arguments

        Returns:
            dict: Dictionary containing the output of the transformer model
        """

        q, k, v, g = self.compute_k_q_v_g(x=input_features)
        attn_output_dict = self.vq_attn(present_q=q,
                                        present_k=k,
                                        present_v=v,
                                        present_doc_ids=doc_ids,
                                        state=state,
                                        loss_mask=vq_spec.loss_mask)
        wv = attn_output_dict.get("attn_out")
        o = wv * g
        res = self.res_proj(o)
        res = self.dropres(res)
        return self._build_output_dicts(res, attn_output_dict, state)
    
    def get_quantized_matrices(self, k: torch.Tensor, q: torch.Tensor, loss_mask: torch.Tensor,):
        return tuple(map(lambda x: x['quantized'], self.quantizer.run_vqk(present_k=k, present_q=q, loss_mask=loss_mask)))

    def run_vqk(self, present_k: torch.Tensor, 
                present_q: torch.Tensor, 
                loss_mask: torch.Tensor, 
                return_vecs_hat: bool=True):
        vq_output_dict_k = self.quantizer_k(present_k, loss_mask, return_vecs_hat)
        if present_q is not None:
            vq_output_dict_q = self.quantizer_q(present_q, loss_mask)
            return (vq_output_dict_k, vq_output_dict_q)
        else:
            return (vq_output_dict_k)

    def compute_k_q_v_g(self, x: torch.Tensor):
        x_tilde = self.input_ln(x)
        q = self.get_q(x_tilde=x_tilde)
        k, v, g = self.get_kvg(x_tilde=x_tilde)
        return q, k, v, g
    
    def vq_attn(self, present_q: torch.Tensor, 
                present_k: torch.Tensor, 
                present_v: torch.Tensor, 
                present_doc_ids: torch.Tensor, 
                state: dict, 
                loss_mask: torch.Tensor) -> dict:
        vq_output_dict_k = self.run_vqk(present_k=present_k,
                                        present_q=present_q if self.quantize_q else None,
                                        loss_mask=loss_mask)
        present_z = vq_output_dict_k["shortcodes"]
        present_k_hat = vq_output_dict_k["quantized"]

        # concatenate sliding window cache k/v onto current block
        position_offset = state["pos_offset"]
        xlcache = state["xlcache"]
        aggcache = state["aggcache"]
        recent_z = torch.cat((xlcache["z_k"], present_z), axis=-1)
        recent_k_hat = torch.cat((xlcache["k_hat"], present_k_hat), axis=-2)
        recent_v = torch.cat((xlcache["v"], present_v), axis=-2)
    
        wv = self.attn(present_q,
                    present_k=recent_k_hat,
                    recent_v=recent_v,
                    aggcache=aggcache,
                    position_offset=position_offset)
        
        out = dict(
        attn_out=wv,
        recent_z=recent_z,
        recent_k_hat=recent_k_hat,
        recent_v=recent_v,
        recent_doc_ids=None,
        l_commit=vq_output_dict_k["l_commit"],
        l_codebook=vq_output_dict_k["l_codebook"],
    )
        return out
    
    def attn(self, present_q: torch.Tensor, recent_k_hat: torch.Tensor,
        recent_v: torch.Tensor, aggcache: dict, position_offset: int) -> dict:
        """
        Method responsible for computing the attention mechanism in the
        transformer model

        Args:
            present_q (torch.Tensor): Q matrix
            present_k (torch.Tensor): K matrix
            present_v (torch.Tensor): V matrix
            present_doc_ids (torch.Tensor): ids for each word sequence
            state (dict): dictionary containing the components for each state
            vq_spec (VQSpec): Specs for VQAttention Module

        Returns:
            dict: Dictionary containing the attention output, recent z,
                recent k_hat, recent v, and recent doc_ids.
            dict: dictionary containing the attention output, commitment and codebook losses
        """

        # compute xl bias helpers
        xl_r, xl_u, xl_v = self.get_xl_helpers()

        # compute aggcache scores
        c = self.quantizer_k.get_codebook()
        cache_scores = torch.einsum("bhlk,hsb->hbls", present_q + xl_u, c)
        cache_biases = self.agg_biases(aggcache["lower_k"])
        cache_biases = cache_biases.unsqueeze(-2)

        cache_scores = cache_scores + cache_biases
        # compute recent scores (present and xlcache)
        recent_scores_ac = torch.einsum("bhlk,bhw->bhlw", present_q + xl_u, recent_k_hat)
        recent_scores_bd = torch.einsum("bhlk,hw->bhlw", present_q + xl_v, xl_r)
        recent_scores_bd = self.rel_shift(recent_scores_bd)

        # recent scores
        recent_scores = self.apply_mask_recent_scores(recent_scores_ac,
                                                    recent_scores_bd,
                                                    position_offset)

        wv = self.compute_wv(cache_scores, recent_scores, aggcache, recent_v, bsz=present_q.shape[0])
        return wv


    def _compute_wv(self, cache_scores: torch.Tensor, recent_scores: torch.Tensor,
                    aggcache: dict, recent_v: int, bsz: int):
        cache_max_scores = torch.max(cache_scores, axis=-1)[0]
        recent_max_scores = torch.max(recent_scores, axis=-1)[0]
        max_scores = torch.maxmimum(cache_max_scores, recent_max_scores).detach(
        )

        cache_scores = cache_scores - max_scores
        recent_scores = recent_scores - max_scores.unsqueeze(-1)
        cache_a = torch.exp(cache_scores)
        recent_a = torch.exp(recent_scores)

        d = torch.sum(recent_a, axis=-1)
        if self.agg_cache:
            d = d + torch.sum(cache_a, axis=-1)
        wv = torch.einsum("bhlw,bhw->bhlv", recent_a / d.unsqueeze(-1), recent_v)
        if self.agg_cache:
            wv = wv + torch.einsum(
                "bhlv,bhsv->bhlv", cache_a / d.unsqueeze(-1),
                aggcache["upper_div_lower_k"]
            )

        wv = torch.permute(wv, (0, 2, 1, 3))
        wv = torch.reshape(wv, (bsz, self.block_len, self.n_head * self.d_v))
        return wv
    

