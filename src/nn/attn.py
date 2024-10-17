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
        recent_doc_ids = torch.cat((xlcache["doc_ids"], present_doc_ids), axis=-1)

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
        recent_doc_ids=recent_doc_ids,
        l_commit=vq_output_dict_k["l_commit"],
        l_codebook=vq_output_dict_k["l_codebook"],
    )
        return out