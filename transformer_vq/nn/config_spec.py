import torch
from dataclasses import dataclass, field
from typing import Any

class VQSpec:
    n_device: torch.Tensor
    n_block_per_update: torch.Tensor
    loss_mask: torch.Tensor

    def __init__(self, n_device: torch.Tensor, n_block_per_update: torch.Tensor, loss_mask: torch.Tensor, **kwargs):
        atrr_list = ["n_device", "n_block_per_update", "loss_mask"]
        self.n_device = n_device
        self.n_block_per_update = n_block_per_update
        self.loss_mask = loss_mask
        for k, v in kwargs.items():
            if k in atrr_list:
                setattr(self, k, v)

    
class TransformerConfig:
    param_dtype: torch.dtype
    dtype: torch.dtype
    global_batch_size: int
    sequence_len: int
    update_len: int
    block_len: int
    mem_len: int
    grad_thru_cache: bool
    agg_cache: bool
    d_model: int
    d_k: int
    d_v: int
    d_ff: int
    n_head: int
    n_code: int
    n_layer: int
    n_vocab: int
    pe_abs: bool
    pe_lam: float
    p_dropemb: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    p_nucleus: float
    c_beta: float
    c_gamma: float
    e_tie: bool
    e_preln: bool
    e_scale: str
    is_train: bool
    no_emb: bool = False
    n_code_q: int = 128
    n_code_k: int = 128
    quantize_q: bool = True

    def _get_attributes(self):
        return [
            "param_dtype", "dtype", "global_batch_size", "sequence_len", 
            "update_len", "block_len", "mem_len", "grad_thru_cache", 
            "agg_cache", "d_model", "d_k", "d_v", "d_ff", "n_head", 
            "n_code", "n_layer", "n_vocab", "pe_abs", "pe_lam", 
            "p_dropemb", "p_dropsin", "p_dropres", "p_droplyr", 
            "p_nucleus", "c_beta", "c_gamma", "e_tie", "e_preln", 
            "e_scale", "is_train", "no_emb"
        ]

    def __init__(self,**kwargs):
        for k, v in kwargs.items():
            if k in self._get_attributes():
                if k == 'param_dtype'or k == 'dtype':
                    if type(v)== str:
                        setattr(self, k, torch.float32 if v == 'float32' else torch.float64)
                    else:
                        setattr(self, k, v)
                else:
                    setattr(self, k, v) 
            else:
                setattr(self, k, v)

