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
from transformer_vq.nn.config_spec import TransformerConfig, VQSpec
from transformer_vq.nn.norm import LayerNorm
from transformer_vq.nn.pe import SinusoidPE
from transformer_vq.nn.vq import LearnableVQ


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

    def __init__(self, config):
        super(VQAttention, self).__init__()
        self.config = config
        self.apply_config()
        self.tau = self.d_k**0.5
        self.n_head = int(self.n_head)
        self.n_code_k = int(self.n_code)
        self.input_ln = LayerNorm(self.d_model, self.d_k, gain=False, bias=False)
        self.q_in = LayerNorm(self.d_model)
        self.q_ch = int(self.n_head * self.d_k)
        self.k_ch = int(self.n_head * self.d_k)
        self.v_ch = int(self.n_head * self.d_v)
        self.k_ln = LayerNorm(self.d_model, self.d_k, gain=False, bias=False)
        self.q_ln = LayerNorm(self.d_model, self.d_k, gain=False, bias=False)
        self.q_proj = nn.Linear(self.d_model, self.q_ch, bias=False)
        self.kvg_proj = nn.Linear(self.d_model, self.k_ch + self.v_ch + self.v_ch , bias=False)
        # self.v_proj = nn.Linear(self.d_model, self.k_ch + self.v_ch, bias=False)
        self.r_proj = nn.Linear(self.v_ch, self.d_model, bias=False)
        self.res_proj = nn.Linear(self.v_ch, self.d_model, bias=False)
        self.x_proj = nn.Parameter(torch.zeros(self.n_head, self.d_k))
        self.x_u = nn.Parameter(torch.zeros(self.n_head, self.d_k))
        self.x_v = nn.Parameter(torch.zeros(self.n_head, self.d_k))
        self.n_code_k = int(self.n_code_k)
        config_k = self.config.__dict__
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
        # try:
        #     data_dict = dataclasses.asdict(self.config)
        # except:
        #     data_dict = self.config.__dict__
        data_dict = self.config.__dict__

        for k, v in data_dict.items():
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
        q = self._get_q(x_tilde=x_tilde)
        k, v, g = self._get_kvg(x_tilde=x_tilde)
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
    

    def build_output_dicts(self, res, attn_output_dict, state):
        new_state = self.update_state(
            recent_z_k=attn_output_dict.get("recent_z_k"),
            recent_k_hat=attn_output_dict.get("recent_k_hat"),
            recent_v=attn_output_dict.get("recent_v"),
            recent_doc_ids=attn_output_dict.get("recent_doc_ids"),
            state=state,
        )
        output_dict = dict(
            res=res,
            l_commit=attn_output_dict.get("l_commit"),
            l_codebook=attn_output_dict.get("l_codebook"),
        )
        return new_state, output_dict

    def _apply_initializers(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    @staticmethod
    def initial_state(config: dict):
        """Method responsible for initializing the state of the transformer model

        Args:
            config (dict): Dictionary containing the configuration of the transformer model

        Returns:
            Dict: Dictionary containing the initial state of the transformer model
        """

        prefix = [int(config.global_batch_size), int(config.n_head)]
        quantize_q = config.quantize_q
        s_k = int(config.n_code_k)
        m = int(config.mem_len)
        d_k = int(config.d_k)
        d_v = int(config.d_v)
        # Setting variables for Key quantization
        cache = {
            'pos_offset': torch.tensor(0, dtype=torch.int32, device=config.device),
            'aggcache': {
                'upper_div_lower_k': torch.zeros([*prefix, s_k, d_v],
                                                dtype=config.dtype, device=config.device),
                'lower_k': torch.zeros([*prefix, s_k],
                                    dtype=config.dtype, device=config.device),
                'latest_doc_id': torch.zeros([prefix[0]],
                                            dtype=torch.int32, device=config.device)
            }
        }

        if not quantize_q:
            cache['xlcache'] = {  'z_k': torch.full([*prefix, m], s_k - 1, dtype=torch.int32, device=config.device), 
                                'k_hat': torch.zeros([*prefix, m, d_k], dtype=config.dtype, device=config.device),
                                'v': torch.zeros([*prefix, m, d_v], dtype=config.dtype, device=config.device),
                                'doc_ids': torch.zeros([*prefix, m], dtype=torch.int32, device=config.device)}

        return cache
    
    @staticmethod
    def rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Method responsible for shifting tensor to the right by adding padding

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Input tensor shifted to the right
        """

        *leading_shape, present_len, past_len = x.shape
        x = torch.reshape(F.pad(x, (1,0)), [*leading_shape, past_len + 1, present_len])
        x = x[..., 1:, :]
        x = torch.reshape(x, [*leading_shape, present_len, past_len])
        return x
    

    @staticmethod
    def get_causal_mask(block_len: int, mem_len: int, invalid_len: torch.Tensor, with_locality, device):
        """
        Method responsible for generating causal mask for the transformer model

        Args:
            block_len (int): size of block-based computation
            mem_len (int): memory steps in sequence length to consider
            invalid_len (torch.Tensor): invalid length of sequence for masking. This relates to the memory lower
                                        as it loops through the sequence
            with_locality (bool): _description_

        Returns:
            _type_: _description_
        """

        assert block_len > 0 and mem_len >= 0
        i = torch.arange(block_len, device=device).unsqueeze(-1)
        j = torch.arange(mem_len + block_len, device=device).unsqueeze(0)
        alloc_mask = j >= torch.tensor([invalid_len], device=device).unsqueeze(0)
        causal_mask = (j - mem_len) <= i
        window_mask = j >= i
        keep_mask = alloc_mask & causal_mask
        if with_locality:
            keep_mask = keep_mask & window_mask
        return keep_mask
    
    @staticmethod
    def get_agg_biases(lower: torch.Tensor) -> torch.Tensor:
        result = torch.where(
            torch.eq(lower, torch.zeros_like(lower)),
            -MASK_INFTY_APPROX,
            torch.log(torch.maximum(lower, torch.ones_like(lower))),
        )
        return result
    

    def _get_q(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tilde (torch.Tensor): Normalized input

        Returns:
            torch.Tensor: Q matrix
        """

        q = self.q_proj(x_tilde)
        q = rearrange(q, 't b s (h d) -> t b s h d', h=self.n_head)
        q = self.q_ln(q) * (self.tau**-0.5)
        q = rearrange(q, 't b s h d -> t b h s d', h=self.n_head)
        return q
    
    def _get_kvg(self, x_tilde: torch.Tensor, verbose: bool=False):
        """
        Method that computes the keys, values, and gates for the transformer model

        Args:
            x_tilde (torch.Tensor): Input

        Returns:
            Tuple[torch.Tensor, Torch.Tensor, torch.Tensor]: The keys, values,
            and g matrix projectors with respect to the input
        """
        # bsz, present_len, _ = x_tilde.shape

        _,_, present_len, _ = x_tilde.shape
        kvg = self.kvg_proj(x_tilde)
        k, v, g = torch.split(kvg, [self.q_ch, self.v_ch, self.v_ch], dim=-1)
        if verbose:
            pass
            # printc((k.shape, "shape of k before rearrangement"), color='green')
            # printc((v.shape, "shape of v before rearrangement"), color='green')
            # printc((g.shape, "shape of g before rearrangement"), color='green')
            # printc((present_len, self.n_head * self.d_k), color='green')

        assert k.shape[-2:] == (present_len, self.n_head * self.d_k)
        assert v.shape[-2:] == (present_len, self.n_head * self.d_v)
        assert g.shape[-2:] == (present_len, self.n_head * self.d_v)
        k = rearrange(k, 't b s (h d) -> t b h s d', h=self.n_head)
        v = rearrange(v, 't b s (h d) -> t b h s d', h=self.n_head)
        k = self.k_ln(k) * (self.tau**-0.5)
        v = self.SiLU(v)
        g = self.SiLU(g)
        # k,v = rearrange(k, 't b h s d -> t b s (h d)'), rearrange(v, 't b h s d -> t b s (h d)')
        return k, v, g
    

    def get_xl_helpers(self):
        """
        Method that computes positional encoding for the transformer model
        """
        # compute helpers for xl biases (z dai et al.; 2019)
        xl_r = SinusoidPE.get(length=self.mem_len + self.block_len,
                            width=self.d_model,
                            lam=self.pe_lam,
                            flip=True, device=self.device)
        assert xl_r.shape == (self.mem_len + self.block_len, self.d_model)

        xl_r = self.dropSin(xl_r)
        xl_r = self.f_proj(xl_r)
        xl_r = torch.reshape(xl_r, [self.mem_len + self.block_len, self.n_head, self.d_k])
        xl_r = torch.permute(xl_r, (1, 0, 2))
        xl_r = xl_r * (self.tau**-0.5)
        xl_u = torch.reshape(self.xl_u, [1, self.n_head, 1, self.d_k]) * (self.tau**-0.5)
        xl_v = torch.reshape(self.xl_v, [1, self.n_head, 1, self.d_k]) * (self.tau**-0.5)

        return xl_r, xl_u, xl_v
    
    def apply_mask_recent_scores(self, recent_scores_ac, recent_scores_bd, position_offset: int):
        recent_scores_bd = recent_scores_bd * self.get_causal_mask(
            block_len=self.block_len,
            mem_len=self.mem_len,
            invalid_len=F.relu(self.mem_len - position_offset),
            with_locality=True,
            device=self.device
        )[None, None, ...].to(torch.int32)
        recent_scores = recent_scores_ac + recent_scores_bd
        keep_mask = self.get_causal_mask(
        block_len=self.block_len,
        mem_len=self.mem_len,
        invalid_len=F.relu(self.mem_len - position_offset),
        with_locality=not self.agg_cache,
        device=self.device
        )[None, None, ...].to(torch.int32)
        recent_scores = recent_scores * keep_mask + MASK_INFTY_APPROX * (1 - keep_mask)
        return recent_scores
    

    def update_state(self, recent_z_k: torch.Tensor, recent_k_hat: torch.Tensor, 
                    recent_v: torch.Tensor, recent_doc_ids: torch.Tensor, state: dict):
        aggcache = state["aggcache"]
        recent_z_k = recent_z_k.long()

        # compute kronecker deltas; invalid z's from xlcache init encode to zero vecs
        delta = F.one_hot(
            recent_z_k[..., : -self.mem_len],
            num_classes=self.n_code,
        )

        # compute new position offset
        new_pos_offset = 0
        new_lower_k = torch.add(aggcache["lower_k"], torch.sum(delta, axis=-2))
        # compute updated upper cache variable (stored in relative format for stability)
        # i.e., we compute new_upper_div_lower by dividing axis S by counts in new_lower
        f1 = aggcache["lower_k"] / torch.clip(new_lower_k, min=1.0)
        f2 = delta / torch.unsqueeze(torch.clip(new_lower_k, min=1.0), -2)
        new_upper_div_lower_k = torch.add(
            f1[..., None] * aggcache["upper_div_lower_k"],
            torch.einsum("bhls,bhlv->bhsv", f2, recent_v[..., : -self.mem_len, :]),
        )

        xlcache = dict(
                z_k=recent_z_k[..., : -self.mem_len :],
                k_hat=recent_k_hat[..., : -self.mem_len :],
                v=recent_v[..., : -self.mem_len :],
                doc_ids=recent_doc_ids[..., : -self.mem_len :],
            )

        aggcache=dict(
            lower_k=new_lower_k,
            upper_div_lower_k=new_upper_div_lower_k,
            latest_doc_id_k=recent_doc_ids[..., : -self.mem_len - 1]
        )

        if not self.grad_thru_cache:
            aggcache = dict(map(self._detach_item, aggcache.items()))
            xlcache = dict(map(self._detach_item, xlcache.items()))
            new_pos_offset = new_pos_offset.detach()

        new_state = dict(
            pos_offset=new_pos_offset,
            xlcache=xlcache,
            aggcache=aggcache
        )

        return new_state

    @staticmethod
    def _detach_item(item):
        key, value = item
        return key, value.detach()