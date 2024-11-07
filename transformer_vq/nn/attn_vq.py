import torch
import torch.nn.functional as F
from einops import rearrange
from dataclasses import asdict

from transformer_vq.nn.base_attn import VQAttention
from transformer_vq.nn.vq import LearnableVQ
from transformer_vq.nn.config_spec import TransformerConfig


class VQAttentionQK(VQAttention):
    def __init__(self, config: TransformerConfig):
        super(VQAttentionQK, self).__init__(config)
        self.n_code_q = config.n_code_q
        try:
            config_q = asdict(self.config)
        except:
            config_q = self.config.__dict__
        config_q.update({'d_model': self.d_k})
        config_q.update({'n_code': self.n_code_q})
        self.quantizer_q = LearnableVQ(config_q)
        self.c_q_k = torch.Tensor().to(dtype=self.d_type)
        self._reset_codebook_mult()  # Initial reset
        # Register full backward hook
        self.register_full_backward_hook(self._reset_after_backward)

    def _reset_codebook_mult(self):
        self.c_k = self.quantizer_k.get_codebook().to(dtype=self.d_type)
        self.c_q = self.quantizer_q.get_codebook().to(dtype=self.d_type)
        self.c_q_k = torch.exp(torch.einsum("hqd, hkd -> hqk", self.c_q, self.c_k)).to(dtype=self.d_type)

    def _reset_after_backward(self, module, grad_input, grad_output):
        """Hook to reset c_q_k after backward pass."""
        self.reset_codebook_mult()
        return grad_input

    def mount_inference_codebooks(self, c_k = None, c_q = None):
        self.quantizer_k.update_index(codebook=c_k)
        self.quantizer_q.update_index(codebook=c_q)

    def vq_attn(self, present_q: torch.Tensor, present_k: torch.Tensor,
                present_v: torch.Tensor, present_doc_ids: torch.Tensor,
                state: dict, loss_mask: torch.Tensor,
                causal: bool = True) -> dict:
        # Implementation continues
        vq_output_dict_k, vq_output_dict_q = self.run_vqk(present_k=present_k,
                                                  present_q=present_q,
                                                  loss_mask=loss_mask)

        # Accessing the codebook vectors along with the shortcodes
        present_z_k = vq_output_dict_k["shortcodes"]
        present_z_q = vq_output_dict_q["shortcodes"]

        if self.agg_cache:
            aggcache = state["aggcache"]

        # Performs attention step
        wv, delta_k_present, delta_k_v_present = self.attn(present_z_k=present_z_k,
                                                        present_z_q=present_z_q,
                                                        aggcache=aggcache,
                                                        present_v=present_v,
                                                        causal=causal)

        l_commit = (vq_output_dict_k["l_commit"] + vq_output_dict_q["l_commit"])
        l_codebook = (vq_output_dict_k["l_codebook"] + vq_output_dict_q["l_codebook"])


        return dict(
            attn_out=wv,
            recent_z_k=[],
            recent_k_hat=[],
            recent_v=[],
            recent_doc_ids=present_doc_ids,
            delta_k_present=delta_k_present,
            delta_k_v_present=delta_k_v_present,
            l_commit=l_commit,
            l_codebook=l_codebook,
            l_commit_k=vq_output_dict_k["l_commit"],
            l_commit_q=vq_output_dict_q["l_commit"]
        )

    def run_vq_attn(self, present_q: torch.Tensor, present_k: torch.Tensor,
                    present_v: torch.Tensor,
                    state: dict, loss_mask: torch.Tensor, causal: bool = True):
        vq_output_dict_k, vq_output_dict_q = self.run_vqk(present_k=present_k,
                                                  present_q=present_q,
                                                  loss_mask=loss_mask, return_vecs_hat=False)
        
        present_z_k = vq_output_dict_k["shortcodes"]
        present_z_q = vq_output_dict_q["shortcodes"]

        if self.agg_cache:
            aggcache = state["aggcache"]
        else:
            aggcache = None

        wv, delta_k_present, delta_k_v_present = self.attn(present_z_k=present_z_k,
                                                        present_z_q=present_z_q,
                                                        aggcache=aggcache,
                                                        present_v=present_v,
                                                        causal=causal)
        return wv, delta_k_present, delta_k_v_present
     

    def attn(self, present_z_k: torch.Tensor, present_z_q: torch.Tensor, present_v: torch.Tensor, aggcache: dict,
             causal: bool = True) -> dict:  
        
        wv, delta_k_present, delta_k_v_present = self._compute_wv(present_z_k=present_z_k,
                                                            present_z_q=present_z_q,
                                                            present_v=present_v,
                                                            aggcache=aggcache,
                                                            causal=causal)
        if wv.dim() == 4:
            wv.unsqueeze(2)
        wv = rearrange(wv, 't b h s d -> t b s (h d)', h=self.head, s=self.block_len, d=self.d_v)   
        return wv, delta_k_present, delta_k_v_present
    
    def _compute_wv(self, present_z_k: torch.Tensor, present_z_q: torch.Tensor, aggcache: dict, present_v: torch.Tensor,causal: bool = True) -> dict:
        # Computing Delta_q = C_q_k
        delta_q = F.one_hot(present_z_q.long(), num_classes=self.n_code_q).to(self.c_q_k.dtype)
        q_cqk = torch.einsum("tbhsn, hnd -> tbhsd", delta_q, self.c_q_k)

        # compute aggcache scores
        # This computes L(n-1) (Cache from previous update step)
        if self.agg_cache:
            cache_biases = aggcache["lower_k"]
            cache_scores = torch.einsum("tbhsd, bhd -> tbhsd", q_cqk, cache_biases)
        
        # Computing L(n)
        present_v = present_v.unsqueeze(0) if present_v.dim() == 4 else present_v
        present_z_k_one_hot = F.one_hot(present_z_k.long(), num_classes=self.n_code_k).to(self.d_type)
        if causal:
            delta_k_present = torch.cumsum(present_z_k_one_hot, dim=-2)
            delta_k_v_present = torch.cumsum(torch.einsum("tbhsc, tbhsd -> tbhcd", delta_k_present, present_v), dim=0)
        else:
            delta_k_present = torch.sum(present_z_k_one_hot, dim=0, keepdim=True)
            delta_k_v_present = torch.sum(torch.einsum("tbhsc, tbhsd -> tbhcd", delta_k_present, present_v), dim=0, keepdim=True)

        recent_qk_hat = torch.einsum("tbhsk, tbhmk -> tbhsm", q_cqk, delta_k_present)
        # This computes the denominator
        d = torch.sum(recent_qk_hat, axis=-1, keepdim=True)
        d = torch.cumsum(d, dim=0) if causal else torch.sum(d, axis=0, keepdim=True)

        if self.agg_cache:
            d = torch.sum(cache_scores, axis=-1, keepdim=True)
        wv = torch.einsum("tbhwl, tbhw->tbhlv", q_cqk / d, delta_k_v_present)

        if aggcache:
            wv = wv + torch.einsum("tbhls, bhsv->tbhlv", cache_scores / d, aggcache["upper_div_lower_k"])
        
        return wv, delta_k_present, delta_k_v_present
    

    def update_state(self, delta_k_present: torch.Tensor,
                        delta_k_v_present: torch.Tensor,
                        state: dict, recent_doc_ids: torch.Tensor):

        aggcache=state["aggcache"]
        delta = delta_k_present[-1]
        new_pos_offset = state['pos_offset'] + self.block_len
        new_lower_k = torch.add(aggcache["lower_k"], torch.sum(delta, axis=-2))

        f1 = aggcache["lower_k"] / torch.clip(new_lower_k, min=1.0)
        f2 = 1 / torch.unsqueeze(torch.clip(new_lower_k, min=1.0), -1)
        new_upper_div_lower_k = torch.add(
            f1[..., None] * aggcache["upper_div_lower_k"],
            f2 * delta_k_v_present[-1],
        )
        aggcache=dict(
            lower_k=new_lower_k,
            upper_div_lower_k=new_upper_div_lower_k,
            latest_doc_id_k=rearrange(recent_doc_ids, 't b s -> b (t s)')
        )
        if not self.grad_thru_cache:
            new_pos_offset = new_pos_offset.detach()
            
        new_state = dict(
        pos_offset=new_pos_offset.detach(),
        aggcache = dict(map(self._detach_item, aggcache.items()))
        )

        return new_state  

    def _build_output_dicts(self, res, attn_output_dict, state):
        new_state = self.update_state(
            delta_k_present=attn_output_dict.get("delta_k_present"),
            delta_k_v_present=attn_output_dict.get("delta_k_v_present"),
            recent_doc_ids=attn_output_dict.get("recent_doc_ids"),
            state=state,
        )
        output_dict = dict(
            res=res,
            l_commit=attn_output_dict.get("l_commit"),
            l_codebook=attn_output_dict.get("l_codebook"),
        )
        return new_state, output_dict

