import torch
from einops import rearrange
import time 
import torch
from transformer_vq.nn.attn_vq import VQAttentionQK
from transformer_vq.nn.config_spec import TransformerConfig
from transformer_vq.nn.emb import TransformerEmbedding as Emb
import yaml
from einops import rearrange
from hyper_attn.hyper_attn import HyperAttention
from vanilla import MultiHeadAttention
from causal_masker import CausalSelfAttention

def setup_config():
    with open('conf.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['d_type'] = torch.float32
    config['param_dtype'] = torch.float32
    config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' # TODO: modify this if working with a different type of device
    return config

def build_data(config):
    n_vocab = config['n_vocab']
    sequence_length = config['sequence_len']
    d_model = config['d_model']
    data = torch.randint(low=0, high=n_vocab, size=(config['global_batch_size'], sequence_length)).to(device=config['device'])
    emb = Emb(n_vocab, d_model).to(device=config['device'])
    data = emb(data)
    return data

def empty_cache(config):
    if config['device'] == 'mps':
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def run_hyper_attn(hyper_attn, q, k, v, causal):
    with torch.inference_mode():
        for i in range(q.shape[0]):
            hyper_attn(q[i], k[i], v[i], causal=causal)

def build_test_pipeline(config, data):
    d_model = config['d_model'] 
    model_config = TransformerConfig(**config)
    model = VQAttentionQK(model_config).to(device=config['device'])
    init_state = model.initial_state(model_config)
    block_len = config['block_len']
    sequence_len = config['sequence_len']
    data = rearrange(data, 'b (l s) d -> l b s d', l=sequence_len//block_len, s=block_len)
    q,k, v, _  = model.compute_k_q_v_g(data)
    return q, k, v, init_state, d_model, model

def get_short_codes(model, q, k):
    vq_output_dict_k, vq_output_dict_q  = model.run_vqk(present_k=k, present_q=q, loss_mask=torch.Tensor([1]), return_vecs_hat=False)
    return vq_output_dict_k["shortcodes"], vq_output_dict_q["shortcodes"]


def compile_hyper_attn(dim, device, block_size=512, sample_size=64):
    attn = HyperAttention(
        input_dim=dim, 
        block_size=block_size,
        sample_size=sample_size,
        min_seq_len=32,
        cuda=False).to(device=device)
    return attn

def compile_vanilla_attn(n_dim , n_heads, device ):
    # shape(batch, sequence_length, num_heads, hidden_size)
    model = MultiHeadAttention(n_dim, n_heads, 0.0).to(device)
    return model

def run_vanilla_attn_non_causal(q, k, v, vanilla_attention_model):
    with torch.inference_mode():
        for i in range(q.shape[0]):
            vanilla_attention_model.attn(q[i], k[i], v[i])

def run_vq_attn(model, present_z_k, present_z_q, v, init_state, causal=True):
    with torch.inference_mode():
        model.attn(present_z_k=present_z_k,
                        present_z_q=present_z_q,
                        aggcache=init_state['aggcache'],
                        present_v=v,
                        causal=causal)
def compile_vanilla_causal(config, d_model, block_len):
    return CausalSelfAttention(d_model, config['n_head'], 0.0, 0.0, block_len).to(device=config['device'])

def run_vanilla_attn_causal(q, k, v, causal_attn):
    with torch.inference_mode():
        for i in range(q.shape[0]):
            causal_attn(q[i], k[i], v[i])
