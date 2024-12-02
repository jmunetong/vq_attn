import torch

import triton
import triton.language as tl


import time 
import torch
from transformer_vq.nn.attn_vq import VQAttentionQK
from transformer_vq.nn.config_spec import TransformerConfig
from transformer_vq.nn.emb import TransformerEmbedding as Emb
import yaml
from einops import rearrange
from hyper_attn.hyper_attn import HyperAttention
from vanilla import MultiHeadAttention
# from vanilla_attn import vanilla_attention

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
    print(sequence_length)
    data = torch.randint(low=0, high=n_vocab, size=(4, sequence_length)).to(device=config['device'])
    emb = Emb(n_vocab, d_model).to(device=config['device'])
    data = emb(data)
    return data

def empty_cache(config):
    if config['device'] == 'mps':
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['vq', 'flash-attn', 'hyper-attn'],  # Possible values for `line_arg`.
        line_names=['vq', 'flash-attn', 'hyper-attn'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    d_model = config['d_model'] 
    model_config = TransformerConfig(**config)
    model = VQAttentionQK(model_config).to(device=config['device'])
    init_state = model.initial_state(model_config)
    block_len = config['block_len']
    sequence_len = config['sequence_len']
    data = rearrange(data, 'b (l s) d -> l b s d', l=sequence_len//block_len, s=block_len)
    print(f"Data shape of experiment {data.shape}")   
    q,k, v, _  = model.compute_k_q_v_g(data)
    if provider == 'vq':
        vq_output_dict_k, vq_output_dict_q  = model.run_vqk(present_k=k, present_q=q, loss_mask=torch.Tensor([1]), return_vecs_hat=False)
        present_z_k = vq_output_dict_k["shortcodes"]
        present_z_q = vq_output_dict_q["shortcodes"]
        del vq_output_dict_k, vq_output_dict_q

        with torch.inference_mode():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: model.attn(present_z_k=present_z_k,
                        present_z_q=present_z_q,
                        aggcache=init_state['aggcache'],
                        present_v=v,
                        causal=causal), quantiles=quantiles)
    if provider == 'flash-attn':
        q = rearrange(q, 't b h s d -> b (t s) h  d')
        k = rearrange(k, 't b h s d -> b (t s) h  d')
        v = rearrange(v, 't b h s d -> b (t s) h  d')
    if provider == 'hyper-attn':
        q = rearrange(q, 't b h s d -> b (t s) h  d')
        k = rearrange(k, 't b h s d -> b (t s) h  d')
        v = rearrange(v, 't b h s d -> b (t s) h  d')

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)