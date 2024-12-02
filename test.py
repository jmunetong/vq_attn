# # %load_ext autoreload
# %autoreload 2
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
    print(data.shape)
    emb = Emb(n_vocab, d_model).to(device=config['device'])
    data = emb(data)
    print(data.shape)
    return data

def empty_cache(config):
    if config['device'] == 'mps':
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def run_experiment(data, causal, config):
    d_model = config['d_model'] 
    model_config = TransformerConfig(**config)
    model = VQAttentionQK(model_config).to(device=config['device'])
    init_state = model.initial_state(model_config)
    block_len = config['block_len']
    sequence_len = config['sequence_len']
    x = data
    data = rearrange(data, 'b (l s) d -> l b s d', l=sequence_len//block_len, s=block_len)
    q,k, v, _  = model.compute_k_q_v_g(data)
    vq_output_dict_k, vq_output_dict_q  = model.run_vqk(present_k=k, present_q=q, loss_mask=torch.Tensor([1]), return_vecs_hat=False)
    present_z_k = vq_output_dict_k["shortcodes"]
    present_z_q = vq_output_dict_q["shortcodes"]
    del vq_output_dict_k, vq_output_dict_q
    empty_cache(config)
    ###### VQ Runtime
    with torch.inference_mode():
        start_time = time.time()
        wv, _, _ = model.attn(present_z_k=present_z_k,
                        present_z_q=present_z_q,
                        aggcache=init_state['aggcache'],
                        present_v=v,
                        causal=causal)
        end_time = time.time()
    vq_time = end_time - start_time
    print(f'VQ time: {vq_time}')
    ###### Hyper Attention Runtime
    del present_z_k, present_z_q
    del model
    torch.cuda.empty_cache()
    q = rearrange(q, 't b h s d -> b (t s) h  d')
    k = rearrange(k, 't b h s d -> b (t s) h  d')
    v = rearrange(v, 't b h s d -> b (t s) h  d')
    
    
    hyper_attn_time = end_time - start_time
    print(f'Hyper Attention time: {hyper_attn_time}')
    hyper_attn_time = 0
    
    vanilla_attention_model = compile_vanilla_attn(config['d_model'], config['n_head'], config['device'])
    with torch.inference_mode():
        start_time = time.time()
        vanilla_attention_model.attn(q, k, v)
        end_time = time.time()
    vanilla_attn_time = end_time - start_time
    print(f'Vanilla Attention time: {vanilla_attn_time}')
    del q, k, v, data, vanilla_attention_model
    torch.cuda.empty_cache()
    with torch.inference_mode():
        causal_attn = CausalSelfAttention(d_model, config['n_head'], 0.0, 0.0, x.shape[-2]).to(device=config['device'])
        start_time = time.time()
        causal_attn(x)
        end_time = time.time()
    print(f'Causal Attention time: {end_time - start_time}')

    return vq_time, hyper_attn_time, vanilla_attn_time

def compile_hyper_attn(dim, device):
    #(batch_size, head_size, seq_len, dim)
    block_size = 512
    sample_size = 512
    attn = HyperAttention(
        input_dim=dim, 
        block_size=block_size,
        sample_size=sample_size,
        min_seq_len=1024,
        cuda=device)
    return attn

def compile_vanilla_attn(n_dim , n_heads, device ):
    # shape(batch, sequence_length, num_heads, hidden_size)
    model = MultiHeadAttention(n_dim, n_heads, 0.0).to(device)
    return model


def main():
    sequence_lengths = [1024 * 2**i for i in range(0,9)]
    causal = True
    config = setup_config()
    vq, hyper_attn, vanilla_attn = [], [], []
    for sequence_length in sequence_lengths:
        config['sequence_len'] = sequence_length
        data = build_data(config)
        vq_time, hyper_attn_time, vanilla_attn_time = run_experiment(data, causal, config)
        vq.append(vq_time)
        hyper_attn.append(hyper_attn_time)
        vanilla_attn.append(vanilla_attn_time)

    print(vq, hyper_attn, vanilla_attn)

# def get_tensors(batch_size, seq_len, head_size, dim):
#     q = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
#     k = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
#     v = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
#     return q, k, v

if __name__ == '__main__':
    main()






