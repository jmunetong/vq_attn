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
from vanilla_attn import vanilla_attention

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
    data = torch.randint(low=0, high=n_vocab, size=(4, sequence_length)).to(device=config['device'])
    emb = Emb(n_vocab, d_model).to(device=config['device']).no_grad()
    data = emb(data)
    return data

def empty_cache(config):
    if config['device'] == 'mps':
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def run_experiment(causal, config):
    d_model = config['d_model'] 
    model_config = TransformerConfig(**config)
    model = VQAttentionQK(model_config).to(device=config['device'])
    init_state = model.initial_state(model_config)
    model.compile()
    block_len = config['block_len']
    data = rearrange(data, 'b (l s) d -> l b s d', l=block_len)
    q,k, v, _  = model.compute_k_q_v_g(data)
    vq_output_dict_k, vq_output_dict_q  = model.run_vqk(present_k=k, present_q=q, loss_mask=torch.Tensor([1]), return_vecs_hat=False)
    present_z_k = vq_output_dict_k["shortcodes"]
    present_z_q = vq_output_dict_q["shortcodes"]
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
    ###### Hyper Attention Runtime
    hyper_attn = compile_hyper_attn(causal, d_model)    
    with torch.inference_mode():
        #TODO: ENSURE Q,K,V ARE IN THE CORRECT SHAPE
        start_time = time.time()
        hyper_attn(q, k, v, causal=causal)
        end_time = time.time()
    hyper_attn_time = end_time - start_time

    with torch.inference_mode():
        start_time = time.time()
        compile_vanilla_attn(q, k, v)
        end_time = time.time()
    vanilla_attn_time = end_time - start_time

    return vq_time, hyper_attn_time, vanilla_attn_time

    
def compile_hyper_attn(causal, dim, device):
    #(batch_size, head_size, seq_len, dim)
    block_size = 256
    sample_size = 256
    attn = HyperAttention(
        input_dim=dim, 
        block_size=block_size,
        sample_size=sample_size,
        min_seq_len=1024,
        cuda=device)
    return attn

def compile_vanilla_attn(queries, keys, values):
    # shape(batch, sequence_length, num_heads, hidden_size)
    vanilla_attention(queries, keys, values)

def main():
    sequence_lengths = [1024 * i*8 for i in range(40)]
    causal = True
    config = setup_config()
    vq, hyper_attn, vanilla_attn = [], [], []
    for sequence_length in sequence_lengths:
        config['sequence_len'] = sequence_length
        data = build_data(config)
        vq_time, hyper_attn_time, vanilla_attn_time = run_experiment(causal, config)
        vq.append(vq_time)
        hyper_attn.append(hyper_attn_time)
        vanilla_attn.append(vanilla_attn_time)



if __name__ == '__main__':



    setup_experiment()





