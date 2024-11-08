# # %load_ext autoreload
# %autoreload 2
import torch
from transformer_vq.nn.attn_vq import VQAttentionQK
from transformer_vq.nn.config_spec import TransformerConfig
from transformer_vq.nn.emb import TransformerEmbedding as Emb
import yaml
from einops import rearrange
from hyper_attn.hyper_attn import HyperAttention

def setup_experiment(sequence_length, causal, ):
    with open('conf.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_vocab = config['n_vocab']
    sequence_length = config['sequence_len']
    batch_size = config['global_batch_size']
    d_model = config['d_model']
    config['d_type'] = torch.float32
    config['param_dtype'] = torch.float32
    config['device'] = 'cuda' if torch.cuda.is_available() else 'mps' # TODO: modify this if working with a different type of device
    data = torch.randint(low=0, high=n_vocab, size=(4, sequence_length)).to(device=config['device'])
    d_model = config['d_model'] 
    emb = Emb(n_vocab, d_model).to(device=config['device'])
    model_config = TransformerConfig(**config)
    model = VQAttentionQK(model_config).to(device=config['device'])
    init_state = model.initial_state(model_config)
    data = emb(data)
    block_len = config['block_len']
    data = rearrange(data, 'b (l s) d -> l b s d', l=block_len)
    q,k, v, _  = model.compute_k_q_v_g(data)
    vq_output_dict_k, vq_output_dict_q  = model.run_vqk(present_k=k, present_q=q, loss_mask=torch.Tensor([1]), return_vecs_hat=False)
    present_z_k = vq_output_dict_k["shortcodes"]
    present_z_q = vq_output_dict_q["shortcodes"]
   
    with torch.inference_mode():
        wv, _, _ = model.attn(present_z_k=present_z_k,
                        present_z_q=present_z_q,
                        aggcache=init_state['aggcache'],
                        present_v=v,
                        causal=True)
    hyper_attn = compile_hyper_attn(causal, d_model)    
    with torch.inference_mode():
        #TODO: ENSURE Q,K,V ARE IN THE CORRECT SHAPE
        hyper_attn(q, k, v, causal=causal)
    


            
def compile_hyper_attn(causal, dim):
    #(batch_size, head_size, seq_len, dim)
    block_size = 256
    sample_size = 256
    cuda = 'cuda' if torch.cuda.is_available() else 'mps'
    attn = HyperAttention(
        input_dim=dim, 
        block_size=block_size,
        sample_size=sample_size,
        min_seq_len=1024,
        cuda=cuda)
    return attn


def compile_flash_attn():
    from 
    pass

def compile_vanilla_attn():
    queries = torch.randn(4, 2048, 1, 128)
    keys = torch.randn(4, 2048, 1, 128)
    values = torch.randn(4, 2048, 1, 128)
    # shape(batch, sequence_length, num_heads, hidden_size)
    attention(queries, keys, values)[0].shape


if __name__ == '__main__':
    setup_experiment()





