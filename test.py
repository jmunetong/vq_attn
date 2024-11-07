# # %load_ext autoreload
# %autoreload 2
import torch
from transformer_vq.nn.attn_vq import VQAttentionQK
from transformer_vq.nn.config_spec import TransformerConfig
from transformer_vq.nn.emb import TransformerEmbedding as Emb
import yaml
from einops import rearrange
with open('conf.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

n_vocab = config['n_vocab']
sequence_length = config['sequence_len']
batch_size = config['global_batch_size']
d_model = config['d_model']
config['d_type'] = torch.float32
config['param_dtype'] = torch.float32
config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.randint(low=0, high=n_vocab, size=(4, sequence_length))
d_model = config['d_model']
emb = Emb(n_vocab, d_model)
model_config = TransformerConfig(**config)
model = VQAttentionQK(model_config)
data = emb(data)
block_len = config['block_len']
data = rearrange(data, 'b (l s) d -> l b s d', l=block_len)
q,k, v, g = model.compute_k_q_v_g(data)
init_state = model.initial_state(model_config)
vq_output_dict_k, vq_output_dict_q  = model.run_vqk(present_k=k, present_q=q, loss_mask=torch.Tensor([1]), return_vecs_hat=False)
present_z_k = vq_output_dict_k["shortcodes"]
present_z_q = vq_output_dict_q["shortcodes"]
wv, _, _ = model.attn(present_z_k=present_z_k,
                    present_z_q=present_z_q,
                    aggcache=init_state['aggcache'],
                    present_v=v,
                    causal=True)

# data = rearrange(data, 'b s d -> b (s d)')