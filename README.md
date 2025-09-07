# Vector Quantization Attention

A PyTorch implementation of efficient attention mechanisms using vector quantization techniques. This project explores and compares different attention mechanisms including VQ-Attention, HyperAttention, and standard multi-head attention.

## Overview

This project implements and benchmarks several attention mechanisms:

- **VQ-Attention (VQAttentionQK)**: Uses vector quantization for both queries and keys to reduce computational complexity
- **HyperAttention**: Efficient attention using Angular Locality Sensitive Hashing (LSH) 
- **Vanilla Multi-Head Attention**: Standard transformer attention for comparison

The implementation focuses on making attention mechanisms more efficient for long sequences while maintaining performance.

## Features

- Vector quantization-based attention with learnable codebooks
- Angular LSH-based attention for approximate similarity search
- Configurable attention parameters (heads, dimensions, sequence lengths)
- Performance benchmarking across different sequence lengths
- Support for both causal and non-causal attention
- GPU acceleration with CUDA and MPS support

## Installation

1. Create a conda environment using the provided environment file:
```bash
conda env create -f environment.yml
conda activate torch
```

2. Alternatively, install the key dependencies manually:
```bash
pip install torch torchvision einops PyYAML flash-attn
```

## Quick Start

### Configuration

Modify `conf.yml` to set your desired parameters:

```yaml
# Model parameters
d_model: 1024      # Model dimension
n_head: 1          # Number of attention heads
n_code: 128        # Number of VQ codes
sequence_len: 1024 # Input sequence length
block_len: 32      # Block size for processing

# Training parameters
global_batch_size: 4
device: 'cuda'     # 'cuda', 'mps', or 'cpu'
```

### Running Experiments

```python
# Run attention mechanism comparison
python test.py

# Or use the interactive notebook
jupyter notebook notebook.ipynb
```

### Basic Usage

```python
import torch
from transformer_vq.nn.attn_vq import VQAttentionQK
from transformer_vq.nn.config_spec import TransformerConfig
import yaml

# Load configuration
with open('conf.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Create model
model_config = TransformerConfig(**config)
vq_attn = VQAttentionQK(model_config)

# Generate sample data
batch_size, seq_len, d_model = 4, 1024, config['d_model']
x = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output = vq_attn(x)
```

## Project Structure

```
vq_attn/
├── transformer_vq/          # VQ-Attention implementation
│   └── nn/
│       ├── attn_vq.py      # Main VQ attention module
│       ├── base_attn.py    # Base attention class
│       ├── vq.py           # Vector quantization utilities
│       ├── config_spec.py  # Configuration classes
│       └── ...
├── hyper_attn/             # HyperAttention implementation
│   ├── hyper_attn.py       # Main HyperAttention module
│   ├── angular_lsh.py      # LSH implementation
│   └── utils.py            # Utility functions
├── test.py                 # Benchmarking script
├── vanilla.py              # Standard attention implementation
├── conf.yml                # Configuration file
├── environment.yml         # Conda environment
└── notebook.ipynb          # Interactive experiments
```

## Benchmarking

The project includes comprehensive benchmarking across different sequence lengths:

```python
# Benchmark attention mechanisms
sequence_lengths = [1024 * 2**i for i in range(0, 9)]  # 1K to 256K tokens
python test.py
```

Performance is measured for:
- VQ-Attention with quantized queries and keys  
- HyperAttention with LSH-based approximation
- Vanilla multi-head attention as baseline

## Key Components

### VQ-Attention
- Quantizes queries and keys using learnable codebooks
- Reduces attention complexity from O(n²) to O(n×k) where k << n
- Maintains attention quality through learned vector quantization

### HyperAttention  
- Uses Angular LSH for efficient similarity search
- Samples and blocks tokens for approximate attention
- Particularly effective for very long sequences

### Configuration
The `TransformerConfig` class manages all model parameters including:
- Model dimensions (d_model, d_k, d_v)
- Quantization settings (n_code, n_code_q, n_code_k)
- Training hyperparameters (dropout rates, learning rates)

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{vq_attention,
  title={Vector Quantization Attention},
  author={Juan Muneton Gallego},
  year={2024},
  url={https://github.com/juanmuneton/vq_attn}
}
```
