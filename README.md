# Efficient Self-Attention Mechanisms Via Vector Quantization

A PyTorch implementation of novel self-attention mechanisms using vector quantization (VQ-VAE) techniques to achieve sub-quadratic runtime complexity. This research project, developed from scratch, addresses the quadratic time and space complexity challenges of transformers in long-sequence processing tasks.

## Abstract

This project presents a novel self-attention mechanism that leverages Vector Quantized Variational Autoencoders (VQ-VAEs) to achieve efficient attention computation over long contexts. By compressing input representations of keys and queries through vector quantization, our approach enables sub-quadratic runtime complexity while maintaining performance stability. Experimental results demonstrate that the VQ-based attention mechanism outperforms vanilla attention and baseline models in throughput, memory efficiency, and computational performance across sequence lengths from 10³ to 10⁵ tokens.

## Overview

This project implements and benchmarks three attention mechanisms:

- **VQ-Attention (VQAttentionQK)**: Novel approach using vector quantization for both queries and keys, achieving sub-quadratic complexity through learned codebook representations
- **HyperAttention**: Baseline efficient attention using Angular Locality Sensitive Hashing (LSH) with O(N log N) complexity
- **Vanilla Multi-Head Attention**: Standard transformer attention with O(N²) complexity for comparison

The core innovation lies in using the associative property of matrix multiplication with quantized representations, enabling precomputation of softmax operations and reducing computational overhead.

## Key Features

- **Sub-quadratic Complexity**: Reduces attention complexity from O(N²) to O(N×K) where K << N through vector quantization
- **Learnable Codebooks**: Dual codebook representations for queries and keys with learnable embeddings
- **Precomputed Softmax**: Matrix M = exp(C_Q C_K^T) can be precomputed once, significantly reducing runtime overhead
- **Scalable Performance**: Demonstrated 30x speedup over baseline methods for long sequences (up to 262K tokens)
- **Memory Efficiency**: Achieves up to 3.5 GB/s memory throughput for long-sequence tasks
- **Configurable Architecture**: Support for varying codebook sizes (256-8192 codevectors) with performance analysis
- **Comprehensive Benchmarking**: Performance evaluation across sequence lengths from 1K to 256K tokens
- **Both Attention Types**: Support for causal and non-causal attention mechanisms

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

## Experimental Results

### Performance Benchmarks

Comprehensive evaluation across sequence lengths from 1,024 to 262,144 tokens:

| Sequence Length | VQ-Causal | VQ-NonCausal | Hyper-Attn | Vanilla Attn |
|----------------|-----------|--------------|-------------|--------------|
| 1,024          | 0.58 ms   | 0.60 ms      | 12.58 ms    | 9.69 ms      |
| 16,384         | 5.24 ms   | 3.95 ms      | 197.38 ms   | 151.78 ms    |
| 65,536         | 19.76 ms  | 14.99 ms     | 794.92 ms   | 620.14 ms    |
| 262,144        | 79.18 ms  | 59.30 ms     | 3203.77 ms  | 2494.38 ms   |

**Key Findings:**
- **~30x Speedup**: VQ-Attention consistently outperforms baseline methods
- **Memory Efficiency**: Up to 3.5 GB/s throughput for VQ-NonCausal
- **Scalability**: Performance advantage increases with sequence length

### Codebook Size Analysis

Performance varies significantly with codebook size (fixed at 16K sequence length):

| Codebook Size | VQ-Causal | VQ-NonCausal | Optimal Range |
|---------------|-----------|--------------|---------------|
| 256           | 2.63 ms   | 1.87 ms      | ✓ Efficient   |
| 1,024         | 12.11 ms  | 9.46 ms      | ✓ Balanced    |
| 4,096         | 93.39 ms  | 83.78 ms     | ⚠ Diminishing |
| 8,192         | 315.18 ms | 295.59 ms    | ✗ Inefficient |

**Insight**: Careful codebook size selection is crucial - larger codebooks can underperform even hyperattention.

## Methodology

### Vector Quantization Approach

Our method extends the work of Lingle (2024) by introducing vector quantization for both queries and keys (rather than keys only). The core innovation uses the associative property of matrix multiplication:

```
W = φ_w(QK^T + B) ≈ φ_w(VQ(Q; C_Q)VQ(K; C_K)^T)
  = φ_w(Δ_Q C_Q C_K^T Δ_K^T)
  = Diag(Δ_Q M Δ_K 1)^{-1} Δ_Q M Δ_K
```

Where:
- `C_Q, C_K ∈ R^{S×D}` are learnable codebook matrices for queries and keys
- `M = exp(C_Q C_K^T)` is the precomputed softmax matrix
- `Δ_Q, Δ_K` are sparse Kronecker delta matrices for code selection
- `VQ(·; C)` denotes the vector quantization function

### Key Advantages

1. **Precomputation**: Matrix M can be computed once during training
2. **Sparse Operations**: Main computation involves constructing sparse Δ matrices
3. **Scalability**: Complexity depends on codebook size S rather than sequence length N

### Training Loss

The complete loss function combines prediction loss with VQ-VAE codebook losses:

```
L = L_pred + Σ_{i=1}^t (L_ct^{Q(i)} + L_ct^{K(i)})
```

Where codebook loss includes reconstruction, embedding, and commitment terms.

## Limitations & Future Work

### Current Limitations
- Hardware constraints prevented full CUDA optimization evaluation
- No integration with advanced tools like Triton for matrix optimization or Faiss for efficient codevector search
- Lack of formal scaling laws for larger model prediction
- Limited to inference-time optimization (training efficiency not evaluated)

### Future Research Directions
- Apply method to diverse domains to evaluate robustness
- Investigate varying quantized query and key sizes
- Develop formal scaling laws for production deployment
- Integration with modern optimization frameworks (Triton, Faiss)
- Training efficiency analysis and end-to-end performance evaluation

## Acknowledgments

This work was completed as part of CS229 at Stanford University. The algorithm was developed entirely from scratch by the authors.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{ankeney2024efficient,
  title={Efficient Self-Attention Mechanisms Via Vector Quantization},
  author={George Ankeney and Juan Muneton Gallego},
  year={2024},
  institution={Stanford University},
  note={CS229 Final Project},
  url={https://github.com/jmunetong/vq_attn}
}
```
