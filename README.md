# Flash-TopkKldiv

Triton kernels for sparse operations in KL divergence computation.

## Installation

```bash
pip install .
# or in development mode:
pip install -e .
```

## Usage

### KL Divergence (Student Side Optimization)

```python
import torch
from flashtopkkldiv import kl_div_org, kl_div_fast

# Prepare teacher top-k indices and probabilities
teacher_indices = torch.tensor([...])  # [B*S, K]
teacher_probs = torch.tensor([...])    # [B*S, K]

# Compute KL divergence
loss_slow = kl_div_org(embedding, hidden_state, teacher_indices, teacher_probs)
loss_fast = kl_div_fast(embedding, hidden_state, teacher_indices, teacher_probs)
```

### Sparse Index Matrix Multiplication

```python
import torch
from flashtopkkldiv import sparse_index_matmul_lib, sparse_index_matmul

# Forward pass via Triton kernel
x = torch.randn(N, D, requires_grad=True)
e = torch.tensor(embeddings)  # [V, D]
idx = torch.tensor(indices)   # [N, K]

output = sparse_index_matmul(x, e, idx)

# Or access the bare op
output = sparse_index_matmul_lib(x, e, idx)
```

## Kernels

- `sparse_index_matmul` — Indexed sparse matrix multiplication (Triton)
- `sparse_index_matmul_backward` — Autograd backward for x and e gradients
- `kl_div_org` — Original KL divergence via explicit gather (memory-heavy)
- `kl_div_fast` — Optimized KL divergence via sparse matmul (memory-efficient, ~6x faster)

## Benchmark Results

| Parameter | Value |
|-----------|-------|
| Vocab Size | 115,936 |
| Hidden Dim | 1,024 |
| Batch Size | 4 |
| Seq Len | 128 |
| Top-K | 512 |

**Speedup:** ~5.8x  
**Memory Saved:** ~1 GiB (intermediate gather tensor)

## Performance Comparison

The `kl_div_fast` implementation avoids materializing the intermediate `[N, K, D]` gather tensor by using sparse index matrix multiplication directly:

- **Original**: Projects hidden states to full vocabulary space → gathers top-k logits
- **Fast**: Directly computes dot product with selected embeddings via Triton kernel
