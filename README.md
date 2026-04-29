# Flash-TopkKldiv

Triton kernels for memory-efficient sparse KL divergence computation with up to **5.5x speedup** and **~70% memory reduction**.

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

## Core Optimization

The key bottleneck in KL divergence computation with large vocabularies is the **top-k embedding lookup** — a naive approach materializes an intermediate `[batch_size * seq_len, vocab_size]` tensor to get logits, then gathers the top-k entries. This wastes both memory and compute since we only need `K` out of `V` embeddings.

This library replaces that with a custom [Triton](https://github.com/triton-lang/triton) kernel (`sparse_index_matmul`) that computes `hidden_state @ embedding[teacher_indices]` **directly**, without materializing the full intermediate tensor:

| Aspect | Original (`kl_div_org`) | Optimized (`kl_div_fast`) |
|--------|--------------------------|---------------------------|
| Approach | Full embedding matmul → gather top-k | Sparse matmul with Triton kernel |
| Intermediate memory | `[N, V]` dense tensor (~1 GiB) | None — computed on-the-fly |
| Kernel | PyTorch native (gather + scatter) | Custom Triton forward + backward kernels |

## Kernels

- `sparse_index_matmul` — Indexed sparse matrix multiplication (Triton)
- `sparse_index_matmul_backward` — Autograd backward for x and e gradients
- `kl_div_org` — Original KL divergence via explicit gather (memory-heavy)
- `kl_div_fast` — Optimized KL divergence via sparse matmul (memory-efficient, ~5.5x faster)

## Benchmark Results

| Parameter | Value |
|-----------|-------|
| Vocab Size | 115,936 |
| Hidden Dim | 1,024 |
| Batch Size | 4 |
| Seq Len | 128 |
| Top-K | 512 |

**Speedup:** ~5.5x  
**Memory Saved:** ~70% (~1 GiB saved for typical configurations)

## Running Benchmarks

```bash
python benchmark.py --device cuda --runs 5
```

## License

MIT License. See [LICENSE](LICENSE) for details.
