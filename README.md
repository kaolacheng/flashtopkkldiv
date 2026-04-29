# Flash-TopkKldiv

Triton kernels for memory-efficient sparse KL divergence computation with up to **5.5x speedup** and **~70% memory reduction**.

## Installation

```bash
pip install .
# or in development mode:
pip install -e .
```

### Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | >=3.9 | |
| PyTorch | >=2.0 | CUDA 12+ recommended |
| Triton | >=2.0 | NVIDIA GPUs (Turing+ / sm_75+) |

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

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Vocab Size | 115,936 |
| Hidden Dim | 1,024 |
| Batch Size | 4 |
| Seq Len | 128 |
| Top-K | 512 |

**Speedup:** ~5x (varies with K, 2.3x~27x)  
**Memory Saved:** ~70% (~1 GiB saved for typical configurations)

### Different K Values

| Top-K (K) | Original (ms) | Fast (ms) | Speedup | Memory Saved |
|-----------|--------------|-----------|---------|--------------|
| 64 | 24.1 ± 0.9 | 0.89 ± 0.04 | **~27x** | ~128 MiB |
| 256 | 22.9 ± 0.5 | 2.64 ± 0.08 | **~8.7x** | ~512 MiB |
| 512 | 24.5 ± 1.7 | 4.9 ± 0.1 | **~5x** | ~1 GiB |
| 1024 | 23.2 ± 1.0 | 9.96 ± 0.9 | **~2.3x** | ~2 GiB |

> **Note:** Smaller K yields higher speedup due to greater sparsity. At larger K, the sparse kernel overhead becomes more significant relative to the benefit, so the speedup decreases but remains substantial.

## Running Benchmarks

```bash
python benchmark.py --device cuda --runs 5
python benchmark.py --topk 256 --vocab_size 115936 --hidden_dim 1024
```

## Limitations & Important Notes

- **Teacher Top-K must be pre-computed.** This library assumes `teacher_indices` and `teacher_probs` are provided as inputs. It optimizes the *student side* only. For **online distillation** (same GPU, simultaneous teacher/student forward pass), you still need techniques like [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint-support.html) or offloading for the teacher model.
- **NVIDIA GPUs only.** Triton supports Turing (sm_75) and newer architectures. Older architectures may experience reduced performance or compatibility issues.
- **Single-batch optimization.** The benchmark uses batch_size=4, seq_len=128. For larger batches or longer sequences, the speedup ratio should scale similarly since the kernel is batch-independent.

## Future Work

- Fuse KL divergence calculation (Softmax + KL) into the Triton kernel to reduce HBM reads/writes
- Support AMD ROCm via triton-rocm backend
- Benchmark with different GPU architectures and larger hidden dimensions

## License

MIT License. See [LICENSE](LICENSE) for details.
