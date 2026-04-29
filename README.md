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

### Pre-computing Teacher Top-K (Online Distillation)

When running **online distillation** on a single GPU, computing `teacher_logits` for the full vocabulary is memory-intensive (`[B*S, V]`). The recommended approach is to use **chunked top-k computation**, processing the teacher's logits in smaller chunks to avoid materializing the full `[N, V]` tensor:

```python
import torch.nn.functional as F

def compute_teacher_topk_chunked(teacher_logits, chunk_size=1024):
    """Compute top-k indices and probs without materializing [N, V]."""
    N = teacher_logits.shape[0]
    V = teacher_logits.shape[1]
    
    all_indices = []
    all_probs = []
    
    for i in range(0, V, chunk_size):
        chunk = teacher_logits[:, i : i + chunk_size]  # [N, chunk_size]
        k_chunk = min(topk, chunk_size)
        k_vals, k_idx = torch.topk(chunk, k=k_chunk, dim=-1)  # [N, k_chunk]
        
        chunk_probs = F.softmax(k_vals / temperature, dim=-1)  # [N, k_chunk]
        
        all_indices.append(k_idx)       # accumulate indices
        all_probs.append(chunk_probs)   # accumulate probs
    
    # Concatenate along K dimension to get [N, topk]
    teacher_indices = torch.cat(all_indices, dim=1)[:, :topk]
    teacher_probs = torch.cat(all_probs, dim=1)[:, :topk]
    
    return teacher_indices, teacher_probs
```

This reduces the peak memory for teacher side from **~V×B×S bytes** to just **~chunk_size×B×S bytes**. For `V=115K` with batch 4 × seq 128: full logits tensor is ~23 MiB (FP32), but chunking reduces this further and allows streaming.

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

### Different Hidden Dimensions (Batch=4, SeqLen=128, Top-K=512)

| Hidden Dim | Original (ms) | Fast (ms) | Speedup | Memory Saved |
|------------|--------------|-----------|---------|--------------|
| 512 | 10.7 ± 0.4 | 2.9 ± 0.7 | **~3.7x** | ~512 MiB |
| 768 | 19.9 ± 1.8 | 4.3 ± 0.2 | **~4.7x** | ~768 MiB |
| 1,024 | 24.5 ± 1.7 | 4.9 ± 0.1 | **~5x** | ~1 GiB |
| 2,048 | 46.7 ± 4.4 | 10.3 ± 0.4 | **~4.6x** | ~2 GiB |
| 4,096 | 104.4 ± 4.5 | 24.1 ± 1.7 | **~4.3x** | ~4 GiB |

> **Note:** Speedup remains stable across hidden dimensions. Larger hidden dim saves more absolute memory but the speedup ratio is similar since the kernel launch overhead scales with batch size, not D.

### Different Sequence Lengths (Batch=4, HiddenDim=1024, Top-K=512)

| Seq Len | Original (ms) | Fast (ms) | Speedup | Memory Saved |
|---------|--------------|-----------|---------|--------------|
| 64 | 12.0 ± 0.6 | 2.5 ± 0.1 | **~4.7x** | ~512 MiB |
| 128 | 24.5 ± 1.7 | 4.9 ± 0.1 | **~5x** | ~1 GiB |
| 256 | 46.3 ± 0.6 | 9.5 ± 0.1 | **~4.9x** | ~2 GiB |

> **Note:** Speedup is stable across sequence lengths. Linear scaling with seq_len confirms the kernel scales well for longer sequences.
python benchmark.py --topk 256 --vocab_size 115936 --hidden_dim 1024
```

## Limitations & Important Notes

- **Teacher Top-K must be pre-computed.** This library assumes `teacher_indices` and `teacher_probs` are provided as inputs. It optimizes the *student side* only. For **online distillation** (same GPU, simultaneous teacher/student forward pass), you still need techniques like [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint-support.html) or offloading for the teacher model. See above for a recommended chunked-topk implementation that reduces teacher memory usage.
- **NVIDIA GPUs only.** Triton supports Turing (sm_75) and newer architectures. Older architectures may experience reduced performance or compatibility issues.
- **Precision considerations.** The library is designed for FP32 computation. For mixed precision training (FP16/BF16), note:
  - KL divergence involves `log_softmax` which can produce numerical overflow in lower precisions — use FP32 accumulators for the logit computation step.
  - The Triton kernel uses FP32 internal accumulation internally to prevent numerical issues.
  - Recommended: convert teacher embeddings and hidden states to FP32 before passing to `kl_div_fast`, then cast back if needed.

## Future Work

- Fuse KL divergence calculation (Softmax + KL) into the Triton kernel to reduce HBM reads/writes
- Support AMD ROCm via triton-rocm backend
- Benchmark with different GPU architectures and larger hidden dimensions

## License

MIT License. See [LICENSE](LICENSE) for details.
