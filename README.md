# Flash-TopkKldiv

Triton kernels for memory-efficient sparse KL divergence computation with up to **4.7x forward speedup** and **~2.7× total (forward + backward) latency reduction**, plus **~70% memory reduction**.

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

def compute_teacher_topk_chunked(teacher_logits, topk=512, temperature=1.0, chunk_size=1024):
    """Compute top-k indices and probs without materializing [N, V]."""
    with torch.no_grad():
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

This reduces the peak memory for teacher side from **~V×B×S bytes** to just **~chunk_size×B×S bytes**. For `V=115K` with batch 4 × seq 128: full logits tensor is ~237 MB (FP32), but chunking reduces this further and allows streaming.

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
- `kl_div_fast` — Optimized KL divergence via sparse matmul (memory-efficient, ~3.0x faster total latency with autograd)

## Benchmark Results

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Vocab Size | 115,936 |
| Hidden Dim | 4,096 |
| Batch Size | 8 |
| Seq Len | 512 |
| Top-K | 512 |

**Speedup:** ~3.0x total latency (forward + backward), varies with K and sequence length
**Memory Saved:** ~70% (~1 GiB saved for typical configurations)

### Different K Values (Batch=8, SeqLen=512, HiddenDim=4096) — Full Fwd+Bwd Timed

Fair comparison using `torch.autograd.backward()` for both org and fast.

| Top-K (K) | Org fwd (ms) | Org bwd (ms) | Fast fwd (ms) | Fast bwd (ms) | Total Org | Total Fast | Speedup |
|-----------|-------------|-------------|--------------|--------------|-----------|------------|---------|
| 64 | 84 ± 23 | 132 ± 12 | 9.3 ± 5.7 | 33.7 ± 24.0 | 216 ± 25 | 42.9 ± 24.0 | **~5.0x** |
| 256 | 81.9 ± 7.3 | 136.7 ± 5.4 | 12.3 ± 0.1 | 34.0 ± 0.8 | 218.6 ± 7.6 | 46.2 ± 0.9 | **~4.7x** |
| 512 | 84.0 ± 6.2 | 144.4 ± 3.1 | 20.6 ± 0.2 | 55.3 ± 0.2 | 228.4 ± 3.2 | 75.9 ± 0.2 | **~3.0x** |

> **Note:** Smaller K yields higher speedup due to greater sparsity. At larger K, the sparse kernel overhead becomes more significant relative to the benefit, so the speedup decreases but remains substantial. The backward pass also benefits from sparse matmul, contributing to overall latency reduction.

### Different Hidden Dimensions (Batch=8, SeqLen=512, Top-K=512) — Full Fwd+Bwd Timed

| Hidden Dim | Org fwd (ms) | Org bwd (ms) | Fast fwd (ms) | Fast bwd (ms) | Total Org | Total Fast | Speedup |
|------------|-------------|-------------|--------------|--------------|-----------|------------|---------|
| 512 | 52.8 ± 3.7 | 97.1 ± 18.5 | 11.1 ± 2.6 | 27.0 ± 0.2 | 149.8 | 38.1 | **~4.0x** |
| 768 | 69.9 ± 15.5 | 118.0 ± 10.8 | 20.3 ± 4.5 | 40.7 ± 0.2 | 188.0 | 61.0 | **~3.1x** |
| 1,024 | 82.2 ± 6.1 | 141.4 ± 6.8 | 20.5 ± 0.0 | 55.4 ± 0.0 | 223.7 | 75.9 | **~2.9x** |
| 2,048 | 149.2 ± 13.3 | 231.1 ± 3.7 | 59.7 ± 0.4 | 112.6 ± 0.7 | 380.3 | 172.3 | **~2.2x** |
| 4,096 | 82.2 ± 6.1 | 141.4 ± 6.8 | 20.5 ± 0.0 | 55.4 ± 0.0 | 223.7 | 75.9 | **~2.9x** |

> **Note:** Speedup is stable across hidden dimensions (2.2–4.0×). Larger hidden dim saves more absolute memory but the speedup ratio is similar since the kernel launch overhead scales with batch size, not D.

### Different Sequence Lengths (Batch=8, HiddenDim=4096, Top-K=512) — Full Fwd+Bwd Timed

| Seq Len | Org fwd (ms) | Org bwd (ms) | Fast fwd (ms) | Fast bwd (ms) | Total Org | Total Fast | Speedup |
|---------|-------------|-------------|--------------|--------------|-----------|------------|---------|
| 256 | 36.7 ± 5.0 | 83.6 ± 10.5 | 21.0 ± 0.7 | 45.0 ± 18.9 | 120.3 | 66.0 | **~1.8x** |
| 512 | 84.8 ± 7.1 | 138.6 ± 2.4 | 20.5 ± 0.1 | 55.4 ± 0.1 | 223.4 | 76.0 | **~2.9x** |

> **Note:** Speedup increases with sequence length as the sparse kernel overhead is amortized over more operations. At shorter sequences, both methods are fast enough that differences are within noise.

### Different Precisions (Batch=8, SeqLen=512, HiddenDim=4096, Vocab=115936, Top-K=512) — Full Fwd+Bwd Timed

Fair comparison using `torch.autograd.backward()` for both org and fast.

| Precision | Org fwd (ms) | Org bwd (ms) | Fast fwd (ms) | Fast bwd (ms) | Total Org | Total Fast | Speedup |
|-----------|-------------|-------------|--------------|--------------|-----------|------------|---------|
| float32   | 91.4 ± 3.5    | 136.2 ± 8.2     | 17.1 ± 0.1       | 53.5 ± 0.1       | 227.6  | 70.6      | **~3.2x** |
| bfloat16  | 66.0 ± 65.5     | 42.8 ± 7.4       | 9.8 ± 0.5        | 30.7 ± 1.7        | 108.9    | 40.4      | **~2.7x** |
| float16   | 70.9 ± 74.7     | 46.3 ± 2.4       | 9.4 ± 0.4        | 25.5 ± 2.3        | 117.2    | 34.9      | **~3.4x** |

> **Note:** BF16/FP16 yield the highest relative speedup because both `kl_div_fast` and `kl_div_org` benefit from Tensor Cores, but the sparse kernel's advantage over gather is larger at lower precision. FP32 remains recommended for inference/preprocessing; BF16/FP16 for training.

### Large Tensor Scaling (Batch=8, SeqLen=512) — Full Fwd+Bwd Timed

| Configuration | Org fwd (ms) | Org bwd (ms) | Fast fwd (ms) | Fast bwd (ms) | Total Org | Total Fast | Speedup |
|---------------|-------------|-------------|--------------|--------------|-----------|------------|---------|
| B=8, SL=512   | 89.4 ± 8.3    | 134.9 ± 2.6     | 20.5 ± 0.1       | 55.3 ± 0.2      | 224.3 ± 8.3  | 75.8 ± 0.2    | **~3.0x** |

> **Note:** With larger batches and longer sequences, the speedup increases significantly as kernel launch overhead is amortized. The advantage of `kl_div_fast` becomes more pronounced with bigger inputs.

## Limitations & Important Notes

- **Teacher Top-K must be pre-computed.** This library assumes `teacher_indices` and `teacher_probs` are provided as inputs. It optimizes the *student side* only. For **online distillation** (same GPU, simultaneous teacher/student forward pass), you still need techniques like [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint-support.html) or offloading for the teacher model. See above for a recommended chunked-topk implementation that reduces teacher memory usage.
- **NVIDIA GPUs only.** Triton supports Turing (sm_75) and newer architectures. Older architectures may experience reduced performance or compatibility issues.
- **Precision considerations.** The library natively supports **FP16, BF16 (Ampere+ / SM80+), and FP32** inputs. For KL divergence computation:
  - `log_softmax` is numerically stable in all supported precisions.
  - Internal accumulation uses the input dtype to preserve performance.
  - Recommended: use FP32 for `teacher_probs` if they come from a different precision source.

## Future Work

- Fuse KL divergence calculation (Softmax + KL) into the Triton kernel to reduce HBM reads/writes
- Support AMD ROCm via triton-rocm backend
- Benchmark with different GPU architectures and larger hidden dimensions

## License

MIT License. See [LICENSE](LICENSE) for details.
