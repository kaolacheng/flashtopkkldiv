import torch
import time
import argparse
import statistics
from flashtopkkldiv import kl_div_org, kl_div_fast


def run_bench(
    device: str = "cuda",
    batch_size: int = 4,
    seq_len: int = 128,
    vocab_size: int = 115936,
    hidden_dim: int = 1024,
    topk: int = 512,
    temperature: float = 1.0,
    chunk_size: int = 64,
    runs: int = 3,
    dtype_str: str = "float32",
):
    if device == "cuda" and not torch.cuda.is_available():
        print("No CUDA available; using CPU (will be slow).")
        device = "cpu"

    B, S, D = batch_size, seq_len, hidden_dim
    V = vocab_size
    N = B * S
    K = topk

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = dtype_map[dtype_str]

    embed = torch.randn(V, D, device=device, dtype=dtype)
    hidden = torch.randn(N, D, device=device, dtype=dtype)
    t_flat = torch.randn(N, D, device=device, dtype=dtype)

    indices = torch.zeros(N, topk, device=device, dtype=torch.int64)
    probs = torch.zeros(N, topk, device=device, dtype=torch.float32)

    print("Preparing teacher topk indices and probs (chunked)...")
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            t_chunk = t_flat[i : i + chunk_size]
            t_logits_fp32 = t_chunk.to(torch.float32) @ embed.T.to(torch.float32)
            t_topk_vals, t_topk_indices = torch.topk(t_logits_fp32, k=topk, dim=-1)
            t_topk_probs = torch.softmax(t_topk_vals / temperature, dim=-1)
            indices[i : i + chunk_size] = t_topk_indices
            probs[i : i + chunk_size] = t_topk_probs

    print("Warming up...")
    for _ in range(10):
        _ = kl_div_org(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        # Warmup fast once to trigger Triton kernel compilation
        _ = kl_div_fast(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"Running benchmark on {device}...")
    print(f"B={B}, S={S}, D={D}, V={V}, topk={topk}, temp={temperature}, runs={runs}")
    print("-" * 60)

    # ── Org: full forward + backward with .backward() ───────────────
    times_org_fwd = []
    times_org_bwd = []
    losses_orig = []

    for i in range(runs):
        # Fresh tensors each iteration so gradients don't accumulate on stale values
        embed_o = embed.clone().detach().requires_grad_(True)
        hidden_o = hidden.clone().detach().requires_grad_(True)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_o = kl_div_org(embed_o, hidden_o, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_org_fwd.append((t1 - t0) * 1000)
        losses_orig.append(loss_o.item())

        # Backward: full autograd backward pass
        if device == "cuda": torch.cuda.synchronize()
        t0_bwd = time.perf_counter()
        loss_o.backward()
        if device == "cuda": torch.cuda.synchronize()
        t1_bwd = time.perf_counter()
        times_org_bwd.append((t1_bwd - t0_bwd) * 1000)

    # ── Fast: full forward + backward with .backward() ──────────────
    times_fast_fwd = []
    times_fast_bwd = []
    losses_fast = []

    for i in range(runs):
        embed_f = embed.clone().detach().requires_grad_(True)
        hidden_f = hidden.clone().detach().requires_grad_(True)

        # Warmup call to avoid Triton JIT compilation overhead in first timing run
        _ = kl_div_fast(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda": torch.cuda.synchronize()
        hidden_f = hidden.clone().detach().requires_grad_(True)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_f = kl_div_fast(embed_f, hidden_f, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_fast_fwd.append((t1 - t0) * 1000)
        losses_fast.append(loss_f.item())

        # Backward: full autograd backward pass (measures complete backward)
        if device == "cuda": torch.cuda.synchronize()
        t0_bwd = time.perf_counter()
        loss_f.backward()
        if device == "cuda": torch.cuda.synchronize()
        t1_bwd = time.perf_counter()
        times_fast_bwd.append((t1_bwd - t0_bwd) * 1000)

        if device == "cuda":
            torch.cuda.empty_cache()

    avg_org_fwd = sum(times_org_fwd) / len(times_org_fwd)
    std_org_fwd = statistics.stdev(times_org_fwd) if len(times_org_fwd) > 1 else 0
    avg_fast_fwd = sum(times_fast_fwd) / len(times_fast_fwd)
    std_fast_fwd = statistics.stdev(times_fast_fwd) if len(times_fast_fwd) > 1 else 0

    avg_org_bwd = sum(times_org_bwd) / len(times_org_bwd) if times_org_bwd else 0
    avg_fast_bwd = sum(times_fast_bwd) / len(times_fast_bwd) if times_fast_bwd else 0

    print(f"\n=== Benchmark dtype={dtype_str} ===")
    print("Results (ms per run):")
    print(f"kl_div_org fwd:     {avg_org_fwd:.2f} ± {std_org_fwd:.2f}")
    print(f"kl_div_org bwd:     {avg_org_bwd:.2f}")
    print(f"kl_div_fast fwd:    {avg_fast_fwd:.2f} ± {std_fast_fwd:.2f}")
    print(f"kl_div_fast bwd:    {avg_fast_bwd:.2f}")

    total_org = avg_org_fwd + avg_org_bwd
    total_fast = avg_fast_fwd + avg_fast_bwd
    speedup_fwd = avg_org_fwd / avg_fast_fwd if avg_fast_fwd > 0 else 0
    speedup_total = total_org / total_fast if total_fast > 0 else 0
    print(f"\nTotal (fwd+bwd):   Org={total_org:.2f}  Fast={total_fast:.2f}")
    print(f"Speedup:           {speedup_fwd:.1f}x forward-only, {speedup_total:.1f}x total")

    print("\nLosses (mean):")
    print(f"kl_div_org:  {sum(losses_orig)/len(losses_orig):.4f}")
    print(f"kl_div_fast: {sum(losses_fast)/len(losses_fast):.4f}")

    input_mem = (embed.numel() + hidden.numel() + indices.numel() + probs.numel()) * embed.element_size() / (1024*1024)
    extra_gather_mem = N * topk * D * 4 / (1024*1024)

    mem_fast = input_mem
    mem_original = input_mem + extra_gather_mem

    print("\nEstimated peak memory footprint (MiB):")
    print(f"kl_div_org:  ~{mem_original:.1f}")
    print(f"kl_div_fast: ~{mem_fast:.1f}")
    print(f"Saved:    ~{extra_gather_mem:.1f} MiB")

    # Return results for comparison table
    return {
        "dtype": dtype_str,
        "orig_fwd_ms": avg_org_fwd,
        "org_bwd_ms": avg_org_bwd,
        "fast_fwd_ms": avg_fast_fwd,
        "fast_bwd_ms": avg_fast_bwd,
        "speedup": speedup_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashTopkKLDiv - Student Side Optimization")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--vocab_size", type=int, default=115936, help="Vocabulary size (default: 115936)")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--topk", type=int, default=512, help="Top-k value (default: 512)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax (default: 1.0)")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for teacher topk (default: 64)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None,
                        help="Data type for tensor inputs. If specified, only this dtype is tested; otherwise all three are benchmarked.")

    args = parser.parse_args()

    if args.dtype:
        dtypes_to_run = [args.dtype]
    else:
        dtypes_to_run = ["float32", "bfloat16", "float16"]
    results = []
    for d in dtypes_to_run:
        r = run_bench(
            device=args.device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            hidden_dim=args.hidden_dim,
            topk=args.topk,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            runs=args.runs,
            dtype_str=d,
        )
        results.append(r)

    # Print comparison table
    print("\n" + "=" * 60)
    print("=== Dtype Comparison Summary ===")
    print("-" * 60)
    header = f"{'Dtype':<12} {'Org_fwd':<14} {'Org_bwd':<14} {'Fst_fwd':<14} {'Fst_bwd':<14} {'Tot_spd':<10}"
    print(header)
    print("-" * 60)
    for r in results:
        total_o = r['orig_fwd_ms'] + r['org_bwd_ms']
        total_f = r['fast_fwd_ms'] + r['fast_bwd_ms']
        print(f"{r['dtype']:<12} {r['orig_fwd_ms']:<14.2f} {r['org_bwd_ms']:<14.2f} {r['fast_fwd_ms']:<14.2f} {r['fast_bwd_ms']:<14.2f} {total_o/total_f if total_f > 0 else 0:<10.1f}x")


if __name__ == "__main__":
    main()
