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

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = dtype_map[dtype_str]

    embed = torch.randn(V, D, device=device, dtype=dtype)
    hidden = torch.randn(N, D, device=device, dtype=dtype)  # 相当于 s_transT
    t_flat = torch.randn(N, D, device=device, dtype=dtype)  # 教师侧隐状态，用于生成 topk

    indices = torch.zeros(N, topk, device=device, dtype=torch.int64)
    probs = torch.zeros(N, topk, device=device, dtype=torch.float32)

    print("Preparing teacher topk indices and probs (chunked)...")
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            t_chunk = t_flat[i : i + chunk_size]
            # 使用 float32 计算 logits 避免数值问题
            t_logits_fp32 = t_chunk.to(torch.float32) @ embed.T.to(torch.float32)
            t_topk_vals, t_topk_indices = torch.topk(t_logits_fp32, k=topk, dim=-1)
            t_topk_probs = torch.softmax(t_topk_vals / temperature, dim=-1)
            indices[i : i + chunk_size] = t_topk_indices
            probs[i : i + chunk_size] = t_topk_probs
    print("Warming up...")
    for _ in range(5):
        _ = kl_div_org(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        _ = kl_div_fast(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"Running benchmark on {device}...")
    print(f"B={B}, S={S}, D={D}, V={V}, topk={topk}, temp={temperature}, runs={runs}")
    print("-" * 60)

    times_orig = []
    times_fast = []
    losses_orig = []
    losses_fast = []
    for i in range(runs):
        # Original
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_o = kl_div_org(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_orig.append((t1 - t0) * 1000)
        losses_orig.append(loss_o.item())

        # Clear cache between org and fast to prevent memory pressure from affecting results
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_f = kl_div_fast(embed, hidden, indices, probs, temperature=temperature, reduction="mean")
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_fast.append((t1 - t0) * 1000)
        losses_fast.append(loss_f.item())

        # Clear cache between iterations to prevent memory pressure from affecting next run
        if device == "cuda":
            torch.cuda.empty_cache()

    avg_orig = sum(times_orig) / len(times_orig)
    avg_fast = sum(times_fast) / len(times_fast)
    std_orig = statistics.stdev(times_orig) if len(times_orig) > 1 else 0
    std_fast = statistics.stdev(times_fast) if len(times_fast) > 1 else 0

    print(f"\n=== Benchmark dtype={dtype_str} ===")
    print("Results (ms per run):")
    print(f"kl_div_org:  {avg_orig:.2f} ± {std_orig:.2f}")
    print(f"kl_div_fast: {avg_fast:.2f} ± {std_fast:.2f}")
    print(f"Speedup:  {avg_orig / avg_fast:.2f}x")

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
        "orig_ms": avg_orig,
        "fast_ms": avg_fast,
        "speedup": avg_orig / avg_fast,
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

    # Run with all supported dtypes and print comparison table, or single dtype if specified
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
    header = f"{'Dtype':<12} {'Org(ms)':<14} {'Fast(ms)':<14} {'Speedup':<10}"
    print(header)
    print("-" * 60)
    for r in results:
        print(f"{r['dtype']:<12} {r['orig_ms']:<14.2f} {r['fast_ms']:<14.2f} {r['speedup']:<10.2f}x")


if __name__ == "__main__":
    main()
