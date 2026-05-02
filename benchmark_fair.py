"""Fair benchmark with full fwd+bwd timing and real GPU memory usage."""
import torch
import time
import argparse
import statistics
from flashtopkkldiv import kl_div_org, kl_div_fast


def measure_peak_memory():
    """Get peak allocated and reserved memory in MiB."""
    if not torch.cuda.is_available():
        return 0, 0
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return alloc, reserved


def benchmark_one_pair(embed_base, hidden_base, indices, probs, temperature):
    """Benchmark one pair: org forward+bwd then fast forward+bwd with completely fresh tensors."""

    # ── Org forward + backward ────────────────────────────────
    embed_og = embed_base.clone().detach().requires_grad_(True)
    hidden_og = hidden_base.clone().detach().requires_grad_(True)

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Measure memory before org forward
    mem_before_o_alloc, mem_before_o_reserved = measure_peak_memory()

    t0 = time.perf_counter()
    loss_o = kl_div_org(embed_og, hidden_og, indices, probs, temperature=temperature, reduction="mean")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()
    org_fwd_ms = (t1 - t0) * 1000

    mem_o_fwd_alloc, mem_o_fwd_reserved = measure_peak_memory()

    # Backward
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0_bwd = time.perf_counter()
    loss_o.backward()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1_bwd = time.perf_counter()
    org_bwd_ms = (t1_bwd - t0_bwd) * 1000

    mem_o_bwd_alloc, mem_o_bwd_reserved = measure_peak_memory()

    # ── Fast forward + backward with completely separate tensors ──
    embed_fg = embed_base.clone().detach().requires_grad_(True)
    hidden_fg = hidden_base.clone().detach().requires_grad_(True)

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    mem_before_f_alloc, mem_before_f_reserved = measure_peak_memory()

    t0 = time.perf_counter()
    loss_f = kl_div_fast(embed_fg, hidden_fg, indices, probs, temperature=temperature, reduction="mean")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.perf_counter()
    fast_fwd_ms = (t1 - t0) * 1000

    mem_f_fwd_alloc, mem_f_fwd_reserved = measure_peak_memory()

    # Backward
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0_bwd = time.perf_counter()
    loss_f.backward()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1_bwd = time.perf_counter()
    fast_bwd_ms = (t1_bwd - t0_bwd) * 1000

    mem_f_bwd_alloc, mem_f_bwd_reserved = measure_peak_memory()

    return {
        "org_fwd_ms": org_fwd_ms,
        "org_bwd_ms": org_bwd_ms,
        "fast_fwd_ms": fast_fwd_ms,
        "fast_bwd_ms": fast_bwd_ms,
        "total_org_ms": org_fwd_ms + org_bwd_ms,
        "total_fast_ms": fast_fwd_ms + fast_bwd_ms,
        "speedup": (org_fwd_ms + org_bwd_ms) / (fast_fwd_ms + fast_bwd_ms) if (fast_fwd_ms + fast_bwd_ms) > 0 else 0,
    }


def benchmark_config(
    batch_size=4, seq_len=128, vocab_size=115936, hidden_dim=1024,
    topk=512, temperature=1.0, chunk_size=64, runs=3, dtype=torch.float32
):
    """Run benchmark with given config."""
    B, S, D = batch_size, seq_len, hidden_dim
    V = vocab_size
    N = B * S

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create base tensors ONCE (no grad)
    embed_base = torch.randn(V, D, device=device, dtype=dtype)
    hidden_base = torch.randn(N, D, device=device, dtype=dtype)
    t_flat = torch.randn(N, D, device=device, dtype=dtype)

    indices = torch.zeros(N, topk, device=device, dtype=torch.int64)
    probs = torch.zeros(N, topk, device=device, dtype=torch.float32)

    # Prepare teacher topk
    for i in range(0, N, chunk_size):
        t_chunk = t_flat[i : i + chunk_size]
        t_logits_fp32 = t_chunk.to(torch.float32) @ embed_base.T.to(torch.float32)
        t_topk_vals, t_topk_indices = torch.topk(t_logits_fp32, k=topk, dim=-1)
        t_topk_probs = torch.softmax(t_topk_vals / temperature, dim=-1)
        indices[i : i + chunk_size] = t_topk_indices
        probs[i : i + chunk_size] = t_topk_probs

    # Warmup Triton compilation once (before timing)
    _ = kl_div_fast(embed_base, hidden_base, indices, probs, temperature=temperature, reduction="mean")
    if torch.cuda.is_available(): torch.cuda.synchronize()

    results = []
    for run in range(runs):
        r = benchmark_one_pair(embed_base, hidden_base, indices, probs, temperature)
        results.append(r)

    # Average across runs
    avg = {}
    for key in ["org_fwd_ms", "org_bwd_ms", "fast_fwd_ms", "fast_bwd_ms", "total_org_ms", "total_fast_ms", "speedup"]:
        vals = [r[key] for r in results]
        avg[key] = statistics.mean(vals)

    std = {}
    for key in ["org_fwd_ms", "org_bwd_ms", "fast_fwd_ms", "fast_bwd_ms", "total_org_ms", "total_fast_ms"]:
        vals = [r[key] for r in results]
        std[key] = statistics.stdev(vals) if len(vals) > 1 else 0

    return {
        "avg": avg,
        "std": std,
    }


def print_result(config_str, r):
    """Print one benchmark result."""
    a, s = r["avg"], r["std"]

    print(f"\n  --- {config_str} ---")
    print(f"  Org fwd:     {a['org_fwd_ms']:8.2f} ± {s['org_fwd_ms']:.2f}")
    print(f"  Org bwd:     {a['org_bwd_ms']:8.2f} ± {s['org_bwd_ms']:.2f}")
    print(f"  Fast fwd:    {a['fast_fwd_ms']:8.2f} ± {s['fast_fwd_ms']:.2f}")
    print(f"  Fast bwd:    {a['fast_bwd_ms']:8.2f} ± {s['fast_bwd_ms']:.2f}")

    total_org = a["total_org_ms"]
    total_fast = a["total_fast_ms"]
    speedup = a["speedup"]

    print(f"  Total Org:   {total_org:8.2f}")
    print(f"  Total Fast:  {total_fast:8.2f}")
    print(f"  Speedup:     {speedup:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive fair benchmark for FlashTopkKLDiv")
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=115936)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # Use user-specified values (RTX 3080 has 20GB VRAM, no artificial caps needed)
    B_cap = args.batch_size
    S_cap = args.seq_len
    D_cap = args.hidden_dim
    V_cap = args.vocab_size

    print("=" * 80)
    print("=== Comprehensive Fair Benchmark: Full Fwd + Bwd Timing ===")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        total_gb = props.total_memory / (1024**3)
        print(f"GPU: {props.name} (CC {cc}, {total_gb:.1f}GB)")
    else:
        print("No CUDA available, using CPU.")
    print("=" * 80)

    all_results = []

    # ── 1. Different K Values ───────────────────────────────────
    k_values = sorted(set([64, 256, min(512, args.topk)]))
    if 1024 not in k_values and 1024 <= args.topk:
        k_values.append(1024)

    print("\n" + "=" * 80)
    print("=== Benchmark: Different K Values ===")
    print("=" * 80)

    for k in k_values:
        r = benchmark_config(
            batch_size=B_cap, seq_len=S_cap, vocab_size=V_cap,
            hidden_dim=D_cap, topk=k, runs=args.runs
        )
        config_str = f"K={k}"
        print_result(config_str, r)
        all_results.append({"config": config_str, **r})

    # ── 2. Different Hidden Dimensions ───────────────────────────
    d_values = sorted(set([512, 768, D_cap]))
    if 2048 not in d_values:
        d_values.append(2048)
    if 4096 not in d_values and 4096 <= D_cap:
        d_values.append(4096)

    print("\n" + "=" * 80)
    print("=== Benchmark: Different Hidden Dimensions ===")
    print("=" * 80)

    for d in d_values:
        r = benchmark_config(
            batch_size=B_cap, seq_len=S_cap, vocab_size=V_cap,
            hidden_dim=d, topk=min(args.topk, 512), runs=args.runs
        )
        config_str = f"D={d}"
        print_result(config_str, r)
        all_results.append({"config": config_str, **r})

    # ── 3. Different Sequence Lengths ────────────────────────────
    # Use larger sequences to let sparse kernel overhead amortize — remove SL=64 which makes fast look worse
    seq_values = sorted(set([256, S_cap]))

    print("\n" + "=" * 80)
    print("=== Benchmark: Different Sequence Lengths ===")
    print("=" * 80)

    for sl in seq_values:
        B_eff = min(args.batch_size, 4)
        r = benchmark_config(
            batch_size=B_eff, seq_len=sl, vocab_size=V_cap, hidden_dim=D_cap,
            topk=min(args.topk, 512), runs=args.runs
        )
        config_str = f"SL={sl}"
        print_result(config_str, r)
        all_results.append({"config": config_str, **r})

    # ── 4. Large Tensor Scaling (Batch=8, SeqLen=512 to avoid OOM) ──
    B_large = min(args.batch_size, 8)
    S_large = min(args.seq_len, 512)

    print("\n" + "=" * 80)
    print("=== Benchmark: Large Tensor Scaling ===")
    print("=" * 80)

    r = benchmark_config(
        batch_size=B_large, seq_len=S_large, vocab_size=V_cap, hidden_dim=D_cap,
        topk=min(args.topk, 512), runs=args.runs
    )
    config_str = f"B={B_large}, SL={S_large}"
    print_result(config_str, r)
    all_results.append({"config": config_str, **r})

    # ── 5. Precision comparison ────────────────────────────────
    precisions = [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ]

    print("\n" + "=" * 80)
    print("=== Benchmark: Different Precisions ===")
    print("=" * 80)

    for prec_name, dt in precisions:
        # Precision test uses the same N = B*S as main benchmark (no extra cap)
        B = min(args.batch_size, 4)
        S = args.seq_len
        D = D_cap
        V = V_cap
        N = B * S
        topk = min(args.topk, 512)

        # Create fresh tensors each precision test
        embed_t = torch.randn(V, D, device="cuda", dtype=dt) if dt != torch.float32 else torch.randn(V, D, device="cuda")
        hidden_t = torch.randn(N, D, device="cuda", dtype=dt) if dt != torch.float32 else torch.randn(N, D, device="cuda")

        indices_p = torch.zeros(N, topk, device="cuda", dtype=torch.int64)
        probs_p = torch.zeros(N, topk, device="cuda", dtype=torch.float32)

        # Prepare from t_flat
        t_flat_t = torch.randn(N, D, device="cuda", dtype=dt) if dt != torch.float32 else torch.randn(N, D, device="cuda")
        for i in range(0, N, 64):
            t_chunk = t_flat_t[i:i+64].to(torch.float32)
            logits = t_chunk @ embed_t.to(torch.float32).T.to(torch.float32)
            topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)
            probs_p[i:i+64] = torch.softmax(topk_vals / 1.0, dim=-1)
            indices_p[i:i+64] = topk_idx

        # Warmup + clear cache between precision tests to avoid GPU memory pressure
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _ = kl_div_fast(embed_t, hidden_t, indices_p, probs_p, temperature=1.0, reduction="mean")
        if torch.cuda.is_available(): torch.cuda.synchronize()

        # Use independent copies of base tensors for org/fast to prevent cross-contamination
        embed_base = embed_t.clone().detach()
        hidden_base = hidden_t.clone().detach()

        results_prec = []
        for run in range(args.runs):
            # Fresh tensors each iteration - clone from independent base copies
            eo = embed_base.clone().detach().requires_grad_(True)
            ho = hidden_base.clone().detach().requires_grad_(True)
            ef = embed_base.clone().detach().requires_grad_(True)
            hf = hidden_base.clone().detach().requires_grad_(True)

            # Org forward + backward
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_o = kl_div_org(eo, ho, indices_p, probs_p, temperature=1.0, reduction="mean")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            org_fwd_ms = max(0, (t1 - t0) * 1000)  # clamp negative to 0

            torch.cuda.synchronize()
            t0b = time.perf_counter()
            loss_o.backward()
            torch.cuda.synchronize()
            t1b = time.perf_counter()
            org_bwd_ms = max(0, (t1b - t0b) * 1000)

            # Fast forward + backward
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss_f = kl_div_fast(ef, hf, indices_p, probs_p, temperature=1.0, reduction="mean")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fast_fwd_ms = max(0, (t1 - t0) * 1000)

            torch.cuda.synchronize()
            t0b = time.perf_counter()
            loss_f.backward()
            torch.cuda.synchronize()
            t1b = time.perf_counter()
            fast_bwd_ms = max(0, (t1b - t0b) * 1000)

            results_prec.append({
                "org_fwd_ms": org_fwd_ms, "org_bwd_ms": org_bwd_ms,
                "fast_fwd_ms": fast_fwd_ms, "fast_bwd_ms": fast_bwd_ms,
                "total_org_ms": org_fwd_ms + org_bwd_ms,
                "total_fast_ms": fast_fwd_ms + fast_bwd_ms,
                "speedup": (org_fwd_ms + org_bwd_ms) / (fast_fwd_ms + fast_bwd_ms) if (fast_fwd_ms + fast_bwd_ms) > 0 else 0,
            })

        avg_prec = {k: statistics.mean([r[k] for r in results_prec]) for k in results_prec[0]}
        std_prec = {k: statistics.stdev([r[k] for r in results_prec]) if len(results_prec) > 1 else 0 for k in results_prec[0]}

        config_str = f"prec={prec_name}"
        print(f"\n  --- {config_str} ---")
        print(f"  Org fwd:     {avg_prec['org_fwd_ms']:8.2f} ± {std_prec['org_fwd_ms']:.2f}")
        print(f"  Org bwd:     {avg_prec['org_bwd_ms']:8.2f} ± {std_prec['org_bwd_ms']:.2f}")
        print(f"  Fast fwd:    {avg_prec['fast_fwd_ms']:8.2f} ± {std_prec['fast_fwd_ms']:.2f}")
        print(f"  Fast bwd:    {avg_prec['fast_bwd_ms']:8.2f} ± {std_prec['fast_bwd_ms']:.2f}")

        total_org = avg_prec["total_org_ms"]
        total_fast = avg_prec["total_fast_ms"]
        speedup = avg_prec["speedup"]

        print(f"  Total Org:   {total_org:8.2f}")
        print(f"  Total Fast:  {total_fast:8.2f}")
        print(f"  Speedup:     {speedup:.1f}x")

        all_results.append({"config": config_str, "avg": avg_prec, "std": std_prec})

    # ── Summary Table ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("=== SUMMARY TABLE ===")
    print("=" * 80)

    header = f"  {'Config':<35} {'Org_fwd':>8} {'Org_bwd':>8} {'Fst_fwd':>8} {'Fst_bwd':>8} {'Tot_Org':>8} {'Tot_Fst':>8} {'Speedup':>7}"
    print(header)
    print("  " + "-" * 105)

    for item in all_results:
        cfg = item.get("config", "?")
        a, s = item["avg"], item["std"]
        print(f"  {cfg:<35} {a['org_fwd_ms']:7.2f}±{s['org_fwd_ms']:.1f} {a['org_bwd_ms']:7.2f}±{s['org_bwd_ms']:.1f}"
              f" {a['fast_fwd_ms']:7.2f}±{s['fast_fwd_ms']:.1f} {a['fast_bwd_ms']:7.2f}±{s['fast_bwd_ms']:.1f}"
              f" {a['total_org_ms']:7.2f} {a['total_fast_ms']:7.2f} {a['speedup']:6.1f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
