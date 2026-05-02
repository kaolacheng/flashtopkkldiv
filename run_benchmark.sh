#!/bin/bash
# Comprehensive fair benchmark for FlashTopkKLDiv
# Designed for RTX 3080 (20GB VRAM) - removes artificial caps from WSL version
set -e

cd "$(dirname "$0")"

echo "=============================================================="
echo " Running Fair Benchmark: Full Fwd + Bwd Timing"
echo " Target GPU: RTX 3080 (20GB)"
echo "=============================================================="

# Run with full configs - no artificial caps
python3 benchmark_fair.py \
    --batch_size 8 \
    --seq_len 512 \
    --hidden_dim 4096 \
    --topk 512 \
    --vocab-size 115936 \
    --runs 5

echo ""
echo "Benchmark complete."
echo "=============================================================="
