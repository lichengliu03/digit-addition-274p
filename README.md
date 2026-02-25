# digit-addition-274p

A **274-parameter** decoder-only transformer that achieves **99.99% exact-match accuracy** on 10-digit integer addition (10 errors in 100,000 test examples across 10 seeds).

This builds on [rezabyt/digit-addition-311p](https://github.com/rezabyt/digit-addition-311p) (311 params, 99.999%) by reducing the positional embedding rank from 3 to 2, saving 37 parameters while preserving grokking.

Submitted to [AdderBoard](https://github.com/anadim/AdderBoard).

## Architecture

Single-layer, single-head, decoder-only transformer:

| Component | Factorization | Parameters |
|-----------|--------------|------------|
| Token embedding (tied with output) | 14 × 4 | 56 |
| Position embedding (rank-2) | 33×2 + 2×4 | **74** |
| RMSNorm (×3) | 3 × 4 | 12 |
| QKV (shareA_tieKV, rank-3) | 4×3 + 3×4 + 3×4 | 36 |
| Attention output (rank-3) | 4×3 + 3×4 | 24 |
| FFN up (rank-3) | 4×3 + 3×8 | 36 |
| FFN down (rank-3) | 8×3 + 3×4 | 36 |
| **Total (unique)** | | **274** |

Key differences from 311p:
- **Position embedding rank 3 → 2**: 33×3+3×4=111 → 33×2+2×4=74 (saves 37 params)
- Everything else identical: d_model=4, d_ff=8, rank-3 factorization, shareA_tieKV, RMSNorm

## Results

Multi-seed validation (10 seeds × 10,000 examples = 100,000 total):

| Seed | Exact Match | Errors |
|------|-------------|--------|
| 41 | 99.99% | 1 |
| 100 | 100% | 0 |
| 200 | 99.97% | 3 |
| 300 | 99.98% | 2 |
| 400 | 100% | 0 |
| 500 | 100% | 0 |
| 999 | 99.97% | 3 |
| 1234 | 99.99% | 1 |
| 7777 | 100% | 0 |
| 31415 | 100% | 0 |
| **Aggregate** | **99.99%** | **10** |

## Training

Training follows the same recipe as 311p with grokking + iterative fine-tuning:

**Base training** (200K steps, seed=42):
- Curriculum learning: 3 phases (1-3 digits → 1-6 → 1-10)
- AdamW, peak lr=0.02, cosine decay, batch size 512
- Grokking at ~130K steps → best val_exact = 99.84%

**Fine-tuning**:
- FT1: lr=0.001, 100K steps → 100% val (99.99% multi-seed)
- FT2: lr=0.0003, 50K steps → 100% val (further stabilization)

Hardware: NVIDIA A100-SXM4-40GB, CUDA 12.x. Grokking is stochastic and may depend on hardware.

## Quick Start

```bash
# Predict a single addition
python -m src.eval predict --ckpt checkpoints/best_274p_ft2.pt --a 1234567890 --b 9876543210

# Run multi-seed evaluation
python evaluate_checkpoints.py checkpoints/best_274p_ft2.pt --device cuda

# Train from scratch (requires GPU, ~30 min on A100)
python -m src.train \
  --run-name 274p_repro \
  --d-model 4 --d-ff 8 --n-head 1 --n-layer 1 \
  --pos-rank 2 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --use-rmsnorm --tie-qkv shareA_tieKV \
  --train-steps 200000 --lr 0.02 --batch-size 512 \
  --device cuda --seed 42

# Fine-tune from base checkpoint
python finetune.py results/runs/274p_repro/checkpoints/best.pt \
  --lr 0.001 --steps 100000 --device cuda
```

## What Worked and What Didn't

Starting from the 311p architecture, we tried several parameter reduction strategies:

| Experiment | Params | Result | Notes |
|-----------|--------|--------|-------|
| Baseline (311p) | 311 | 99.999% | Reproduced |
| **pos_rank=2** | **274** | **99.99%** | **Success — new record** |
| pos_rank=2 + attn_out_rank=2 | 266 | 0% | Stacking broke grokking |
| Parametric sinusoidal PE | 212 | 0.74% | Did not grok after 180K steps |
| Parametric sin PE + attn_r2 | 204 | 1.52% | Did not grok |
| Parametric sin PE + attn_r2 + tok_r2 | 184 | 0% | Complete failure |

**Key insight**: Reducing one component at a time is critical. Stacking multiple reductions prevents grokking even when each individual reduction seems safe based on weight analysis.

## Leaderboard Context

| Rank | Params | Accuracy | Author |
|------|--------|----------|--------|
| **1** | **274** | **99.99%** | **this work** |
| 2 | 311 | 99.999% | [rezabyt](https://github.com/rezabyt/digit-addition-311p) |
| 3 | 456 | 100% | [yinglunz](https://github.com/yinglunz/A-456-Parameter-Transformer-Solves-10-Digit-Addition) |
| 4 | 491 | 99.97% | [rezabyt](https://github.com/rezabyt/digit-addition-491p) |

## License

MIT
