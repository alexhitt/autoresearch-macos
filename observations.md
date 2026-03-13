# Autoresearch Phase 1 Observations — Mar 14, 2026

## Summary

13 runs total: 3 baselines, 10 experimental (7 hyperparameter changes + 3 OOM tests).
Best result: val_bpb **1.797** (WINDOW_PATTERN=SLSL), down from baseline best **2.067**.
Only 1 of 10 experiments improved over baseline.

## M2 Pro Memory Ceiling

| Config | Params | Result |
|--------|--------|--------|
| DEPTH=4, BATCH=16 | 11.5M | Works (~600MB MPS) |
| DEPTH=6 | 26.3M | OOM at step 10 (15.55GB + 1.76GB) |
| DEPTH=8 | ~50M | OOM at step 0 (16.32GB + 1.00GB) |
| DEPTH=12 | ~100M | OOM at step 0 (15.67GB + 1.50GB) |
| BATCH=32 | 11.5M | OOM at step 0 (14.09GB + 2.00GB alloc) |

**Conclusion:** DEPTH=4 with DEVICE_BATCH_SIZE=16 is the practical ceiling for M2 Pro 16GB.
The MPS max allowed memory is 18.13GB (unified with system). Model activations at DEPTH>=6
exhaust this. Larger batch sizes also hit the ceiling due to larger activation tensors.

## Step Count Variance (Critical Finding)

Identical runs with identical code produce different step counts:
- Run 1: 15 steps → val_bpb 2.128
- Run 2: 19 steps → val_bpb 2.081
- Run 3: 20 steps → val_bpb 2.067

Root cause: MPS throughput varies with thermal state, shader caching, and system load.
Each step takes 15-30 seconds, so ±5 steps is ±75-150 seconds — significant on a 300s budget.
Loss curves are deterministic (same loss at same step number), so val_bpb differences
are entirely driven by step count.

**Implication for Phase 2 (GEPA):** The evaluator MUST normalize by step count.
Raw val_bpb comparison between runs is confounded by MPS throughput jitter.

## MPS-Specific Issues

1. **No VRAM tracking:** `peak_vram_mb` is always 0.0. MPS uses unified memory —
   PyTorch can't report GPU VRAM separately. Memory tracking requires system-level
   tools (Activity Monitor, `vm_stat`).

2. **No torch.compile:** MPS backend doesn't support compilation. All computation
   runs eagerly, adding overhead per operation.

3. **No autocast (mixed precision):** MPS doesn't support autocast. Everything
   runs in fp32, using ~2x the memory of fp16.

4. **Throughput much lower than expected:** Got 15-65 steps in 5 min, not 300-500.
   MPS kernel launch overhead dominates at small model scale.

## Eval Phase Timing

The 21M token validation eval takes ~10-15 minutes on MPS at DEPTH=4.
Total wall time per experiment is ~15-20 minutes, not 5 minutes.
At DEPTH=6, eval would take even longer (but runs crash before getting there).

## What Worked

**WINDOW_PATTERN "SLSL"** — the only improvement. Mechanism: alternating short (half-context)
and long (full-context) attention windows reduces per-step compute. Short windows attend
to only 1024 tokens instead of 2048. Result: 3.25x more steps (65 vs ~20) in the same
time budget, and the model still learns effectively despite reduced attention span.

Note: the code forces the last layer to always use full attention (line 228), so even
"SSSS" becomes "SSSL". SLSL already has the last layer as L.

## What Didn't Work

| Change | Why it failed |
|--------|---------------|
| MATRIX_LR 2x | Learning instability at higher LR, convergence worse |
| WARMUP_RATIO 0.1 | Wastes 10% of training time at low LR — no benefit at 300s budget |
| TOTAL_BATCH_SIZE halved | More steps but each step processes fewer tokens — net loss |
| WARMDOWN_RATIO 0.3 | Less cosine decay didn't help convergence |
| WINDOW_PATTERN SSSS | All-short loses too much long-range attention, worse than SLSL |
| WEIGHT_DECAY 0.0 | No regularization slightly worse; WD 0.2 is near-optimal for this model size |
| EMBEDDING_LR 2x | Embedding LR 0.6 is already high; 1.2 overshoots |
| ASPECT_RATIO 96 | Wider model (19.7M params) too slow — only 16 steps in budget |

## Key Insight

On MPS with a fixed time budget, **throughput is everything**. The single biggest
improvement came from reducing per-step compute (sliding windows), not from
architectural changes or learning rate tuning. All experiments that increased
compute per step (wider model, deeper model, larger batch) performed worse
because they got fewer steps.

## Phase 2 Recommendations

1. GEPA evaluator must normalize val_bpb by step count
2. Focus GEPA search on compute-reducing changes (window patterns, sparse attention, etc.)
3. Consider running multiple eval runs per config to average out throughput variance
4. The 5-min budget is actually ~20 min wall time due to eval — plan accordingly
