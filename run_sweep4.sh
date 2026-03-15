#!/bin/bash
# Sweep 4: DeepThink Session 29 — Test 3 "interaction island" candidates
# that greedy search structurally could NOT reach.
#
# Each candidate is a full config reset (not incremental from current best).
# After all 3 runs, the best config overall is restored.
#
# Candidate 1: "Narrow & Aggressive" — AR=24 + SLSL + BATCH 2^15 + LR=0.09 + WR=0.4
#   Predicted: 1.390-1.405. Tests AR×LR interaction (AR=24 was discarded on wrong LR).
#
# Candidate 2: "Slow & Deep Capacity" — AR=64 + L + BATCH 2^16 + LR=0.06 + WR=0.6
#   Predicted: 1.395-1.408. Tests throughput-basin escape. Also cross-validates
#   Session 28's scaling law (should LOSE at 5min, confirming AR=64 needs longer budgets).
#
# Candidate 3: "High-Frequency Extremist" — AR=16 + SSSS + BATCH 2^16 + LR=0.12 + EMBED=1.2
#   Predicted: 1.400-1.410. Tests brute-force step count with extreme LRs.
#   Session 27 predicts this won't work (attention only 16% of FLOPs at small AR).
#
# Usage: nohup bash run_sweep4.sh > sweep4.log 2>&1 &
#
# 3 runs, ~60 minutes on idle machine.

cd "$(dirname "$0")"

TRAIN_PY="train.py"
RESULTS="results.tsv"
CURRENT_BEST_BPB="1.413163"
SWEEP_BEST_BPB="99.0"
RUN_NUM=0

echo "=== SWEEP 4: DeepThink interaction island candidates ==="
echo "Started: $(date)"
echo "Current best: $CURRENT_BEST_BPB (AR=32 + SLSL + LR=0.06 + WR=0.5)"
echo ""

# Save current best config
cp "$TRAIN_PY" "$TRAIN_PY.sweep4_backup"

run_experiment() {
    local desc="$1"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_sweep4_${RUN_NUM}.log"

    echo "────────────────────────────────────────"
    echo "Run $RUN_NUM: $desc"
    echo "Time: $(date +%H:%M:%S)"

    uv run train.py > "$logfile" 2>&1 || true

    local bpb=$(grep "^val_bpb:" "$logfile" | awk '{print $2}')
    local steps=$(grep "^num_steps:" "$logfile" | awk '{print $2}')
    local commit=$(git rev-parse --short HEAD)

    echo "  val_bpb=$bpb, steps=$steps"

    if [ -z "$bpb" ]; then
        echo "  CRASH — see $logfile"
        echo -e "${commit}\t0.000000\t0.0\tcrash\tsweep4: $desc" >> "$RESULTS"
        return 0
    fi

    # Check against current global best
    local beats_global=$(echo "$bpb $CURRENT_BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')
    if [ "$beats_global" = "yes" ]; then
        echo "  NEW GLOBAL BEST! (beats $CURRENT_BEST_BPB)"
    else
        echo "  (current global best: $CURRENT_BEST_BPB)"
    fi

    # Track sweep best
    local beats_sweep=$(echo "$bpb $SWEEP_BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')
    if [ "$beats_sweep" = "yes" ]; then
        SWEEP_BEST_BPB="$bpb"
        cp "$TRAIN_PY" "$TRAIN_PY.sweep4_new_best"
    fi

    echo -e "${commit}\t${bpb}\t0.0\tkeep\tsweep4: $desc" >> "$RESULTS"
}

set_full_config() {
    # Reset to baseline defaults first, then apply candidate config
    # This ensures each candidate starts from a clean state
    local ar="$1"
    local window="$2"
    local batch="$3"
    local matrix_lr="$4"
    local warmdown="$5"
    local embed_lr="${6:-0.6}"
    local final_lr="${7:-0.05}"

    # Start from backup (current best)
    cp "$TRAIN_PY.sweep4_backup" "$TRAIN_PY"

    # Apply all params
    sed -i '' "s/^ASPECT_RATIO = .*/ASPECT_RATIO = ${ar}/" "$TRAIN_PY"
    sed -i '' "s/^WINDOW_PATTERN = .*/WINDOW_PATTERN = \"${window}\"/" "$TRAIN_PY"
    sed -i '' "s/^TOTAL_BATCH_SIZE = .*/TOTAL_BATCH_SIZE = ${batch}/" "$TRAIN_PY"
    sed -i '' "s/^MATRIX_LR = .*/MATRIX_LR = ${matrix_lr}/" "$TRAIN_PY"
    sed -i '' "s/^WARMDOWN_RATIO = .*/WARMDOWN_RATIO = ${warmdown}/" "$TRAIN_PY"
    sed -i '' "s/^EMBEDDING_LR = .*/EMBEDDING_LR = ${embed_lr}/" "$TRAIN_PY"
    sed -i '' "s/^FINAL_LR_FRAC = .*/FINAL_LR_FRAC = ${final_lr}/" "$TRAIN_PY"
}

# ══════════════════════════════════════════════════════════════
# Candidate 1: "Narrow & Aggressive"
# AR=24 + SLSL + BATCH 2^15 + MATRIX_LR=0.09 + WR=0.4
# Predicted: 1.390-1.405
# Rationale: AR=24 was discarded on low LR. Narrower models need higher LRs.
# ══════════════════════════════════════════════════════════════

set_full_config 24 "SLSL" "2**15" "0.09" "0.4" "0.6" "0.05"
run_experiment "Candidate 1: Narrow+Aggressive (AR=24, SLSL, LR=0.09, WR=0.4)"

# ══════════════════════════════════════════════════════════════
# Candidate 2: "Slow & Deep Capacity"
# AR=64 + L + BATCH 2^16 + MATRIX_LR=0.06 + WR=0.6
# Predicted: 1.395-1.408
# Rationale: Full attention AR=64 maximizes per-step learning.
# Should LOSE at 5min (validates Session 28 scaling law).
# ══════════════════════════════════════════════════════════════

set_full_config 64 "L" "2**16" "0.06" "0.6" "0.6" "0.05"
run_experiment "Candidate 2: Slow+Deep (AR=64, L, BATCH=2^16, LR=0.06, WR=0.6)"

# ══════════════════════════════════════════════════════════════
# Candidate 3: "High-Frequency Extremist"
# AR=16 + SSSS + BATCH 2^16 + MATRIX_LR=0.12 + EMBEDDING_LR=1.2
# Predicted: 1.400-1.410
# Rationale: Brute-force steps with extreme LRs.
# Session 27 predicts this won't gain much (attention only 16% of FLOPs).
# ══════════════════════════════════════════════════════════════

set_full_config 16 "SSSS" "2**16" "0.12" "0.4" "1.2" "0.05"
run_experiment "Candidate 3: High-Freq Extremist (AR=16, SSSS, BATCH=2^16, LR=0.12, EMBED=1.2)"

# ══════════════════════════════════════════════════════════════
# Restore best config
# ══════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────"
echo "SWEEP 4 RESULTS SUMMARY"
echo "────────────────────────────────────────"
echo ""

if [ -f "$TRAIN_PY.sweep4_new_best" ]; then
    local_beats=$(echo "$SWEEP_BEST_BPB $CURRENT_BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')
    if [ "$local_beats" = "yes" ]; then
        echo "NEW GLOBAL BEST FOUND: $SWEEP_BEST_BPB"
        echo "Restoring new best config."
        cp "$TRAIN_PY.sweep4_new_best" "$TRAIN_PY"
    else
        echo "Sweep best ($SWEEP_BEST_BPB) did not beat global best ($CURRENT_BEST_BPB)."
        echo "Restoring original best config."
        cp "$TRAIN_PY.sweep4_backup" "$TRAIN_PY"
    fi
    rm -f "$TRAIN_PY.sweep4_new_best"
else
    echo "No candidates produced valid results. Restoring original config."
    cp "$TRAIN_PY.sweep4_backup" "$TRAIN_PY"
fi

rm -f "$TRAIN_PY.sweep4_backup"

echo ""
echo "=== SWEEP 4 COMPLETE ==="
echo "Finished: $(date)"
echo "Sweep best val_bpb: $SWEEP_BEST_BPB"
echo "Current global best: $CURRENT_BEST_BPB"
echo ""
echo "Cross-validation checklist:"
echo "  [ ] Candidate 1 vs current best — did AR=24+LR=0.09 beat AR=32+LR=0.06?"
echo "  [ ] Candidate 2 vs Session 28 — did AR=64 LOSE at 5min? (validates scaling law)"
echo "  [ ] Candidate 3 vs Session 27 — did SSSS fail to gain throughput? (validates FLOP analysis)"
echo ""
echo "Results appended to results.tsv"
echo "Logs in run_sweep4_*.log"
