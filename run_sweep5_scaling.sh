#!/bin/bash
# Sweep 5: Time-Optimal Scaling Law Validation (T=30min)
#
# DeepThink Session 28 derived a cubic scaling law: N³ - c₁TN - c₂T = 0
# At T=5min (300s), the overhead-bound regime forces AR=32 optimal.
# At T=30min (1800s), the cubic predicts AR=64 becomes optimal again.
#
# This sweep tests AR=16, 24, 32, 64 at T=1800s to validate the crossover.
# If AR=64 wins at 30min, the scaling law is confirmed at two timescales.
#
# Prediction from Session 28:
#   N*(1800) = 11.46M params → AR≈64
#   Expected ranking: AR=64 > AR=32 > AR=24 > AR=16
#
# WARNING: This sweep takes ~2 hours (4 runs × 30 min each).
# Run overnight or when machine is idle.
#
# Usage: nohup bash run_sweep5_scaling.sh > sweep5_scaling.log 2>&1 &

cd "$(dirname "$0")"

TRAIN_PY="train.py"
PREPARE_PY="prepare.py"
RESULTS="results.tsv"
CURRENT_BEST_BPB="1.413163"
RUN_NUM=0

echo "=== SWEEP 5: Time-Optimal Scaling Law Validation (T=30min) ==="
echo "Started: $(date)"
echo "Testing: AR=16, 24, 32, 64 at TIME_BUDGET=1800s"
echo "Prediction (Session 28): AR=64 wins at 30min"
echo "Current 5min best: $CURRENT_BEST_BPB (AR=32 + SLSL + LR=0.06 + WR=0.5)"
echo ""

# Save originals
cp "$TRAIN_PY" "$TRAIN_PY.sweep5_backup"
cp "$PREPARE_PY" "$PREPARE_PY.sweep5_backup"

# Set TIME_BUDGET to 1800s
sed -i '' "s/^TIME_BUDGET = .*/TIME_BUDGET = 1800        # 30 minutes for scaling law validation/" "$PREPARE_PY"

run_experiment() {
    local desc="$1"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_sweep5_${RUN_NUM}.log"

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
        echo -e "${commit}\t0.000000\t0.0\tcrash\tsweep5-30min: $desc" >> "$RESULTS"
        return 0
    fi

    echo -e "${commit}\t${bpb}\t0.0\tkeep\tsweep5-30min: $desc" >> "$RESULTS"
}

set_config() {
    local ar="$1"
    local window="$2"
    local batch="$3"
    local matrix_lr="$4"
    local warmdown="$5"

    # Reset train.py to current best config
    cp "$TRAIN_PY.sweep5_backup" "$TRAIN_PY"

    # Apply params
    sed -i '' "s/^ASPECT_RATIO = .*/ASPECT_RATIO = ${ar}/" "$TRAIN_PY"
    sed -i '' "s/^WINDOW_PATTERN = .*/WINDOW_PATTERN = \"${window}\"/" "$TRAIN_PY"
    sed -i '' "s/^TOTAL_BATCH_SIZE = .*/TOTAL_BATCH_SIZE = ${batch}/" "$TRAIN_PY"
    sed -i '' "s/^MATRIX_LR = .*/MATRIX_LR = ${matrix_lr}/" "$TRAIN_PY"
    sed -i '' "s/^WARMDOWN_RATIO = .*/WARMDOWN_RATIO = ${warmdown}/" "$TRAIN_PY"
}

# ══════════════════════════════════════════════════════════════
# Run 1: AR=16 (smallest model, most steps)
# At 5min this was bad (1.446). At 30min the extra steps should help
# but LM Head Amdahl floor limits gains.
# ══════════════════════════════════════════════════════════════

set_config 16 "SLSL" "2**15" "0.06" "0.5"
run_experiment "AR=16 + SLSL + LR=0.06 (smallest model, most steps)"

# ══════════════════════════════════════════════════════════════
# Run 2: AR=24 (narrow model)
# Tests the transition zone between overhead-bound and compute-bound.
# ══════════════════════════════════════════════════════════════

set_config 24 "SLSL" "2**15" "0.06" "0.5"
run_experiment "AR=24 + SLSL + LR=0.06 (narrow model)"

# ══════════════════════════════════════════════════════════════
# Run 3: AR=32 (current 5min champion)
# Control — this is the 5min optimal. At 30min it should lose to AR=64
# if the scaling law holds.
# ══════════════════════════════════════════════════════════════

set_config 32 "SLSL" "2**15" "0.06" "0.5"
run_experiment "AR=32 + SLSL + LR=0.06 (5min champion — control)"

# ══════════════════════════════════════════════════════════════
# Run 4: AR=64 (Session 28 predicted winner at 30min)
# The key test. At 5min this got 1.571 (catastrophic). At 30min the
# cubic predicts it should beat AR=32. Uses full attention (L) and
# larger batch since it has capacity for it.
# ══════════════════════════════════════════════════════════════

set_config 64 "SLSL" "2**15" "0.06" "0.5"
run_experiment "AR=64 + SLSL + LR=0.06 (Session 28 predicted winner)"

# ══════════════════════════════════════════════════════════════
# Restore original configs
# ══════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────"
echo "SWEEP 5 RESULTS SUMMARY (T=30min)"
echo "────────────────────────────────────────"
echo ""

# Restore both files
cp "$TRAIN_PY.sweep5_backup" "$TRAIN_PY"
cp "$PREPARE_PY.sweep5_backup" "$PREPARE_PY"
rm -f "$TRAIN_PY.sweep5_backup" "$PREPARE_PY.sweep5_backup"

echo "Configs restored to original (TIME_BUDGET=300s, AR=32)."
echo ""
echo "=== SWEEP 5 COMPLETE ==="
echo "Finished: $(date)"
echo ""
echo "Scaling law validation checklist:"
echo "  [ ] Did AR=64 beat AR=32? (Session 28 predicts YES)"
echo "  [ ] Is the ranking AR=64 > AR=32 > AR=24 > AR=16? (cubic prediction)"
echo "  [ ] Does AR=16 hit the Amdahl floor? (Session 27 prediction)"
echo ""
echo "If AR=64 wins: scaling law VALIDATED at two timescales → publish"
echo "If AR=32 still wins: overhead-bound regime extends further than predicted"
echo ""
echo "Results appended to results.tsv"
echo "Logs in run_sweep5_*.log"
