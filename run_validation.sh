#!/bin/bash
# Validation + exploration runs for post-overnight-sweep analysis.
# Usage: nohup bash run_validation.sh > validation.log 2>&1 &
#
# 9 runs, ~90 minutes on idle machine.
# Phase A: Validate current best (3x)
# Phase B: Rerun original SSSS baseline (2x)
# Phase C: Push ASPECT_RATIO smaller (2 values)
# Phase D: Revisit close discards on smaller model (2 retests)

cd "$(dirname "$0")"

TRAIN_PY="train.py"
RESULTS="results.tsv"
BEST_BPB="99.0"
RUN_NUM=0

echo "=== VALIDATION + EXPLORATION ==="
echo "Started: $(date)"
echo "Machine should be idle for reliable step counts (200+)"
echo ""

# Save current train.py (this is the accumulated best from overnight)
cp "$TRAIN_PY" "$TRAIN_PY.current_best"

run_experiment() {
    local desc="$1"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_validation_${RUN_NUM}.log"

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
        echo -e "${commit}\t0.000000\t0.0\tcrash\tvalidation: $desc" >> "$RESULTS"
        return 0
    fi

    # Track best across ALL runs (not just per-phase)
    local is_better=$(echo "$bpb $BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')
    if [ "$is_better" = "yes" ]; then
        echo "  NEW GLOBAL BEST!"
        BEST_BPB="$bpb"
        cp "$TRAIN_PY" "$TRAIN_PY.new_best"
        echo -e "${commit}\t${bpb}\t0.0\tkeep\tvalidation: $desc" >> "$RESULTS"
    else
        echo "  (best so far: $BEST_BPB)"
        echo -e "${commit}\t${bpb}\t0.0\tdiscard\tvalidation: $desc" >> "$RESULTS"
    fi
}

restore_current_best() {
    cp "$TRAIN_PY.current_best" "$TRAIN_PY"
}

set_param() {
    local param="$1"
    local value="$2"
    sed -i '' "s/^${param} = .*/${param} = ${value}/" "$TRAIN_PY"
}

# ──────────────────────────────────────────────────────────
# PHASE A: Validate current best config (3 runs)
#   Current best = SLSL + WARMDOWN 0.3 + FINAL_LR_FRAC 0.05
#                + MATRIX_LR 0.03 + ASPECT_RATIO 32
#   Expecting ~1.427 bpb, ~523 steps
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE A: Validate current best (3 runs) ==="
echo ""

restore_current_best
run_experiment "current best — validation run 1/3"

restore_current_best
run_experiment "current best — validation run 2/3"

restore_current_best
run_experiment "current best — validation run 3/3"

# ──────────────────────────────────────────────────────────
# PHASE B: Rerun original SSSS baseline (2 runs)
#   Reset to pre-overnight defaults for clean measurement.
#   Overnight Run 1 only got 118 steps (unreliable).
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE B: Original SSSS baseline rerun (2 runs) ==="
echo ""

restore_current_best
set_param "ASPECT_RATIO" "64"
set_param "WINDOW_PATTERN" '"SSSS"'
set_param "MATRIX_LR" "0.04"
set_param "WARMDOWN_RATIO" "0.5"
set_param "FINAL_LR_FRAC" "0.0"
run_experiment "original SSSS baseline — rerun 1/2"

restore_current_best
set_param "ASPECT_RATIO" "64"
set_param "WINDOW_PATTERN" '"SSSS"'
set_param "MATRIX_LR" "0.04"
set_param "WARMDOWN_RATIO" "0.5"
set_param "FINAL_LR_FRAC" "0.0"
run_experiment "original SSSS baseline — rerun 2/2"

# ──────────────────────────────────────────────────────────
# PHASE C: Push ASPECT_RATIO smaller (2 runs)
#   32 was half size and won. Test 24 and 16.
#   Uses current best for all other params.
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE C: Push ASPECT_RATIO further (2 runs) ==="
echo ""

restore_current_best
set_param "ASPECT_RATIO" "24"
run_experiment "ASPECT_RATIO=24 (smaller model)"

restore_current_best
set_param "ASPECT_RATIO" "16"
run_experiment "ASPECT_RATIO=16 (quarter model size)"

# ──────────────────────────────────────────────────────────
# PHASE D: Revisit close discards on smaller model (2 runs)
#   These were tested on ASPECT_RATIO=64.
#   Optimal HPs shift with model size — retest on 32.
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE D: Revisit discards on smaller model (2 runs) ==="
echo ""

restore_current_best
sed -i '' 's/ADAM_BETAS = (0.8, 0.95)/ADAM_BETAS = (0.9, 0.99)/' "$TRAIN_PY"
run_experiment "ADAM_BETAS=(0.9,0.99) on ASPECT_RATIO=32"

restore_current_best
set_param "MATRIX_LR" "0.06"
run_experiment "MATRIX_LR=0.06 on ASPECT_RATIO=32"

# ──────────────────────────────────────────────────────────
# Restore best config
# ──────────────────────────────────────────────────────────
if [ -f "$TRAIN_PY.new_best" ]; then
    echo ""
    echo "New best config found — restoring it."
    cp "$TRAIN_PY.new_best" "$TRAIN_PY"
    rm -f "$TRAIN_PY.new_best"
else
    echo ""
    echo "No improvement found — restoring current best."
    restore_current_best
fi
rm -f "$TRAIN_PY.current_best"

echo ""
echo "=== VALIDATION COMPLETE ==="
echo "Finished: $(date)"
echo "Best val_bpb: $BEST_BPB"
echo "Results appended to results.tsv"
echo "Logs in run_validation_*.log"
