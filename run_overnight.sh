#!/bin/bash
# Overnight hyperparameter optimization — run on idle machine for reliable data.
# Usage: nohup bash run_overnight.sh > overnight.log 2>&1 &

set -e
cd "$(dirname "$0")"

TRAIN_PY="train.py"
RESULTS="results.tsv"
BEST_BPB="99.0"
RUN_NUM=0

echo "=== OVERNIGHT OPTIMIZATION ==="
echo "Started: $(date)"
echo "Machine should be idle for reliable step counts (200+)"
echo ""

# Save original train.py
cp "$TRAIN_PY" "$TRAIN_PY.backup"

run_experiment() {
    local desc="$1"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_overnight_${RUN_NUM}.log"

    echo "────────────────────────────────────────"
    echo "Run $RUN_NUM: $desc"
    echo "Time: $(date +%H:%M:%S)"

    uv run train.py > "$logfile" 2>&1

    local bpb=$(grep "^val_bpb:" "$logfile" | awk '{print $2}')
    local steps=$(grep "^num_steps:" "$logfile" | awk '{print $2}')
    local commit=$(git rev-parse --short HEAD)

    echo "  val_bpb=$bpb, steps=$steps"

    if [ -z "$bpb" ]; then
        echo "  CRASH"
        echo -e "${commit}\t0.000000\t0.0\tcrash\tovernight: $desc" >> "$RESULTS"
        return 1
    fi

    # Compare using awk (bash can't do float comparison)
    local is_better=$(echo "$bpb $BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')

    if [ "$is_better" = "yes" ]; then
        echo "  KEEP — new best!"
        BEST_BPB="$bpb"
        echo -e "${commit}\t${bpb}\t0.0\tkeep\tovernight: $desc" >> "$RESULTS"
        # Save this config as the new baseline
        cp "$TRAIN_PY" "$TRAIN_PY.best"
    else
        echo "  DISCARD"
        echo -e "${commit}\t${bpb}\t0.0\tdiscard\tovernight: $desc" >> "$RESULTS"
    fi
}

restore_baseline() {
    cp "$TRAIN_PY.best" "$TRAIN_PY" 2>/dev/null || cp "$TRAIN_PY.backup" "$TRAIN_PY"
}

set_param() {
    # Usage: set_param "PARAM_NAME" "new_value"
    local param="$1"
    local value="$2"
    sed -i '' "s/^${param} = .*/${param} = ${value}/" "$TRAIN_PY"
}

# ──────────────────────────────────────────────────────────
# Run 1: Baseline control (SSSS + 2^15, no changes)
# ──────────────────────────────────────────────────────────
run_experiment "SSSS baseline control (no changes)"
cp "$TRAIN_PY" "$TRAIN_PY.best"

# ──────────────────────────────────────────────────────────
# Run 2: SLSL control (for comparison)
# ──────────────────────────────────────────────────────────
sed -i '' 's/WINDOW_PATTERN = "SSSS"/WINDOW_PATTERN = "SLSL"/' "$TRAIN_PY"
run_experiment "SLSL control (compare window patterns)"
restore_baseline

# ──────────────────────────────────────────────────────────
# Runs 3-10: LR/schedule sweep (single-factor changes)
# ──────────────────────────────────────────────────────────

# Run 3: WARMDOWN_RATIO 0.7
restore_baseline
set_param "WARMDOWN_RATIO" "0.7"
run_experiment "WARMDOWN_RATIO=0.7"

# Run 4: WARMDOWN_RATIO 0.3
restore_baseline
set_param "WARMDOWN_RATIO" "0.3"
run_experiment "WARMDOWN_RATIO=0.3"

# Run 5: FINAL_LR_FRAC 0.05
restore_baseline
set_param "FINAL_LR_FRAC" "0.05"
run_experiment "FINAL_LR_FRAC=0.05"

# Run 6: MATRIX_LR 0.03
restore_baseline
set_param "MATRIX_LR" "0.03"
run_experiment "MATRIX_LR=0.03"

# Run 7: MATRIX_LR 0.06
restore_baseline
set_param "MATRIX_LR" "0.06"
run_experiment "MATRIX_LR=0.06"

# Run 8: EMBEDDING_LR 0.8
restore_baseline
set_param "EMBEDDING_LR" "0.8"
run_experiment "EMBEDDING_LR=0.8"

# Run 9: ADAM_BETAS (0.9, 0.99)
restore_baseline
sed -i '' 's/ADAM_BETAS = (0.8, 0.95)/ADAM_BETAS = (0.9, 0.99)/' "$TRAIN_PY"
run_experiment "ADAM_BETAS=(0.9,0.99)"

# Run 10: ASPECT_RATIO 32 (smaller model, more steps)
restore_baseline
set_param "ASPECT_RATIO" "32"
run_experiment "ASPECT_RATIO=32 (half model size)"

# ──────────────────────────────────────────────────────────
# Restore best config
# ──────────────────────────────────────────────────────────
restore_baseline
rm -f "$TRAIN_PY.backup" "$TRAIN_PY.best"

echo ""
echo "=== OVERNIGHT COMPLETE ==="
echo "Finished: $(date)"
echo "Best val_bpb: $BEST_BPB"
echo "Results appended to results.tsv"
echo "Logs in run_overnight_*.log"
