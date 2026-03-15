#!/bin/bash
# Sweep 2: Chase MATRIX_LR trend + retest combos on new best config.
# Usage: nohup bash run_sweep2.sh > sweep2.log 2>&1 &
#
# 8 runs, ~60 minutes on idle machine.
# Phase A: Push MATRIX_LR higher (0.09, 0.12)
# Phase B: Combo tests on MATRIX_LR=0.06 base
# Phase C: Retest warmdown and embedding LR on new config

cd "$(dirname "$0")"

TRAIN_PY="train.py"
RESULTS="results.tsv"
BEST_BPB="99.0"
RUN_NUM=0

echo "=== SWEEP 2: Chase MATRIX_LR + combos ==="
echo "Started: $(date)"
echo ""

cp "$TRAIN_PY" "$TRAIN_PY.current_best"

run_experiment() {
    local desc="$1"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_sweep2_${RUN_NUM}.log"

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
        echo -e "${commit}\t0.000000\t0.0\tcrash\tsweep2: $desc" >> "$RESULTS"
        return 0
    fi

    local is_better=$(echo "$bpb $BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')
    if [ "$is_better" = "yes" ]; then
        echo "  NEW GLOBAL BEST!"
        BEST_BPB="$bpb"
        cp "$TRAIN_PY" "$TRAIN_PY.new_best"
        echo -e "${commit}\t${bpb}\t0.0\tkeep\tsweep2: $desc" >> "$RESULTS"
    else
        echo "  (best so far: $BEST_BPB)"
        echo -e "${commit}\t${bpb}\t0.0\tdiscard\tsweep2: $desc" >> "$RESULTS"
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
# PHASE A: Push MATRIX_LR higher (2 runs)
#   0.03 → 0.06 improved by 0.012 bpb. Chase the trend.
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE A: Push MATRIX_LR higher ==="
echo ""

restore_current_best
set_param "MATRIX_LR" "0.09"
run_experiment "MATRIX_LR=0.09"

restore_current_best
set_param "MATRIX_LR" "0.12"
run_experiment "MATRIX_LR=0.12"

# ──────────────────────────────────────────────────────────
# PHASE B: Combo tests (3 runs)
#   Test interactions that weren't tested together.
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE B: Combo tests ==="
echo ""

restore_current_best
sed -i '' 's/ADAM_BETAS = (0.8, 0.95)/ADAM_BETAS = (0.9, 0.99)/' "$TRAIN_PY"
run_experiment "MATRIX_LR=0.06 + ADAM_BETAS=(0.9,0.99)"

restore_current_best
set_param "WARMDOWN_RATIO" "0.2"
run_experiment "WARMDOWN_RATIO=0.2 on new config"

restore_current_best
set_param "WARMDOWN_RATIO" "0.4"
run_experiment "WARMDOWN_RATIO=0.4 on new config"

# ──────────────────────────────────────────────────────────
# PHASE C: Retest other discards on new config (3 runs)
#   Embedding LR and weight decay — both discarded on AR=64.
# ──────────────────────────────────────────────────────────
echo ""
echo "=== PHASE C: Retest discards on new config ==="
echo ""

restore_current_best
set_param "EMBEDDING_LR" "0.8"
run_experiment "EMBEDDING_LR=0.8 on new config"

restore_current_best
set_param "MATRIX_LR" "0.045"
run_experiment "MATRIX_LR=0.045 (interpolate 0.03-0.06)"

restore_current_best
set_param "FINAL_LR_FRAC" "0.1"
run_experiment "FINAL_LR_FRAC=0.1 on new config"

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
echo "=== SWEEP 2 COMPLETE ==="
echo "Finished: $(date)"
echo "Best val_bpb: $BEST_BPB"
echo "Results appended to results.tsv"
echo "Logs in run_sweep2_*.log"
