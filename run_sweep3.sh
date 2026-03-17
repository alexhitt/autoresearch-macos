#!/bin/bash
# Sweep 3: Find WARMDOWN_RATIO peak.
# Trend: 0.2 (1.419) → 0.3 (1.415) → 0.4 (1.414) — monotonically better.
# Test 0.5 and 0.6 to find where it peaks.
# Usage: nohup bash run_sweep3.sh > sweep3.log 2>&1 &
#
# 2 runs, ~15 minutes on idle machine.

cd "$(dirname "$0")"

TRAIN_PY="train.py"
RESULTS="results.tsv"
BEST_BPB="99.0"
RUN_NUM=0

echo "=== SWEEP 3: WARMDOWN_RATIO peak finder ==="
echo "Started: $(date)"
echo ""

cp "$TRAIN_PY" "$TRAIN_PY.current_best"

run_experiment() {
    local desc="$1"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_sweep3_${RUN_NUM}.log"

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
        echo -e "${commit}\t0.000000\t0.0\tcrash\tsweep3: $desc" >> "$RESULTS"
        return 0
    fi

    local is_better=$(echo "$bpb $BEST_BPB" | awk '{print ($1 < $2) ? "yes" : "no"}')
    if [ "$is_better" = "yes" ]; then
        echo "  NEW BEST THIS SWEEP!"
        BEST_BPB="$bpb"
        cp "$TRAIN_PY" "$TRAIN_PY.new_best"
        echo -e "${commit}\t${bpb}\t0.0\tkeep\tsweep3: $desc" >> "$RESULTS"
    else
        echo "  (best so far: $BEST_BPB)"
        echo -e "${commit}\t${bpb}\t0.0\tdiscard\tsweep3: $desc" >> "$RESULTS"
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
# Run 1: WARMDOWN_RATIO=0.5
# ──────────────────────────────────────────────────────────

restore_current_best
set_param "WARMDOWN_RATIO" "0.5"
run_experiment "WARMDOWN_RATIO=0.5"

# ──────────────────────────────────────────────────────────
# Run 2: WARMDOWN_RATIO=0.6
# ──────────────────────────────────────────────────────────

restore_current_best
set_param "WARMDOWN_RATIO" "0.6"
run_experiment "WARMDOWN_RATIO=0.6"

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
    echo "No improvement over current config (WARMDOWN_RATIO=0.4 peaked)."
    restore_current_best
fi
rm -f "$TRAIN_PY.current_best"

echo ""
echo "=== SWEEP 3 COMPLETE ==="
echo "Finished: $(date)"
echo "Best val_bpb this sweep: $BEST_BPB"
echo "Current best overall: 1.413607 (WARMDOWN_RATIO=0.4)"
echo "Results appended to results.tsv"
echo "Logs in run_sweep3_*.log"
