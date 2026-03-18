#!/bin/bash
# Sweep 6: Proper Scaling Law Validation with Distinct Model Sizes
#
# Sweep 5 revealed that HEAD_DIM=128 quantization collapsed AR=16/24/32
# to identical models (all → model_dim=128). Only AR=64 was distinct.
# "Scaling law" was overclaimed from 2 data points.
#
# This sweep tests AR values that produce DISTINCT model_dim values:
#   AR=32  → model_dim=128  (1 head,  ~5.0M params)  — known works
#   AR=64  → model_dim=256  (2 heads, ~11.5M params)  — known works
#   AR=96  → model_dim=384  (3 heads, ~25M params)    — may OOM
#   AR=128 → model_dim=512  (4 heads, ~38M params)    — higher OOM risk
#
# Phase 1: T=5min for all 4 AR values (OOM detection + clean same-session data)
# Phase 2: T=30min for all AR values that survived Phase 1
#
# If 3+ distinct sizes survive → cubic scaling law becomes testable.
# If AR=128 OOMs → still have 3 points (128, 256, 384) which beats sweep5's 2.
#
# Expected time: ~3.5 hours (4×10min + 4×40min, minus any OOMs)
#
# IMPORTANT: Run on IDLE machine. Close all apps except Terminal.
#   GPU contention disproportionately affects larger models (sweep5 confirmed:
#   AR=64 was 0.071 bpb worse under contention).
#
# Usage: nohup bash run_sweep6_scaling.sh > sweep6_scaling.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")"

TRAIN_PY="train.py"
PREPARE_PY="prepare.py"
RESULTS="results.tsv"
RUN_NUM=0

# Track which AR values survive Phase 1
SURVIVED_AR=()

# macOS doesn't have GNU timeout — pure bash replacement
run_with_timeout() {
    local secs="$1"
    shift
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watchdog=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return $rc
}

echo "=== SWEEP 6: Proper Scaling Law — 4 Distinct Model Sizes ==="
echo "Started: $(date)"
echo "Testing: AR=32,64,96,128 → model_dim=128,256,384,512"
echo "Phase 1: T=5min (OOM check + clean baseline)"
echo "Phase 2: T=30min (scaling law data, survivors only)"
echo ""

# Save originals
cp "$TRAIN_PY" "$TRAIN_PY.sweep6_backup"
cp "$PREPARE_PY" "$PREPARE_PY.sweep6_backup"

set_time_budget() {
    local budget="$1"
    cp "$PREPARE_PY.sweep6_backup" "$PREPARE_PY"
    sed -i '' "s/^TIME_BUDGET = .*/TIME_BUDGET = ${budget}/" "$PREPARE_PY"
}

set_config() {
    local ar="$1"
    # Reset train.py to current best config each time
    cp "$TRAIN_PY.sweep6_backup" "$TRAIN_PY"
    # Only change AR — everything else stays at champion config
    # (SLSL, BATCH=2^15, LR=0.06, WR=0.5, FINAL_LR_FRAC=0.05)
    sed -i '' "s/^ASPECT_RATIO = .*/ASPECT_RATIO = ${ar}/" "$TRAIN_PY"
}

run_experiment() {
    local phase="$1"
    local ar="$2"
    local time_label="$3"
    local desc="$4"
    RUN_NUM=$((RUN_NUM + 1))
    local logfile="run_sweep6_${RUN_NUM}.log"

    echo "────────────────────────────────────────"
    echo "Run $RUN_NUM (Phase $phase): AR=$ar @ T=$time_label — $desc"
    echo "Time: $(date +%H:%M:%S)"

    # Run with timeout: 2x the time budget + 5 min for eval overhead
    local timeout_sec
    if [ "$time_label" = "5min" ]; then
        timeout_sec=900   # 15 min max (5 train + 10 eval)
    else
        timeout_sec=3300  # 55 min max (30 train + 25 eval)
    fi

    run_with_timeout "$timeout_sec" uv run train.py > "$logfile" 2>&1 || true

    local bpb=$(grep "^val_bpb:" "$logfile" 2>/dev/null | awk '{print $2}')
    local steps=$(grep "^num_steps:" "$logfile" 2>/dev/null | awk '{print $2}')
    local params=$(grep "^num_params_M:" "$logfile" 2>/dev/null | awk '{print $2}')
    local commit=$(git rev-parse --short HEAD)

    if [ -z "$bpb" ]; then
        echo "  CRASH/OOM — see $logfile"
        local crash_reason=$(tail -5 "$logfile" 2>/dev/null | grep -i "oom\|memory\|alloc\|killed\|error" | head -1)
        echo "  Reason: ${crash_reason:-unknown}"
        echo -e "${commit}\t0.000000\t0.0\tcrash\tsweep6-${time_label}: AR=${ar} (model_dim=$((((4 * ar + 127) / 128) * 128))) — ${desc} [CRASH]" >> "$RESULTS"
        return 1  # signal crash
    fi

    echo "  val_bpb=$bpb, steps=$steps, params=${params}M"
    echo -e "${commit}\t${bpb}\t0.0\tkeep\tsweep6-${time_label}: AR=${ar} (model_dim=$((((4 * ar + 127) / 128) * 128)), ${params}M params) — ${desc}" >> "$RESULTS"
    return 0  # signal success
}

# ══════════════════════════════════════════════════════════════
# PHASE 1: T=5min — OOM detection + clean same-session baselines
# ══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════"
echo "PHASE 1: T=5min runs (OOM detection + baselines)"
echo "═══════════════════════════════════════════════"
echo ""

set_time_budget 300

for ar in 32 64 96 128; do
    model_dim=$(( ((4 * ar + 127) / 128) * 128 ))
    heads=$((model_dim / 128))
    set_config "$ar"

    if run_experiment 1 "$ar" "5min" "${model_dim}-dim, ${heads} head(s)"; then
        SURVIVED_AR+=("$ar")
        echo "  ✓ AR=$ar survived — queued for Phase 2"
    else
        echo "  ✗ AR=$ar crashed — skipping Phase 2"
    fi
    echo ""
done

echo "────────────────────────────────────────"
echo "PHASE 1 COMPLETE"
echo "Survivors: ${SURVIVED_AR[*]:-none}"
echo "────────────────────────────────────────"
echo ""

SURVIVED_COUNT=${#SURVIVED_AR[@]}
if [ "$SURVIVED_COUNT" -lt 3 ]; then
    echo "WARNING: Only $SURVIVED_COUNT configs survived. Need 3+ for scaling law."
    echo "Consider testing intermediate AR values (e.g., AR=80 → model_dim=384 with fewer params)."
fi

if [ "$SURVIVED_COUNT" -eq 0 ]; then
    echo "No survivors — skipping Phase 2."
    cp "$TRAIN_PY.sweep6_backup" "$TRAIN_PY"
    cp "$PREPARE_PY.sweep6_backup" "$PREPARE_PY"
    rm -f "$TRAIN_PY.sweep6_backup" "$PREPARE_PY.sweep6_backup"
    echo "=== SWEEP 6 COMPLETE (all crashed) ==="
    echo "Finished: $(date)"
    exit 1
fi

# ══════════════════════════════════════════════════════════════
# PHASE 2: T=30min — scaling law data for survivors
# ══════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════"
echo "PHASE 2: T=30min runs (scaling law validation)"
echo "═══════════════════════════════════════════════"
echo ""

set_time_budget 1800

for ar in "${SURVIVED_AR[@]}"; do
    model_dim=$(( ((4 * ar + 127) / 128) * 128 ))
    heads=$((model_dim / 128))
    set_config "$ar"

    run_experiment 2 "$ar" "30min" "${model_dim}-dim, ${heads} head(s)" || true
    echo ""
done

# ══════════════════════════════════════════════════════════════
# Restore and summarize
# ══════════════════════════════════════════════════════════════

cp "$TRAIN_PY.sweep6_backup" "$TRAIN_PY"
cp "$PREPARE_PY.sweep6_backup" "$PREPARE_PY"
rm -f "$TRAIN_PY.sweep6_backup" "$PREPARE_PY.sweep6_backup"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "SWEEP 6 RESULTS SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Configs restored to original."
echo "Survivors: ${SURVIVED_AR[*]:-none}"
echo ""
echo "Scaling law validation checklist:"
echo "  [ ] How many distinct model sizes survived? (need 3+ for curve fitting)"
echo "  [ ] At T=5min: does smallest model win? (overhead-bound prediction)"
echo "  [ ] At T=30min: does largest surviving model win? (compute-scaling prediction)"
echo "  [ ] Is the ranking monotonic with model size at T=30min?"
echo "  [ ] Can you fit Session 28's cubic to 3+ points?"
echo ""
echo "If 3+ sizes and ranking is monotonic → SCALING LAW VALIDATED"
echo "If 2 sizes only → same as sweep5, need different approach"
echo ""
echo "=== SWEEP 6 COMPLETE ==="
echo "Finished: $(date)"
echo ""
echo "Results appended to results.tsv"
echo "Logs in run_sweep6_*.log"
