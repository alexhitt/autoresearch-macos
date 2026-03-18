#!/bin/bash
# Retest AR=96 on IDLE machine
#
# Sweep6 AR=96 crashed, but Chrome/Tailscale was open during the test.
# M2 Pro unified memory means Chrome steals from Metal's pool.
# AR=96 completed 22 training steps at 0.0% MFU before crashing at validation.
# Hypothesis: it's in the gray zone — crashes under contention, might survive idle.
#
# BEFORE RUNNING:
#   1. Quit Chrome, Safari, Slack, everything except Terminal
#   2. Verify: ps aux | grep -i chrome  (should be empty)
#   3. Run: nohup bash run_retest_ar96.sh > retest_ar96.log 2>&1 &
#   4. echo $! > retest_ar96.pid
#
# Phase 1: T=5min (survive check)
# Phase 2: T=30min (if Phase 1 survives)
# Total time: ~40min if both survive, ~10min if Phase 1 crashes

set -euo pipefail
cd "$(dirname "$0")"

TRAIN_PY="train.py"
PREPARE_PY="prepare.py"
RESULTS="results.tsv"

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

echo "=== AR=96 RETEST (idle machine) ==="
echo "Started: $(date)"
echo ""

# Check for Chrome/browser processes
if pgrep -i "chrome|safari|firefox|brave" > /dev/null 2>&1; then
    echo "WARNING: Browser process detected! Results may be unreliable."
    echo "Quit all browsers before running. Continue anyway? (10s to Ctrl-C)"
    sleep 10
fi

# Log memory state
echo "Memory state at start:"
vm_stat | head -5
echo ""

# Save originals
cp "$TRAIN_PY" "$TRAIN_PY.retest_backup"
cp "$PREPARE_PY" "$PREPARE_PY.retest_backup"

cleanup() {
    cp "$TRAIN_PY.retest_backup" "$TRAIN_PY"
    cp "$PREPARE_PY.retest_backup" "$PREPARE_PY"
    rm -f "$TRAIN_PY.retest_backup" "$PREPARE_PY.retest_backup"
    echo "Configs restored."
}
trap cleanup EXIT

# Set AR=96
sed -i '' "s/^ASPECT_RATIO = .*/ASPECT_RATIO = 96/" "$TRAIN_PY"

# ── Phase 1: T=5min ──────────────────────────────────
echo "═══════════════════════════════════"
echo "Phase 1: AR=96 @ T=5min"
echo "═══════════════════════════════════"

sed -i '' "s/^TIME_BUDGET = .*/TIME_BUDGET = 300/" "$PREPARE_PY"

run_with_timeout 900 uv run train.py > retest_ar96_5min.log 2>&1 || true

bpb=$(grep "^val_bpb:" retest_ar96_5min.log 2>/dev/null | awk '{print $2}')
steps=$(grep "^num_steps:" retest_ar96_5min.log 2>/dev/null | awk '{print $2}')
params=$(grep "^num_params_M:" retest_ar96_5min.log 2>/dev/null | awk '{print $2}')
commit=$(git rev-parse --short HEAD)

if [ -z "$bpb" ]; then
    echo "CRASH at T=5min — AR=96 genuinely can't fit on this hardware."
    echo "Crash log tail:"
    tail -10 retest_ar96_5min.log
    echo ""
    echo -e "${commit}\t0.000000\t0.0\tcrash\tretest-5min: AR=96 (384-dim, idle machine) [CRASH]" >> "$RESULTS"
    echo ""
    echo "VERDICT: AR=96 crashes even on idle machine."
    echo "Proceed with HEAD_DIM=64 sweep7 for intermediate model sizes."
    echo "Finished: $(date)"
    exit 1
fi

echo "val_bpb=$bpb, steps=$steps, params=${params}M"
echo -e "${commit}\t${bpb}\t0.0\tkeep\tretest-5min: AR=96 (384-dim, ${params}M params, idle machine)" >> "$RESULTS"
echo ""
echo "✓ AR=96 SURVIVED T=5min on idle machine!"
echo "  (Sweep6 crashed here with Chrome open)"
echo ""

# Log memory state between phases
echo "Memory state after Phase 1:"
vm_stat | head -5
echo ""

# ── Phase 2: T=30min ─────────────────────────────────
echo "═══════════════════════════════════"
echo "Phase 2: AR=96 @ T=30min"
echo "═══════════════════════════════════"

# Reset train.py (fresh start, not warm)
cp "$TRAIN_PY.retest_backup" "$TRAIN_PY"
sed -i '' "s/^ASPECT_RATIO = .*/ASPECT_RATIO = 96/" "$TRAIN_PY"
sed -i '' "s/^TIME_BUDGET = .*/TIME_BUDGET = 1800/" "$PREPARE_PY"

run_with_timeout 3300 uv run train.py > retest_ar96_30min.log 2>&1 || true

bpb=$(grep "^val_bpb:" retest_ar96_30min.log 2>/dev/null | awk '{print $2}')
steps=$(grep "^num_steps:" retest_ar96_30min.log 2>/dev/null | awk '{print $2}')
params=$(grep "^num_params_M:" retest_ar96_30min.log 2>/dev/null | awk '{print $2}')

if [ -z "$bpb" ]; then
    echo "CRASH at T=30min — AR=96 survives short runs but OOMs on longer ones."
    echo "Crash log tail:"
    tail -10 retest_ar96_30min.log
    echo ""
    echo -e "${commit}\t0.000000\t0.0\tcrash\tretest-30min: AR=96 (384-dim, idle machine) [CRASH]" >> "$RESULTS"
    echo ""
    echo "VERDICT: AR=96 survives T=5min but crashes T=30min."
    echo "Partial win — 3 data points at T=5min but only 2 at T=30min."
    echo "Finished: $(date)"
    exit 1
fi

echo "val_bpb=$bpb, steps=$steps, params=${params}M"
echo -e "${commit}\t${bpb}\t0.0\tkeep\tretest-30min: AR=96 (384-dim, ${params}M params, idle machine)" >> "$RESULTS"

echo ""
echo "════════════════════════════════════════════════════"
echo "AR=96 RETEST RESULTS"
echo "════════════════════════════════════════════════════"
echo ""
echo "✓ AR=96 survived BOTH phases on idle machine!"
echo "  Sweep6 crash was caused by Chrome/Tailscale memory contention."
echo ""
echo "You now have 3 data points for scaling law validation:"
echo "  AR=32  (128-dim,  5.0M)  — from sweep6"
echo "  AR=64  (256-dim, 11.5M)  — from sweep6"
echo "  AR=96  (384-dim, ${params}M) — from this retest"
echo ""
echo "Next: analyze T=5min and T=30min results across all 3 sizes."
echo "Finished: $(date)"
