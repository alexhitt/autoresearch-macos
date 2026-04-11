#!/bin/bash
set -euo pipefail

# Autoresearch scoring optimizer — overnight wrapper
# Runs the optimization loop, exports weights if improved.
# Designed to be called by launchd at 3 AM.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/output/autoresearch-scoring/logs"
LATEST_RUN="$HOME/output/autoresearch-scoring/latest_run.json"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

echo "=== Autoresearch Scoring Optimizer ===" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

# Refresh pipeline data from Helsinki
echo "Refreshing pipeline data from Helsinki..." | tee -a "$LOG_FILE"
mkdir -p "$HOME/.lead-command/signals"
scp -q root@100.94.225.67:~/.lead-command/signals/items.json "$HOME/.lead-command/signals/items.json" >> "$LOG_FILE" 2>&1
scp -q root@100.94.225.67:~/.lead-command/signals/claims.json "$HOME/.lead-command/signals/claims.json" >> "$LOG_FILE" 2>&1

# Rebuild dataset from fresh data
echo "Rebuilding dataset..." | tee -a "$LOG_FILE"
python3 "$SCRIPT_DIR/build_dataset.py" >> "$LOG_FILE" 2>&1

# Run optimization loop
echo "Running optimizer (max 30 iterations, early stop 5)..." | tee -a "$LOG_FILE"
BEFORE_COMPOSITE=$(python3 -c "
import json, os
results_file = os.path.join('$SCRIPT_DIR', 'optimize_results.jsonl')
best = 0.0
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('kept') and entry.get('composite', 0) > best:
                best = entry['composite']
print(f'{best:.6f}')
" 2>/dev/null || echo "0.0")

python3 "$SCRIPT_DIR/optimize_scorer.py" --max-iters 30 --early-stop 5 >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# Check if we improved
AFTER_COMPOSITE=$(python3 -c "
import json, os
results_file = os.path.join('$SCRIPT_DIR', 'optimize_results.jsonl')
best = 0.0
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('kept') and entry.get('composite', 0) > best:
                best = entry['composite']
print(f'{best:.6f}')
" 2>/dev/null || echo "0.0")

IMPROVED="false"
if python3 -c "exit(0 if float('$AFTER_COMPOSITE') > float('$BEFORE_COMPOSITE') else 1)" 2>/dev/null; then
    IMPROVED="true"
    echo "Model improved! Exporting weights..." | tee -a "$LOG_FILE"
    python3 "$SCRIPT_DIR/export_weights.py" >> "$LOG_FILE" 2>&1
fi

# Write summary
cat > "$LATEST_RUN" <<EOF
{
  "date": "$DATE",
  "exit_code": $EXIT_CODE,
  "before_composite": $BEFORE_COMPOSITE,
  "after_composite": $AFTER_COMPOSITE,
  "improved": $IMPROVED,
  "log_file": "$LOG_FILE"
}
EOF

echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Before: $BEFORE_COMPOSITE → After: $AFTER_COMPOSITE (improved: $IMPROVED)" | tee -a "$LOG_FILE"
