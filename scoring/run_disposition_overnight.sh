#!/bin/bash
set -euo pipefail

# Autoresearch disposition prompt optimizer — overnight wrapper
# Runs the optimization loop for the disposition prompt.
# Designed to be called by launchd at 3:30 AM (after scoring optimizer).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$HOME/output/autoresearch-disposition/logs"
LATEST_RUN="$HOME/output/autoresearch-disposition/latest_run.json"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

echo "=== Autoresearch Disposition Prompt Optimizer ===" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

# Refresh test set from Helsinki
echo "Refreshing test set from Helsinki..." | tee -a "$LOG_FILE"
mkdir -p "$SCRIPT_DIR/disposition_data"
scp -q root@100.94.225.67:~/.lead-command/signals/claims.json "$SCRIPT_DIR/disposition_data/claims.json" >> "$LOG_FILE" 2>&1
scp -q root@100.94.225.67:~/.lead-command/signals/items.json "$SCRIPT_DIR/disposition_data/items.json" >> "$LOG_FILE" 2>&1
python3 "$SCRIPT_DIR/build_disposition_testset.py" >> "$LOG_FILE" 2>&1

# Record before-F1
BEFORE_F1=$(python3 -c "
import json, os
results_file = os.path.join('$SCRIPT_DIR', 'optimize_disposition_results.jsonl')
best = 0.0
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('kept') and entry.get('apply_now_f1', 0) > best:
                best = entry['apply_now_f1']
print(f'{best:.4f}')
" 2>/dev/null || echo "0.0")

# Run optimization loop
echo "Running optimizer (max 15 iterations, early stop 4)..." | tee -a "$LOG_FILE"
python3 "$SCRIPT_DIR/optimize_disposition.py" --max-iters 15 --early-stop 4 >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

# Check if we improved
AFTER_F1=$(python3 -c "
import json, os
results_file = os.path.join('$SCRIPT_DIR', 'optimize_disposition_results.jsonl')
best = 0.0
if os.path.exists(results_file):
    with open(results_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('kept') and entry.get('apply_now_f1', 0) > best:
                best = entry['apply_now_f1']
print(f'{best:.4f}')
" 2>/dev/null || echo "0.0")

IMPROVED="false"
if python3 -c "exit(0 if float('$AFTER_F1') > float('$BEFORE_F1') else 1)" 2>/dev/null; then
    IMPROVED="true"
    echo "Prompt improved! Winner exported." | tee -a "$LOG_FILE"
fi

# Write summary
cat > "$LATEST_RUN" <<EOF
{
  "date": "$DATE",
  "exit_code": $EXIT_CODE,
  "before_f1": $BEFORE_F1,
  "after_f1": $AFTER_F1,
  "improved": $IMPROVED,
  "log_file": "$LOG_FILE"
}
EOF

echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Before: $BEFORE_F1 → After: $AFTER_F1 (improved: $IMPROVED)" | tee -a "$LOG_FILE"
