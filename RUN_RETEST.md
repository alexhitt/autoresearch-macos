# AR=96 Retest — Idle Machine

## Why
Sweep6 AR=96 crashed, but Chrome/Tailscale was stealing unified memory.
This retest determines if AR=96 fits on an idle M2 Pro 16GB.
If it survives → 3 data points → scaling law fittable. No sweep7 needed.

## Before You Run

```bash
# Quit EVERYTHING except Terminal (Chrome, Safari, Slack, Obsidian, etc.)
# Then verify:
pgrep -i "chrome|safari|slack|obsidian|firefox|brave"
# Should return nothing. If it does, kill those processes.
```

## Run

```bash
cd ~/projects/autoresearch-macos
nohup bash run_retest_ar96.sh > retest_ar96.log 2>&1 & echo $! > retest_ar96.pid
```

Takes ~40 min if both phases pass, ~10 min if Phase 1 crashes.

## Check Progress

```bash
# Is it still running?
ps -p $(cat ~/projects/autoresearch-macos/retest_ar96.pid 2>/dev/null) 2>/dev/null && echo "RUNNING" || echo "DONE"

# Watch live output
tail -f ~/projects/autoresearch-macos/retest_ar96.log

# Quick status (which phase, any crashes?)
grep -E "^(Phase|CRASH|VERDICT|✓|val_bpb)" ~/projects/autoresearch-macos/retest_ar96.log
```

## Get Results

```bash
# Full log
cat ~/projects/autoresearch-macos/retest_ar96.log

# Just the numbers
grep "^val_bpb:\|^num_steps:\|^num_params_M:" ~/projects/autoresearch-macos/retest_ar96_5min.log 2>/dev/null; echo "---"; grep "^val_bpb:\|^num_steps:\|^num_params_M:" ~/projects/autoresearch-macos/retest_ar96_30min.log 2>/dev/null

# Results in TSV
grep "retest" ~/projects/autoresearch-macos/results.tsv
```

## Resume With Claude

Paste this:

```
Autoresearch retest results are in. Here's what happened:

<paste output of: cat ~/projects/autoresearch-macos/retest_ar96.log>

And the numbers:

<paste output of: grep "retest" ~/projects/autoresearch-macos/results.tsv>

Analyze: did AR=96 survive? Update progress.txt and memory. If survived,
we have 3 points — fit the scaling law. If crashed, design sweep7 with
HEAD_DIM=64.
```
