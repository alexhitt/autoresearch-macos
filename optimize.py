"""
Direct hyperparameter optimization loop for train.py.

Simple propose → eval → keep/discard loop using claude --print as proposer.
One eval per iteration. No framework overhead. Matches program.md pattern.

Uses Max subscription via CLI — zero extra API cost.

Usage: uv run optimize.py [--max-iters N] [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_PY = os.path.join(os.path.dirname(__file__), "train.py")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "optimize_results.jsonl")

HYPERPARAM_MARKER_START = "# Hyperparameters (edit these directly"
HYPERPARAM_MARKER_END = "# Setup: tokenizer, model, optimizer, dataloader"

BASELINE_VAL_BPB = 1.984  # Best Phase 1 result under real load

# Phase 1 history for the proposer's context
PHASE1_HISTORY = """
PHASE 1 RESULTS (22 runs, sorted by val_bpb):
- SLSL + BATCH 2^15: val_bpb=1.984, 54 steps — BEST under load
- SLSL + WARMDOWN 0.3: val_bpb=2.001, 28 steps
- SLSL + BATCH 2^15 + WARMDOWN 0.3: val_bpb=2.001, 47 steps — doesn't stack
- SLSL only: val_bpb=2.012, 27 steps
- WEIGHT_DECAY 0.0: val_bpb=2.024, 26 steps
- EMBEDDING_LR 1.2: val_bpb=2.032, 24 steps
- WINDOW "SSSS": val_bpb=2.040, 23 steps
- Baseline "L" under load: val_bpb=2.051, 22 steps
- Baseline (overnight): val_bpb=2.067-2.128, 15-20 steps
- MATRIX_LR 0.08: val_bpb=2.093, 22 steps
- ASPECT_RATIO 96: val_bpb=2.142, 16 steps — wider model too slow
- WARMUP_RATIO 0.1: val_bpb=2.180, 24 steps
- DEPTH 6/8/12: OOM crash
- DEVICE_BATCH_SIZE 32: OOM crash

KEY INSIGHT: On MPS with fixed 5-min time budget, throughput is everything.
More steps = lower val_bpb. The only wins came from reducing per-step compute
(sliding windows). LR tuning had marginal effects.
NOTE: TOTAL_BATCH_SIZE minimum is 2**15 (32768). Smaller values crash.
"""


# ---------------------------------------------------------------------------
# Hyperparameter block extraction/injection
# ---------------------------------------------------------------------------

def extract_hyperparam_block():
    """Extract the hyperparameter block from train.py."""
    with open(TRAIN_PY) as f:
        content = f.read()
    start = content.find(HYPERPARAM_MARKER_START)
    end = content.find(HYPERPARAM_MARKER_END)
    if start == -1 or end == -1:
        raise ValueError("Could not find hyperparameter markers in train.py")
    return content[start:end].strip()


def inject_hyperparam_block(new_block):
    """Replace the hyperparameter block in train.py."""
    with open(TRAIN_PY) as f:
        content = f.read()
    start = content.find(HYPERPARAM_MARKER_START)
    end = content.find(HYPERPARAM_MARKER_END)
    if start == -1 or end == -1:
        raise ValueError("Could not find hyperparameter markers in train.py")
    new_content = content[:start] + new_block + "\n\n" + content[end:]
    with open(TRAIN_PY, "w") as f:
        f.write(new_content)


def validate_hyperparams(block):
    """Safety checks — prevent OOM and assertion crashes."""
    depth_match = re.search(r'DEPTH\s*=\s*(\d+)', block)
    if not depth_match or int(depth_match.group(1)) != 4:
        return False, "DEPTH must be 4 (OOM otherwise)"

    batch_match = re.search(r'DEVICE_BATCH_SIZE\s*=\s*(\d+)', block)
    if not batch_match or int(batch_match.group(1)) != 16:
        return False, "DEVICE_BATCH_SIZE must be 16 (OOM otherwise)"

    # TOTAL_BATCH_SIZE must be a multiple of tokens_per_fwdbwd (DEVICE_BATCH_SIZE * 2048 = 32768)
    total_batch_match = re.search(r'TOTAL_BATCH_SIZE\s*=\s*(.+?)(?:\s*#|$)', block, re.MULTILINE)
    if total_batch_match:
        try:
            total_batch = eval(total_batch_match.group(1).strip())
            if total_batch % 32768 != 0:
                return False, f"TOTAL_BATCH_SIZE ({total_batch}) must be multiple of 32768 (DEVICE_BATCH_SIZE * SEQ_LEN)"
        except Exception:
            return False, "Could not parse TOTAL_BATCH_SIZE"

    if "import " in block:
        return False, "Cannot add imports"

    return True, "ok"


# ---------------------------------------------------------------------------
# LLM proposer
# ---------------------------------------------------------------------------

def propose(current_block, history):
    """Ask claude --print to propose new hyperparameters."""

    history_text = ""
    if history:
        history_text = "\nOPTIMIZATION LOOP RESULTS SO FAR:\n"
        for i, h in enumerate(history):
            status = "BETTER (kept)" if h["kept"] else "WORSE (discarded)"
            history_text += (
                f"  Iter {i+1}: val_bpb={h['val_bpb']:.6f}, "
                f"steps={h['steps']}, {status}"
            )
            if h.get("error"):
                history_text += f" — ERROR: {h['error']}"
            history_text += "\n"

    prompt = f"""You are optimizing a GPT language model's hyperparameters.
The ONLY metric is val_bpb (validation bits per byte) — LOWER is better.
Training runs for exactly 5 minutes wall clock on Apple M2 Pro (MPS backend).

HARD CONSTRAINTS (violations = crash):
- DEPTH must be 4 (OOM otherwise)
- DEVICE_BATCH_SIZE must be 16 (OOM otherwise)
- TOTAL_BATCH_SIZE must be a MULTIPLE of 32768 (= DEVICE_BATCH_SIZE * SEQ_LEN). Minimum is 2**15=32768. Valid values: 2**15, 2**16, 3*2**15, 2**17, etc. Values like 2**14 or 2**13 CRASH.
- Do NOT add new imports or code — ONLY hyperparameter assignments

SOFT CONSTRAINTS:
- All learning rates must be positive floats
- WARMUP_RATIO + WARMDOWN_RATIO <= 1.0
- WINDOW_PATTERN is a string of 'S' and 'L' characters (length = DEPTH)

{PHASE1_HISTORY}
{history_text}

CURRENT BEST HYPERPARAMETERS (val_bpb={BASELINE_VAL_BPB}):
{current_block}

Propose a NEW set of hyperparameters that you think will achieve LOWER val_bpb.
Think about what hasn't been tried yet. Consider:
- Different WINDOW_PATTERN combinations (SLLS, LSSL, SSLS, etc.)
- TOTAL_BATCH_SIZE = 2**16 or 2**17 (larger batch, fewer but bigger steps)
- Fine-tuning learning rates together (not just one at a time)
- Different ADAM_BETAS values
- WARMDOWN_RATIO + FINAL_LR_FRAC combinations
- HEAD_DIM changes (64 instead of 128)
- SCALAR_LR adjustments

Return ONLY the complete Python hyperparameter block starting with the marker comment.
No explanation, no markdown fences. Just the code."""

    result = subprocess.run(
        ["claude", "--print", "--model", "opus", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude --print failed: {result.stderr[:500]}")

    response = result.stdout.strip()

    # Strip markdown fences if present
    if "```" in response:
        lines = response.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        response = "\n".join(lines)

    # Strip any non-code text the LLM may have included.
    # Find the marker comment that starts the block; discard everything before it.
    marker_idx = response.find("# Hyperparameters (edit these directly")
    if marker_idx > 0:
        response = response[marker_idx:]

    # Find the closing separator; discard everything after it.
    # The block ends with a line of dashes.
    closing = response.rfind("# ----")
    if closing == -1:
        closing = response.rfind("# -----")
    if closing != -1:
        # Keep through the end of that line
        end_of_line = response.find("\n", closing)
        if end_of_line != -1:
            response = response[:end_of_line + 1]

    # Final safety: remove any lines that aren't comments, assignments,
    # blank lines, or the separator. This catches stray prose.
    clean_lines = []
    for line in response.split("\n"):
        stripped = line.strip()
        if (stripped == "" or
            stripped.startswith("#") or
            "=" in stripped or
            stripped.startswith("WINDOW_PATTERN") or
            stripped.startswith("ADAM_BETAS")):
            clean_lines.append(line)
    response = "\n".join(clean_lines)

    return response


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def run_training():
    """Run train.py, return (val_bpb, num_steps, wall_time) or raise on crash."""
    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "train.py"],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=os.path.dirname(TRAIN_PY),
    )
    wall_time = time.time() - t0
    output = result.stdout + result.stderr

    bpb_match = re.search(r'^val_bpb:\s+([0-9.]+)', output, re.MULTILINE)
    steps_match = re.search(r'^num_steps:\s+(\d+)', output, re.MULTILINE)

    if not bpb_match:
        error_lines = output.strip().split('\n')[-15:]
        error_msg = '\n'.join(error_lines)
        raise RuntimeError(f"Training crashed:\n{error_msg[:500]}")

    return (
        float(bpb_match.group(1)),
        int(steps_match.group(1)) if steps_match else 0,
        wall_time,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Direct hyperparameter optimizer")
    parser.add_argument("--max-iters", type=int, default=8,
                        help="Maximum optimization iterations (each ~10 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test LLM connection without training")
    args = parser.parse_args()

    print("Direct Hyperparameter Optimizer")
    print(f"  LLM: claude --print --model opus (Max sub, $0)")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Baseline val_bpb: {BASELINE_VAL_BPB}")
    print(f"  Est. time: ~{args.max_iters * 10} min ({args.max_iters} iters × ~10 min)")
    print()

    current_block = extract_hyperparam_block()
    print("Current hyperparameters:")
    print(current_block)
    print()

    if args.dry_run:
        print("Dry run: testing LLM proposer...")
        proposal = propose(current_block, [])
        valid, reason = validate_hyperparams(proposal)
        print(f"Proposal valid: {valid} ({reason})")
        print(f"Proposal preview:\n{proposal[:300]}...")
        print("\nDry run complete.")
        return

    # Save original train.py for safety
    with open(TRAIN_PY) as f:
        original_train = f.read()

    best_bpb = BASELINE_VAL_BPB
    best_block = current_block
    history = []

    print(f"{'='*60}")
    print(f"Starting optimization loop ({args.max_iters} iterations)")
    print(f"{'='*60}")

    for i in range(args.max_iters):
        print(f"\n{'─'*60}")
        print(f"ITERATION {i+1}/{args.max_iters}")
        print(f"{'─'*60}")
        iter_start = time.time()

        # Step 1: Propose
        print("Proposing new hyperparameters...")
        try:
            proposal = propose(best_block, history)
        except Exception as e:
            print(f"  Proposer failed: {e}")
            history.append({
                "val_bpb": 0.0, "steps": 0, "wall_time": 0,
                "kept": False, "error": f"proposer: {e}",
            })
            continue

        # Step 2: Validate
        valid, reason = validate_hyperparams(proposal)
        if not valid:
            print(f"  Rejected: {reason}")
            history.append({
                "val_bpb": 0.0, "steps": 0, "wall_time": 0,
                "kept": False, "error": f"validation: {reason}",
            })
            continue

        # Show what changed
        print("  Proposed changes vs current best:")
        for line in proposal.split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                if line not in best_block:
                    print(f"    CHANGED: {line}")

        # Step 3: Inject and run
        print("  Running training...")
        try:
            inject_hyperparam_block(proposal)
            val_bpb, steps, wall_time = run_training()
        except Exception as e:
            print(f"  CRASH: {e}")
            history.append({
                "val_bpb": 0.0, "steps": 0, "wall_time": time.time() - iter_start,
                "kept": False, "error": str(e)[:200],
            })
            continue
        finally:
            # Always restore — we only update train.py if we decide to keep
            with open(TRAIN_PY, "w") as f:
                f.write(original_train)

        # Step 4: Keep or discard
        delta_pct = ((val_bpb - best_bpb) / best_bpb) * 100
        kept = val_bpb < best_bpb

        if kept:
            print(f"  KEEP: val_bpb={val_bpb:.6f} ({delta_pct:+.1f}%), "
                  f"steps={steps}, wall={wall_time:.0f}s — NEW BEST")
            best_bpb = val_bpb
            best_block = proposal
        else:
            print(f"  DISCARD: val_bpb={val_bpb:.6f} ({delta_pct:+.1f}%), "
                  f"steps={steps}, wall={wall_time:.0f}s")

        entry = {
            "iter": i + 1,
            "val_bpb": val_bpb,
            "steps": steps,
            "wall_time": round(wall_time, 1),
            "kept": kept,
            "delta_pct": round(delta_pct, 2),
            "error": None,
        }
        history.append(entry)

        # Append to results file
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Iterations run: {len(history)}")
    print(f"Best val_bpb: {best_bpb:.6f} (baseline was {BASELINE_VAL_BPB})")
    print(f"Improvement: {((best_bpb - BASELINE_VAL_BPB) / BASELINE_VAL_BPB) * 100:+.1f}%")
    print()

    print("Run history:")
    for h in history:
        status = "KEEP" if h["kept"] else "DISCARD"
        if h.get("error"):
            status = "ERROR"
        print(f"  [{h.get('iter', '?')}] val_bpb={h['val_bpb']:.6f} "
              f"steps={h['steps']} {status}")
    print()

    if best_bpb < BASELINE_VAL_BPB:
        print("Best hyperparameter block:")
        print(best_block)
        with open("optimize_best.txt", "w") as f:
            f.write(best_block + "\n")
            f.write(f"\n# Best val_bpb: {best_bpb:.6f}\n")
            f.write(f"# vs baseline: {((best_bpb - BASELINE_VAL_BPB) / BASELINE_VAL_BPB) * 100:+.1f}%\n")
        print("\nSaved to optimize_best.txt")
    else:
        print("No improvement over baseline. Current config is already best.")


if __name__ == "__main__":
    main()
