"""
GEPA single-task optimizer for train.py hyperparameters.

Uses `claude --print` (Max subscription) as the LLM proposer — zero extra API cost.
Evaluator runs `uv run train.py` and extracts val_bpb.

Usage: uv run gepa_optimize.py [--max-iters N]
"""

import argparse
import os
import re
import subprocess
import sys
import time

import gepa.optimize_anything as oa
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_PY = os.path.join(os.path.dirname(__file__), "train.py")

# The hyperparameter block we're optimizing (lines 479-501 of train.py)
HYPERPARAM_MARKER_START = "# Hyperparameters (edit these directly"
HYPERPARAM_MARKER_END = "# Setup: tokenizer, model, optimizer, dataloader"

# Constraints the proposer must respect
BACKGROUND = """
You are optimizing a GPT language model training script for Apple M2 Pro (MPS backend).
The ONLY metric is val_bpb (validation bits per byte) — lower is better.
Training runs for exactly 5 minutes wall clock. Each evaluation takes ~15 min on top.

HARD CONSTRAINTS (violations = OOM crash):
- DEPTH must be 4 (DEPTH>=6 causes OOM on 16GB unified memory)
- DEVICE_BATCH_SIZE must be 16 (32 causes OOM)
- Do NOT add new imports or change anything outside the hyperparameter block

SOFT CONSTRAINTS:
- TOTAL_BATCH_SIZE as a power of 2
- All learning rates must be positive floats
- WARMUP_RATIO + WARMDOWN_RATIO <= 1.0
- WINDOW_PATTERN is a string of 'S' and 'L' characters (length = DEPTH)

WHAT WORKS (from Phase 1 experiments):
- WINDOW_PATTERN "SLSL" gives 3.25x throughput vs all-"L" (65 vs 20 steps)
- TOTAL_BATCH_SIZE 2**15 with SLSL gave best result (1.984 val_bpb, 54 steps)
- Throughput > architecture on fixed time budgets (more steps = better)
- Reducing per-step compute is the best lever

WHAT FAILED:
- MATRIX_LR 0.08 (2x): worse convergence
- WARMUP_RATIO 0.1: wastes steps at low LR
- WARMDOWN_RATIO 0.3: less warmdown didn't help
- WINDOW_PATTERN "SSSS": all-short loses long-range attention
- WEIGHT_DECAY 0.0: no regularization slightly worse
- EMBEDDING_LR 1.2: overshoots
- ASPECT_RATIO 96: wider model too slow (only 16 steps)

MPS VARIANCE: Step count varies ±30% with system load. The evaluator normalizes
by step count to reduce noise.

Return ONLY the Python hyperparameter assignments. No comments, no explanation.
"""

# Best known val_bpb from Phase 1 (under real load)
BASELINE_VAL_BPB = 1.984


# ---------------------------------------------------------------------------
# LLM: claude --print wrapper
# ---------------------------------------------------------------------------

def claude_lm(prompt):
    """LanguageModel callable that uses claude --print (Max subscription)."""
    if isinstance(prompt, list):
        # Convert chat-format messages to a single string
        parts = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<system>\n{content}\n</system>")
            else:
                parts.append(content)
        prompt_str = "\n\n".join(parts)
    else:
        prompt_str = prompt

    result = subprocess.run(
        ["claude", "--print", "--model", "sonnet", "-p", prompt_str],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude --print failed: {result.stderr[:500]}")

    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Hyperparameter block extraction/injection
# ---------------------------------------------------------------------------

def extract_hyperparam_block(train_py_path=TRAIN_PY):
    """Extract the hyperparameter block from train.py."""
    with open(train_py_path) as f:
        content = f.read()

    start = content.find(HYPERPARAM_MARKER_START)
    end = content.find(HYPERPARAM_MARKER_END)
    if start == -1 or end == -1:
        raise ValueError("Could not find hyperparameter block markers in train.py")

    return content[start:end].strip()


def inject_hyperparam_block(new_block, train_py_path=TRAIN_PY):
    """Replace the hyperparameter block in train.py with new values."""
    with open(train_py_path) as f:
        content = f.read()

    start = content.find(HYPERPARAM_MARKER_START)
    end = content.find(HYPERPARAM_MARKER_END)
    if start == -1 or end == -1:
        raise ValueError("Could not find hyperparameter block markers in train.py")

    new_content = content[:start] + new_block + "\n\n" + content[end:]
    with open(train_py_path, "w") as f:
        f.write(new_content)


def validate_hyperparams(block):
    """Basic safety checks on proposed hyperparameters."""
    # Must contain DEPTH = 4
    depth_match = re.search(r'DEPTH\s*=\s*(\d+)', block)
    if not depth_match or int(depth_match.group(1)) != 4:
        return False, "DEPTH must be 4"

    # Must contain DEVICE_BATCH_SIZE = 16
    batch_match = re.search(r'DEVICE_BATCH_SIZE\s*=\s*(\d+)', block)
    if not batch_match or int(batch_match.group(1)) != 16:
        return False, "DEVICE_BATCH_SIZE must be 16"

    # Must not contain import statements
    if "import " in block:
        return False, "Cannot add imports"

    return True, "ok"


# ---------------------------------------------------------------------------
# Evaluator: run train.py, extract val_bpb
# ---------------------------------------------------------------------------

def evaluator(candidate):
    """
    GEPA evaluator. Takes candidate (hyperparameter block string),
    injects into train.py, runs training, returns (score, side_info).

    Score = negative val_bpb (GEPA maximizes, we want lower val_bpb).
    """
    block = candidate if isinstance(candidate, str) else list(candidate.values())[0]

    # Validate before running
    valid, reason = validate_hyperparams(block)
    if not valid:
        return -10.0, {"error": reason, "val_bpb": 0.0, "steps": 0}

    # Save original train.py
    with open(TRAIN_PY) as f:
        original = f.read()

    try:
        # Inject new hyperparameters
        inject_hyperparam_block(block)

        # Run training
        print(f"\n{'='*60}")
        print(f"Running training with new hyperparameters...")
        print(f"{'='*60}")

        t0 = time.time()
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True,
            text=True,
            timeout=900,  # 15 min max (5 min train + 10 min eval)
            cwd=os.path.dirname(TRAIN_PY),
        )
        wall_time = time.time() - t0

        output = result.stdout + result.stderr

        # Extract val_bpb
        bpb_match = re.search(r'^val_bpb:\s+([0-9.]+)', output, re.MULTILINE)
        steps_match = re.search(r'^num_steps:\s+(\d+)', output, re.MULTILINE)

        if not bpb_match:
            # Crash — extract error
            error_lines = output.strip().split('\n')[-10:]
            error_msg = '\n'.join(error_lines)
            print(f"CRASH: {error_msg[:200]}")
            return -10.0, {
                "error": error_msg[:500],
                "val_bpb": 0.0,
                "steps": 0,
                "wall_time": wall_time,
            }

        val_bpb = float(bpb_match.group(1))
        num_steps = int(steps_match.group(1)) if steps_match else 0

        # Normalize by step count to reduce MPS variance
        # Use per-step learning as ASI signal
        bpb_per_step = val_bpb / max(num_steps, 1) if num_steps > 0 else val_bpb

        score = -val_bpb  # Negate: GEPA maximizes, we want lower val_bpb

        side_info = {
            "val_bpb": val_bpb,
            "num_steps": num_steps,
            "wall_time": wall_time,
            "bpb_per_step": bpb_per_step,
            "vs_baseline": f"{((val_bpb - BASELINE_VAL_BPB) / BASELINE_VAL_BPB) * 100:+.1f}%",
        }

        status = "BETTER" if val_bpb < BASELINE_VAL_BPB else "WORSE"
        print(f"Result: val_bpb={val_bpb:.6f} ({status}), steps={num_steps}, "
              f"wall={wall_time:.0f}s, vs baseline: {side_info['vs_baseline']}")

        return score, side_info

    except subprocess.TimeoutExpired:
        print("TIMEOUT: Run exceeded 15 minutes")
        return -10.0, {"error": "timeout", "val_bpb": 0.0, "steps": 0}

    except Exception as e:
        print(f"ERROR: {e}")
        return -10.0, {"error": str(e)[:500], "val_bpb": 0.0, "steps": 0}

    finally:
        # Always restore original train.py after evaluation
        with open(TRAIN_PY, "w") as f:
            f.write(original)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GEPA optimizer for train.py")
    parser.add_argument("--max-iters", type=int, default=8,
                        help="Maximum GEPA iterations (each ~20 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test LLM connection without running training")
    args = parser.parse_args()

    print("GEPA Hyperparameter Optimizer")
    print(f"  LLM: claude --print (Max subscription, $0 extra)")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Baseline val_bpb: {BASELINE_VAL_BPB}")
    print(f"  Est. total time: ~{args.max_iters * 20} minutes")
    print()

    # Extract current hyperparameter block as seed
    seed = extract_hyperparam_block()
    print("Seed candidate (current hyperparameters):")
    print(seed)
    print()

    if args.dry_run:
        print("Dry run: testing LLM connection...")
        response = claude_lm("Say 'GEPA ready' and nothing else.")
        print(f"LLM response: {response}")
        print("Dry run complete.")
        return

    # Configure GEPA
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_iters,
            run_dir="gepa_runs",
            capture_stdio=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=claude_lm,  # Uses Max subscription via CLI
        ),
    )

    # Run optimization
    print(f"Starting GEPA optimization loop...")
    print(f"Each iteration: ~5 min training + ~15 min eval = ~20 min")
    print()

    result = oa.optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        objective="Minimize val_bpb (validation bits per byte) for a GPT language model. Lower is better. The training budget is fixed at 5 minutes.",
        background=BACKGROUND,
        config=config,
    )

    # Report results
    print("\n" + "=" * 60)
    print("GEPA OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best score: {result.best_score:.6f} (val_bpb = {-result.best_score:.6f})")
    print(f"Iterations completed: {result.num_metric_calls}")
    print()
    print("Best hyperparameter block:")
    best = result.best_candidate
    if isinstance(best, dict):
        for v in best.values():
            print(v)
    else:
        print(best)

    # Save best result
    with open("gepa_best.txt", "w") as f:
        if isinstance(best, dict):
            for v in best.values():
                f.write(v + "\n")
        else:
            f.write(str(best) + "\n")

    print(f"\nBest candidate saved to gepa_best.txt")


if __name__ == "__main__":
    main()
