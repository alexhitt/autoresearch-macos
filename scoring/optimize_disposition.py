"""
Autoresearch optimization loop for the disposition prompt.

Iteratively proposes prompt variants via claude --print, evaluates them
against the ground truth test set, and keeps improvements.

Uses claude --print (Max subscription, $0) as the LLM proposer.

Usage:
  python scoring/optimize_disposition.py [--max-iters N] [--dry-run]
  python scoring/optimize_disposition.py --max-iters 15  # full overnight run
"""

import argparse
import json
import os
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(SCRIPT_DIR, "disposition_prompt.txt")
TESTSET_PATH = os.path.join(SCRIPT_DIR, "disposition_testset.json")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "optimize_disposition_results.jsonl")

# Sentinels that must be present in every valid prompt
REQUIRED_SENTINELS = ["__SYSTEM_CONTEXT__", "__ABSTRACT__", "__CLAIM_LIST__"]
MAX_PROMPT_CHARS = 5000

# Import evaluator
import sys
sys.path.insert(0, SCRIPT_DIR)
from evaluate_disposition import evaluate_full


def load_prompt(path=None):
    with open(path or PROMPT_PATH) as f:
        return f.read()


def save_prompt(text, path=None):
    with open(path or PROMPT_PATH, "w") as f:
        f.write(text)


def load_history():
    if not os.path.exists(RESULTS_FILE):
        return []
    history = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def validate_prompt(prompt):
    """Safety checks on proposed prompt."""
    for sentinel in REQUIRED_SENTINELS:
        if sentinel not in prompt:
            return False, f"Missing sentinel: {sentinel}"

    if len(prompt) > MAX_PROMPT_CHARS:
        return False, f"Prompt too long: {len(prompt)} > {MAX_PROMPT_CHARS}"

    # Must contain JSON output instruction
    if "json" not in prompt.lower():
        return False, "Missing JSON output instruction"

    # Must contain claimId instruction
    if "claimId" not in prompt:
        return False, "Missing claimId instruction"

    # Must contain at least the three disposition types
    for disp in ["apply_now", "candidate_insight", "discard"]:
        if disp not in prompt:
            return False, f"Missing disposition type: {disp}"

    return True, "ok"


def propose(current_prompt, current_metrics, history):
    """Ask claude --print to propose a modified disposition prompt."""
    history_text = ""
    if history:
        history_text = "\nOPTIMIZATION HISTORY (most recent first):\n"
        for h in reversed(history[-10:]):
            status = "BETTER (kept)" if h.get("kept") else "WORSE (discarded)"
            f1 = h.get("apply_now_f1", 0)
            history_text += f"  Iter {h.get('iter', '?')}: F1={f1:.1%} — {status}"
            if h.get("error"):
                history_text += f" ERROR: {h['error']}"
            history_text += "\n"

    # Build error analysis
    error_text = ""
    errors = current_metrics.get("error_details", [])
    if errors:
        error_text = "\nMISCLASSIFIED CLAIMS (the prompt got these wrong):\n"
        for e in errors:
            error_text += f"  Ground truth: {e['ground_truth']} | Predicted: {e['predicted_raw']}\n"
            error_text += f"    Claim: \"{e['claim_text']}\"\n"

    proposer_prompt = f"""You are optimizing a classification prompt for an AI intelligence pipeline.

The prompt classifies research claims into: apply_now, candidate_insight, or discard.
It runs via "claude --print" with paper abstracts and system context injected.

EVALUATION METRIC: apply_now F1 score — HIGHER is better.
  - apply_now = claims the owner should act on (implemented or applied)
  - not_apply_now = claims the owner saw but chose to skip

CURRENT METRICS (baseline to beat):
  F1:        {current_metrics.get('apply_now_f1', 0):.1%}
  Precision: {current_metrics.get('apply_now_precision', 0):.1%}  (TP={current_metrics.get('tp', 0)} FP={current_metrics.get('fp', 0)})
  Recall:    {current_metrics.get('apply_now_recall', 0):.1%}  (FN={current_metrics.get('fn', 0)})
  Accuracy:  {current_metrics.get('overall_accuracy', 0):.1%}
{error_text}
{history_text}
CURRENT PROMPT:
---
{current_prompt}
---

CONSTRAINTS:
- You MUST keep these exact sentinel strings (they are replaced at runtime):
  __SYSTEM_CONTEXT__  (replaced with live AI OS state, rules, project status)
  __ABSTRACT__        (replaced with the paper's abstract)
  __CLAIM_LIST__      (replaced with numbered claims)
- You MUST keep the "claimId" instruction and "JSON array" output format instruction
- You MUST keep all three disposition types: apply_now, candidate_insight, discard
- Max length: {MAX_PROMPT_CHARS} characters
- You MUST keep the output fields for apply_now claims (actionSummary, targetSurface, impactLevel, whatThisIs, whatItMeans, whatChanges, whyNot) — the production system requires them

STRATEGY NOTES:
- False positives (FP={current_metrics.get('fp', 0)}): The prompt classified these as apply_now but the owner skipped them. Consider tightening the apply_now criteria.
- False negatives (FN={current_metrics.get('fn', 0)}): The prompt classified these as candidate_insight but the owner actually implemented them. Consider broadening what counts as actionable.
- The system context section contains existing rules — claims already covered should be discard, not apply_now.
- The "IMPORTANT" instruction about checking existing rules is critical for reducing false positives.
- Think about what makes a claim truly actionable vs merely interesting.

Propose a MODIFIED prompt that you believe will achieve a HIGHER F1 score.
Change the instruction text, criteria definitions, examples, or structure.
Do NOT change the sentinel strings or remove required output fields.

Return ONLY the full prompt text. No explanation, no markdown fences, no commentary."""

    result = subprocess.run(
        ["claude", "--print", "-p", proposer_prompt],
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude --print failed: {result.stderr[:300]}")

    response = result.stdout.strip()

    # Strip markdown fences if present
    if response.startswith("```"):
        lines = response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines).strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Autoresearch disposition prompt optimizer")
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--early-stop", type=int, default=4,
                        help="Stop after N consecutive non-improvements")
    parser.add_argument("--min-improvement", type=float, default=0.01,
                        help="Minimum F1 improvement to keep (noise floor guard)")
    args = parser.parse_args()

    print("Autoresearch Disposition Prompt Optimizer")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Early stop: {args.early_stop} consecutive non-improvements")
    print(f"  Min improvement: {args.min_improvement:.0%} F1")
    print(f"  LLM: claude --print (Max subscription, $0)")
    print()

    # Load current prompt
    current_prompt = load_prompt()
    print(f"Loaded prompt ({len(current_prompt)} chars)")

    # Compute baseline
    print("\nComputing baseline metrics (this takes ~10-15 minutes)...")
    baseline_metrics = evaluate_full(verbose=True)
    best_f1 = baseline_metrics["apply_now_f1"]
    best_prompt = current_prompt

    print(f"\n  Baseline F1:        {best_f1:.1%}")
    print(f"  Baseline precision: {baseline_metrics['apply_now_precision']:.1%}")
    print(f"  Baseline recall:    {baseline_metrics['apply_now_recall']:.1%}")
    print(f"  Baseline accuracy:  {baseline_metrics['overall_accuracy']:.1%}")
    print(f"  Baseline wall time: {baseline_metrics['wall_time_total']:.0f}s")

    if args.dry_run:
        print("\nDry run: testing proposer...")
        proposal = propose(current_prompt, baseline_metrics, [])
        valid, reason = validate_prompt(proposal)
        print(f"  Proposal valid: {valid} ({reason})")
        print(f"  Proposal length: {len(proposal)} chars")
        # Show a diff summary
        old_lines = set(current_prompt.splitlines())
        new_lines = set(proposal.splitlines())
        added = new_lines - old_lines
        removed = old_lines - new_lines
        print(f"  Lines added: {len(added)}, removed: {len(removed)}")
        if added:
            print(f"  Sample addition: {list(added)[0][:100]}")
        print("\nDry run complete.")
        return

    history = load_history()
    no_improvement_streak = 0
    consecutive_proposer_failures = 0

    print(f"\n{'=' * 60}")
    print(f"Starting optimization ({args.max_iters} iterations)")
    print(f"{'=' * 60}")

    for _ in range(args.max_iters):
        iter_num = len(history) + 1
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iter_num}")
        print(f"{'─' * 60}")

        # Step 1: Propose
        print("Proposing new prompt...")
        t_propose = time.time()
        try:
            proposal = propose(best_prompt, baseline_metrics if iter_num == 1 else history[-1].get("metrics", baseline_metrics), history)
        except Exception as e:
            consecutive_proposer_failures += 1
            backoff_times = {1: 60, 2: 120, 3: 240}
            sleep_time = backoff_times.get(consecutive_proposer_failures, 240)
            print(f"  Proposer failed: {e}")
            print(f"  proposer failed, backing off {sleep_time}s (attempt {consecutive_proposer_failures}/3)")
            entry = {"iter": iter_num, "kept": False, "error": f"proposer: {str(e)[:200]}"}
            history.append(entry)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            if consecutive_proposer_failures >= 3:
                print("  3 consecutive proposer failures — rate limit not recovering, stopping.")
                break
            time.sleep(sleep_time)
            continue
        consecutive_proposer_failures = 0  # reset on success
        propose_time = time.time() - t_propose
        print(f"  Proposed in {propose_time:.1f}s ({len(proposal)} chars)")

        # Step 2: Validate
        valid, reason = validate_prompt(proposal)
        if not valid:
            print(f"  Rejected: {reason}")
            entry = {"iter": iter_num, "kept": False, "error": f"validation: {reason}"}
            history.append(entry)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            continue

        # Step 3: Evaluate
        print("  Evaluating (21 items via claude --print)...")
        # Temporarily write proposal to a temp file for evaluation
        temp_prompt_path = os.path.join(SCRIPT_DIR, "disposition_prompt_candidate.txt")
        with open(temp_prompt_path, "w") as f:
            f.write(proposal)

        t_eval = time.time()
        try:
            metrics = evaluate_full(prompt_path=temp_prompt_path)
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            entry = {"iter": iter_num, "kept": False, "error": f"evaluation: {str(e)[:200]}"}
            history.append(entry)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            continue
        eval_time = time.time() - t_eval

        new_f1 = metrics["apply_now_f1"]
        delta = new_f1 - best_f1
        kept = delta > args.min_improvement

        # Step 4: Keep or discard
        if kept:
            print(f"  KEEP: F1={new_f1:.1%} ({delta:+.1%}) "
                  f"P={metrics['apply_now_precision']:.1%} R={metrics['apply_now_recall']:.1%} "
                  f"— NEW BEST [{eval_time:.0f}s]")
            best_f1 = new_f1
            best_prompt = proposal
            no_improvement_streak = 0

            # Save as the new current prompt
            save_prompt(best_prompt)
        else:
            no_improvement_streak += 1
            print(f"  DISCARD: F1={new_f1:.1%} ({delta:+.1%}) "
                  f"P={metrics['apply_now_precision']:.1%} R={metrics['apply_now_recall']:.1%} "
                  f"[{eval_time:.0f}s] (streak: {no_improvement_streak}/{args.early_stop})")

        entry = {
            "iter": iter_num,
            "apply_now_f1": round(new_f1, 4),
            "apply_now_precision": round(metrics["apply_now_precision"], 4),
            "apply_now_recall": round(metrics["apply_now_recall"], 4),
            "overall_accuracy": round(metrics["overall_accuracy"], 4),
            "kept": kept,
            "delta": round(delta, 4),
            "wall_time": round(eval_time, 1),
            "propose_time": round(propose_time, 1),
            "prompt_length": len(proposal),
            "error": None,
            "metrics": {
                "apply_now_f1": metrics["apply_now_f1"],
                "apply_now_precision": metrics["apply_now_precision"],
                "apply_now_recall": metrics["apply_now_recall"],
                "overall_accuracy": metrics["overall_accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "error_details": metrics["error_details"],
            },
        }
        history.append(entry)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Early stopping
        if no_improvement_streak >= args.early_stop:
            print(f"\nEarly stop: {args.early_stop} consecutive non-improvements.")
            break

    # Cleanup temp file
    temp_path = os.path.join(SCRIPT_DIR, "disposition_prompt_candidate.txt")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Summary
    kept_count = sum(1 for h in history if h.get("kept"))
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"  Iterations: {len(history)}")
    print(f"  Improvements: {kept_count}")
    print(f"  Best F1: {best_f1:.1%}")
    print(f"  Prompt saved to: {PROMPT_PATH}")
    print(f"{'=' * 60}")

    # Export winning prompt with metadata
    winner_path = os.path.join(SCRIPT_DIR, "disposition_prompt_winner.txt")
    with open(winner_path, "w") as f:
        f.write(f"# Disposition Prompt — Optimized\n")
        f.write(f"# Best F1: {best_f1:.1%}\n")
        f.write(f"# Iterations: {len(history)}, Improvements: {kept_count}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"#\n")
        f.write(f"# To deploy: copy the content below (excluding these comment lines)\n")
        f.write(f"# into application.ts:classifyClaimDispositions(), replacing the prompt\n")
        f.write(f"# template. Map sentinels: __SYSTEM_CONTEXT__ -> ${{systemContext}},\n")
        f.write(f"# __ABSTRACT__ -> ${{abstract}}, __CLAIM_LIST__ -> ${{claimList}}\n")
        f.write(f"#\n\n")
        f.write(best_prompt)
    print(f"  Winner exported to: {winner_path}")


if __name__ == "__main__":
    main()
