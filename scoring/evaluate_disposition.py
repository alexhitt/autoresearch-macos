"""
Evaluate a disposition prompt against the ground truth test set.

For each item in the test set, constructs the full prompt (template +
system context + abstract + claims), calls claude --print, parses the
JSON output, and compares dispositions to ground truth labels.

Metrics:
  - apply_now_precision: TP / (TP + FP)
  - apply_now_recall: TP / (TP + FN)
  - apply_now_f1: harmonic mean (PRIMARY COMPOSITE)
  - overall_accuracy: correct binary predictions / total
  - error_details: list of misclassified claims

Usage:
  python scoring/evaluate_disposition.py [--prompt PATH] [--testset PATH] [--verbose]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PROMPT_PATH = os.path.join(SCRIPT_DIR, "disposition_prompt.txt")
DEFAULT_TESTSET_PATH = os.path.join(SCRIPT_DIR, "disposition_testset.json")

# Import system context builder
sys.path.insert(0, SCRIPT_DIR)
from build_system_context import build_system_context


def load_prompt(path):
    with open(path) as f:
        return f.read()


def load_testset(path):
    with open(path) as f:
        return json.load(f)


def format_claim_list(claims):
    """Format claims the same way production does."""
    lines = []
    for i, c in enumerate(claims):
        lines.append(f'{i + 1}. [id: {c["id"]}] [{c["category"]}] "{c["claim"]}"')
    return "\n".join(lines)


def assemble_prompt(template, system_context, abstract, claim_list):
    """Fill sentinel placeholders in the prompt template."""
    prompt = template.replace("__SYSTEM_CONTEXT__", system_context)
    prompt = prompt.replace("__ABSTRACT__", abstract)
    prompt = prompt.replace("__CLAIM_LIST__", claim_list)
    return prompt


def strip_markdown_fences(text):
    """Remove ```json ... ``` wrappers."""
    text = re.sub(r"^[\s]*```(?:json)?[\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s]*```[\s]*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_claude_output(raw):
    """Parse claude --print --output-format json response."""
    parsed = json.loads(raw)
    # claude --print with --output-format json wraps in {"result": "..."}
    content = parsed
    if isinstance(parsed, dict) and "result" in parsed:
        content = parsed["result"]
    if isinstance(content, str):
        content = strip_markdown_fences(content)
        content = json.loads(content)
    if not isinstance(content, list):
        raise ValueError(f"Expected JSON array, got {type(content).__name__}")
    return content


def run_claude_print(prompt, timeout=120):
    """Call claude --print and return raw stdout. Retries once after 60s on failure."""
    result = subprocess.run(
        ["claude", "--print", "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"claude --print failed, retrying in 60s...", file=sys.stderr)
        time.sleep(60)
        result = subprocess.run(
            ["claude", "--print", "-p", prompt, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude --print failed (rc={result.returncode}): {result.stderr[:300]}")
    return result.stdout.strip()


def evaluate_item(template, system_context, item, verbose=False):
    """Evaluate one item's claims. Returns (results_list, wall_time)."""
    claim_list = format_claim_list(item["claims"])
    prompt = assemble_prompt(template, system_context, item["summary"], claim_list)

    t0 = time.time()
    try:
        raw = run_claude_print(prompt)
        dispositions = parse_claude_output(raw)
    except Exception as e:
        wall_time = time.time() - t0
        if verbose:
            print(f"    ERROR: {e}")
        # Return all claims as errors
        return [
            {
                "claim_id": c["id"],
                "ground_truth": c["ground_truth"],
                "predicted": "error",
                "error": str(e)[:200],
            }
            for c in item["claims"]
        ], wall_time

    wall_time = time.time() - t0

    # Build lookup from response
    disp_by_id = {}
    for d in dispositions:
        cid = d.get("claimId")
        if cid:
            disp_by_id[cid] = d.get("disposition", "unknown")

    results = []
    for c in item["claims"]:
        predicted_raw = disp_by_id.get(c["id"], "missing")
        # Map to binary: apply_now vs not_apply_now
        if predicted_raw == "apply_now":
            predicted = "apply_now"
        elif predicted_raw in ("candidate_insight", "discard"):
            predicted = "not_apply_now"
        elif predicted_raw == "missing":
            predicted = "error"
        else:
            predicted = "not_apply_now"  # unknown dispositions count as not_apply_now

        results.append({
            "claim_id": c["id"],
            "claim_text": c["claim"][:120],
            "ground_truth": c["ground_truth"],
            "predicted": predicted,
            "predicted_raw": predicted_raw,
            "correct": predicted == c["ground_truth"],
        })

    return results, wall_time


def compute_metrics(all_results):
    """Compute aggregate metrics from per-claim results."""
    valid = [r for r in all_results if r["predicted"] != "error"]
    errors = [r for r in all_results if r["predicted"] == "error"]

    if not valid:
        return {
            "apply_now_f1": 0.0,
            "apply_now_precision": 0.0,
            "apply_now_recall": 0.0,
            "overall_accuracy": 0.0,
            "total_claims": len(all_results),
            "error_count": len(errors),
            "error_details": [],
        }

    tp = sum(1 for r in valid if r["ground_truth"] == "apply_now" and r["predicted"] == "apply_now")
    fp = sum(1 for r in valid if r["ground_truth"] == "not_apply_now" and r["predicted"] == "apply_now")
    fn = sum(1 for r in valid if r["ground_truth"] == "apply_now" and r["predicted"] != "apply_now")
    tn = sum(1 for r in valid if r["ground_truth"] == "not_apply_now" and r["predicted"] != "apply_now")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(valid)

    # Collect misclassified claims for proposer feedback
    misclassified = [
        {
            "claim_id": r["claim_id"],
            "claim_text": r["claim_text"],
            "ground_truth": r["ground_truth"],
            "predicted": r["predicted"],
            "predicted_raw": r["predicted_raw"],
        }
        for r in valid
        if not r["correct"]
    ]

    return {
        "apply_now_f1": round(f1, 4),
        "apply_now_precision": round(precision, 4),
        "apply_now_recall": round(recall, 4),
        "overall_accuracy": round(accuracy, 4),
        "total_claims": len(all_results),
        "valid_claims": len(valid),
        "error_count": len(errors),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "error_details": misclassified,
    }


def evaluate_full(prompt_path=None, testset_path=None, verbose=False):
    """Run full evaluation, return metrics dict."""
    template = load_prompt(prompt_path or DEFAULT_PROMPT_PATH)
    testset = load_testset(testset_path or DEFAULT_TESTSET_PATH)
    system_context = build_system_context()

    all_results = []
    total_time = 0.0

    for i, item in enumerate(testset["items"]):
        n_claims = len(item["claims"])
        if verbose:
            print(f"  [{i+1}/{len(testset['items'])}] {item['title'][:60]}... ({n_claims} claims)")

        results, wall_time = evaluate_item(template, system_context, item, verbose=verbose)
        all_results.extend(results)
        total_time += wall_time

        if verbose:
            correct = sum(1 for r in results if r.get("correct"))
            print(f"    {correct}/{n_claims} correct [{wall_time:.1f}s]")

    metrics = compute_metrics(all_results)
    metrics["wall_time_total"] = round(total_time, 1)
    metrics["wall_time_per_item"] = round(total_time / len(testset["items"]), 1) if testset["items"] else 0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate disposition prompt")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--testset", default=DEFAULT_TESTSET_PATH)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    args = parser.parse_args()

    if args.verbose:
        print("Evaluating disposition prompt...")
        print(f"  Prompt: {args.prompt}")
        print(f"  Test set: {args.testset}")
        print()

    metrics = evaluate_full(args.prompt, args.testset, verbose=args.verbose)

    if args.json:
        # Strip error_details for compact output
        out = {k: v for k, v in metrics.items() if k != "error_details"}
        print(json.dumps(out, indent=2))
    else:
        print(f"\nDISPOSITION EVALUATION RESULTS")
        print(f"{'=' * 40}")
        print(f"  apply_now F1:        {metrics['apply_now_f1']:.1%}")
        print(f"  apply_now precision: {metrics['apply_now_precision']:.1%}")
        print(f"  apply_now recall:    {metrics['apply_now_recall']:.1%}")
        print(f"  Overall accuracy:    {metrics['overall_accuracy']:.1%}")
        print(f"  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
        print(f"  Claims: {metrics['valid_claims']}/{metrics['total_claims']} evaluated ({metrics['error_count']} errors)")
        print(f"  Wall time: {metrics['wall_time_total']:.0f}s ({metrics['wall_time_per_item']:.0f}s/item)")

        if metrics["error_details"]:
            print(f"\nMISCLASSIFIED ({len(metrics['error_details'])}):")
            for e in metrics["error_details"]:
                print(f"  [{e['ground_truth']}→{e['predicted_raw']}] {e['claim_text']}")


if __name__ == "__main__":
    main()
