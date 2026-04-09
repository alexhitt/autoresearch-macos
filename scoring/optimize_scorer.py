"""
Autoresearch optimization loop for the scoring model.

Adapts optimize.py's propose → validate → train → evaluate → keep/revert
pattern to iterate on feature engineering and model selection for the
Intelligence Pipeline item scorer.

Uses claude --print (Max subscription, $0) as the LLM proposer.

Usage:
  python scoring/optimize_scorer.py [--max-iters N] [--dry-run]
  python scoring/optimize_scorer.py --max-iters 30  # full overnight run
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "scoring_config.json")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset.csv")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "optimize_results.jsonl")

# Import from train_scorer
from train_scorer import (
    load_dataset,
    load_config,
    evaluate_cv,
    DEFAULT_CONFIG,
)

# Keyword baselines from discovery
KEYWORD_BASELINES = {
    100: 0.263,
    200: 0.386,
    672: 0.965,
}

VALID_MODEL_TYPES = {"logistic_regression", "gradient_boosting", "random_forest"}


def load_history():
    """Load prior optimization results."""
    if not os.path.exists(RESULTS_FILE):
        return []
    history = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def compute_composite(avg_metrics):
    """Composite metric: 0.6 * recall@200 + 0.4 * precision@recall_0.95"""
    r200 = avg_metrics.get("recall@200", 0)
    p95 = avg_metrics.get("precision@recall_0.95", 0)
    return 0.6 * r200 + 0.4 * p95


def get_feature_importance(rows, config):
    """Train full model and return top feature importances."""
    from train_scorer import train_full
    model, tfidf, feature_names, cat_vocab, X, y = train_full(rows, config)
    coefs = model.coef_[0]

    # Top 15 positive and negative features
    indices = np.argsort(coefs)
    top_positive = [(feature_names[i], float(coefs[i])) for i in indices[-10:]][::-1]
    top_negative = [(feature_names[i], float(coefs[i])) for i in indices[:5]]

    return top_positive, top_negative


def validate_config(config):
    """Safety checks on proposed config."""
    tfidf = config.get("tfidf", {})
    model = config.get("model", {})
    label = config.get("label", {})
    features = config.get("features", {})

    mf = tfidf.get("max_features", 500)
    if not (100 <= mf <= 2000):
        return False, f"max_features={mf} not in [100, 2000]"

    min_df = tfidf.get("min_df", 2)
    if min_df < 2:
        return False, f"min_df={min_df} < 2"

    ngram = tfidf.get("ngram_range", [1, 2])
    if not (isinstance(ngram, list) and len(ngram) == 2 and 1 <= ngram[0] <= ngram[1] <= 3):
        return False, f"ngram_range={ngram} invalid"

    model_type = model.get("type", "logistic_regression")
    if model_type not in VALID_MODEL_TYPES:
        return False, f"model type '{model_type}' not in {VALID_MODEL_TYPES}"

    gw = label.get("gold_weight", 3.0)
    if not (1.0 <= gw <= 10.0):
        return False, f"gold_weight={gw} not in [1.0, 10.0]"

    # At least one feature category must be enabled
    any_feature = any(features.get(k, False) for k in features)
    if not any_feature:
        return False, "all features disabled"

    return True, "ok"


def propose(current_config, history, feature_importance, rows_info):
    """Ask claude --print to propose a new scoring config."""
    history_text = ""
    if history:
        history_text = "\nOPTIMIZATION HISTORY (most recent first):\n"
        for h in reversed(history[-15:]):
            status = "BETTER (kept)" if h.get("kept") else "WORSE (discarded)"
            composite = h.get("composite", 0)
            r200 = h.get("recall@200", 0)
            history_text += f"  Iter {h.get('iter', '?')}: composite={composite:.4f} R@200={r200:.1%} — {status}"
            if h.get("error"):
                history_text += f" ERROR: {h['error']}"
            history_text += "\n"

    top_pos, top_neg = feature_importance
    importance_text = "TOP POSITIVE FEATURES (predict valuable):\n"
    for name, coef in top_pos:
        importance_text += f"  {coef:+.4f}  {name}\n"
    importance_text += "\nTOP NEGATIVE FEATURES (predict not valuable):\n"
    for name, coef in top_neg:
        importance_text += f"  {coef:+.4f}  {name}\n"

    prompt = f"""You are optimizing a text classification model that scores research items for an AI intelligence pipeline.

The model predicts whether an item (paper, blog post, GitHub commit) will produce actionable claims when processed by the pipeline.

METRIC: composite = 0.6 * recall@200 + 0.4 * precision@recall_0.95 — HIGHER is better.
Keyword scoring baseline: R@200 = 38.6%

DATASET: {rows_info}

{importance_text}

{history_text}

CURRENT CONFIG (composite={compute_composite(history[-1]) if history and history[-1].get('kept') else 'baseline'}):
{json.dumps(current_config, indent=2)}

CONSTRAINTS:
- max_features: 100-2000 (vocabulary size for TF-IDF)
- min_df: >= 2 (minimum document frequency)
- ngram_range: [min, max] where 1 <= min <= max <= 3
- model.type: one of "logistic_regression", "gradient_boosting", "random_forest"
- model.C: regularization for logistic_regression (0.01 to 100)
- model.class_weight: "balanced" or null
- label.gold_weight: 1.0-10.0 (weight for gold vs soft positive items)
- label.treat_soft_as: "positive" or "negative" (whether candidate_insight counts as positive)
- features: toggle which feature groups to include
- tfidf.sublinear_tf: true/false (logarithmic TF scaling)
- tfidf.max_df: 0.5-1.0 (maximum document frequency)

STRATEGY NOTES:
- source_id features capture which RSS feeds/repos historically produce value (42x range in gold rates)
- TF-IDF captures text patterns that correlate with actionable content
- Larger ngram_range captures phrases but increases dimensionality
- Higher gold_weight makes the model focus more on the rare gold items
- gradient_boosting and random_forest can capture non-linear interactions but may overfit on 390 items

Propose a NEW config that you think will achieve a HIGHER composite score.
Think about what hasn't been tried. Consider combinations and interactions.

Return ONLY valid JSON — the complete config object. No explanation, no markdown fences."""

    result = subprocess.run(
        ["claude", "--print", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude --print failed: {result.stderr[:300]}")

    response = result.stdout.strip()

    # Strip markdown fences if present
    if "```" in response:
        lines = response.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response = "\n".join(lines).strip()

    return json.loads(response)


def evaluate_config(config, rows):
    """Train and evaluate a config, return metrics dict."""
    cv_results = evaluate_cv(rows, config)
    avg = cv_results["average"]
    composite = compute_composite(avg)
    return {
        "composite": composite,
        "recall@100": avg.get("recall@100", 0),
        "recall@200": avg.get("recall@200", 0),
        "recall@400": avg.get("recall@400", 0),
        "recall@672": avg.get("recall@672", 0),
        "precision@recall_0.95": avg.get("precision@recall_0.95", 0),
        "auc_roc": avg.get("auc_roc", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Autoresearch scoring optimizer")
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--early-stop", type=int, default=5,
                        help="Stop after N consecutive non-improvements")
    args = parser.parse_args()

    rows = load_dataset(DATASET_PATH)
    rows_info = f"{len(rows)} items (gold: {sum(1 for r in rows if r['label']=='gold')}, soft: {sum(1 for r in rows if r['label']=='soft')}, negative: {sum(1 for r in rows if r['label']=='negative')})"

    print("Autoresearch Scoring Optimizer")
    print(f"  Dataset: {rows_info}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Early stop: {args.early_stop} consecutive non-improvements")
    print(f"  LLM: claude --print (Max subscription, $0)")
    print()

    # Load or initialize config
    current_config = load_config(CONFIG_PATH)

    # Compute baseline
    print("Computing baseline metrics...")
    baseline_metrics = evaluate_config(current_config, rows)
    best_composite = baseline_metrics["composite"]
    best_config = current_config
    print(f"  Baseline composite: {best_composite:.4f}")
    print(f"  Baseline R@200: {baseline_metrics['recall@200']:.1%}")
    print()

    if args.dry_run:
        print("Dry run: testing LLM proposer...")
        fi = get_feature_importance(rows, current_config)
        proposal = propose(current_config, [], fi, rows_info)
        valid, reason = validate_config(proposal)
        print(f"  Proposal valid: {valid} ({reason})")
        print(f"  Proposal:\n{json.dumps(proposal, indent=2)[:500]}")
        print("\nDry run complete.")
        return

    history = load_history()
    no_improvement_streak = 0

    print(f"{'=' * 60}")
    print(f"Starting optimization ({args.max_iters} iterations)")
    print(f"{'=' * 60}")

    for i in range(args.max_iters):
        iter_num = len(history) + 1
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iter_num}")
        print(f"{'─' * 60}")

        # Step 1: Get feature importance for proposer context
        fi = get_feature_importance(rows, best_config)

        # Step 2: Propose
        print("Proposing new config...")
        try:
            proposal = propose(best_config, history, fi, rows_info)
        except Exception as e:
            print(f"  Proposer failed: {e}")
            entry = {"iter": iter_num, "kept": False, "error": f"proposer: {str(e)[:200]}"}
            history.append(entry)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            continue

        # Step 3: Validate
        valid, reason = validate_config(proposal)
        if not valid:
            print(f"  Rejected: {reason}")
            entry = {"iter": iter_num, "kept": False, "error": f"validation: {reason}"}
            history.append(entry)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            continue

        # Show key changes
        print("  Key changes:")
        for section in ["tfidf", "model", "label", "features"]:
            old = best_config.get(section, {})
            new = proposal.get(section, {})
            for k in set(list(old.keys()) + list(new.keys())):
                if old.get(k) != new.get(k):
                    print(f"    {section}.{k}: {old.get(k)} → {new.get(k)}")

        # Step 4: Evaluate
        print("  Training and evaluating (5-fold CV)...")
        t0 = time.time()
        try:
            metrics = evaluate_config(proposal, rows)
        except Exception as e:
            print(f"  CRASH: {e}")
            entry = {"iter": iter_num, "kept": False, "error": str(e)[:200]}
            history.append(entry)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            continue
        wall_time = time.time() - t0

        composite = metrics["composite"]
        delta = composite - best_composite
        kept = composite > best_composite

        # Step 5: Keep or discard
        if kept:
            print(f"  KEEP: composite={composite:.4f} ({delta:+.4f}) "
                  f"R@200={metrics['recall@200']:.1%} — NEW BEST [{wall_time:.1f}s]")
            best_composite = composite
            best_config = proposal
            no_improvement_streak = 0

            # Save best config
            with open(CONFIG_PATH, "w") as f:
                json.dump(best_config, f, indent=2)
        else:
            no_improvement_streak += 1
            print(f"  DISCARD: composite={composite:.4f} ({delta:+.4f}) "
                  f"R@200={metrics['recall@200']:.1%} [{wall_time:.1f}s] "
                  f"(streak: {no_improvement_streak}/{args.early_stop})")

        entry = {
            "iter": iter_num,
            "composite": round(composite, 6),
            "recall@200": round(metrics["recall@200"], 4),
            "precision@recall_0.95": round(metrics["precision@recall_0.95"], 4),
            "auc_roc": round(metrics["auc_roc"], 4),
            "kept": kept,
            "delta": round(delta, 6),
            "wall_time": round(wall_time, 1),
            "error": None,
            "config_snapshot": {
                "max_features": proposal["tfidf"]["max_features"],
                "ngram_range": proposal["tfidf"]["ngram_range"],
                "model_type": proposal["model"]["type"],
                "gold_weight": proposal["label"]["gold_weight"],
            },
        }
        history.append(entry)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Early stopping
        if no_improvement_streak >= args.early_stop:
            print(f"\nEarly stop: {args.early_stop} consecutive non-improvements.")
            break

    # Summary
    kept_count = sum(1 for h in history if h.get("kept"))
    print(f"\n{'=' * 60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"  Iterations: {len(history)}")
    print(f"  Improvements: {kept_count}")
    print(f"  Best composite: {best_composite:.4f}")
    print(f"  Best R@200: {best_config and evaluate_config(best_config, rows).get('recall@200', 0):.1%}")
    print(f"  Config saved to: {CONFIG_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
