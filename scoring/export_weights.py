"""
Export trained scoring model as JSON for TypeScript consumption.

Trains on the full dataset, exports TF-IDF vocabulary + model coefficients
as scoring-model.json, then validates by round-tripping predictions.

Usage: python scoring/export_weights.py [--config PATH] [--output PATH]
"""

import argparse
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Export scoring model weights")
    parser.add_argument("--config", help="Path to scoring config JSON")
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "scoring-model.json"),
        help="Output path for model weights JSON",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(SCRIPT_DIR, "dataset.csv"),
    )
    args = parser.parse_args()

    # Import from train_scorer (same directory)
    from train_scorer import load_dataset, load_config, train_full, build_features

    config = load_config(args.config)
    rows = load_dataset(args.dataset)
    print(f"Training on {len(rows)} items...")

    model, tfidf, feature_names, cat_vocab, X, y = train_full(rows, config)

    # Build the export structure
    tfidf_cfg = config.get("tfidf", {})
    vocabulary = {word: int(idx) for word, idx in tfidf.vocabulary_.items()}

    export = {
        "version": 1,
        "config": config,
        "tfidf": {
            "vocabulary": vocabulary,
            "idf": tfidf.idf_.tolist(),
            "max_features": tfidf_cfg.get("max_features", 500),
            "ngram_range": list(tfidf_cfg.get("ngram_range", [1, 2])),
            "sublinear_tf": tfidf_cfg.get("sublinear_tf", True),
        },
        "categories": cat_vocab,
        "feature_names": feature_names,
        "model": {
            "type": "logistic_regression",
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        },
    }

    with open(args.output, "w") as f:
        json.dump(export, f)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"Exported {args.output} ({size_kb:.1f} KB)")
    print(f"  Vocabulary size: {len(vocabulary)}")
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Coefficient count: {len(export['model']['coefficients'])}")

    # Round-trip validation: reload and re-score
    print("\nRound-trip validation...")
    with open(args.output) as f:
        reloaded = json.load(f)

    # Verify structure
    assert reloaded["version"] == 1
    assert len(reloaded["model"]["coefficients"]) == len(feature_names)
    assert len(reloaded["tfidf"]["vocabulary"]) == len(vocabulary)
    assert len(reloaded["tfidf"]["idf"]) == len(tfidf.idf_)

    # Verify predictions match
    original_scores = model.predict_proba(X)[:, 1]

    coef = np.array(reloaded["model"]["coefficients"])
    intercept = reloaded["model"]["intercept"]
    # Logistic regression: P(y=1) = sigmoid(X @ coef + intercept)
    raw = X.toarray() @ coef + intercept
    reloaded_scores = 1 / (1 + np.exp(-raw))

    max_diff = np.max(np.abs(original_scores - reloaded_scores))
    mean_diff = np.mean(np.abs(original_scores - reloaded_scores))

    print(f"  Max score difference: {max_diff:.2e}")
    print(f"  Mean score difference: {mean_diff:.2e}")

    if max_diff < 1e-6:
        print("  Round-trip: PASS")
    else:
        print(f"  Round-trip: WARN (max diff {max_diff:.2e} > 1e-6)")

    # Rank agreement check
    original_rank = np.argsort(original_scores)[::-1]
    reloaded_rank = np.argsort(reloaded_scores)[::-1]
    rank_match = np.all(original_rank[:50] == reloaded_rank[:50])
    print(f"  Top-50 rank agreement: {'PASS' if rank_match else 'FAIL'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
