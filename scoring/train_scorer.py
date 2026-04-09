"""
Train baseline scoring model and measure recall@K.

Loads dataset.csv, builds TF-IDF + source features, trains LogisticRegression,
evaluates via 5-fold stratified CV, and compares to keyword scoring baseline.

Usage: python scoring/train_scorer.py [--config PATH]
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Keyword scoring baselines from discovery analysis
KEYWORD_BASELINES = {
    100: 0.263,
    200: 0.386,
    300: 0.544,   # interpolated
    400: 0.702,
    672: 0.965,
}

DEFAULT_CONFIG = {
    "tfidf": {
        "max_features": 500,
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
    },
    "features": {
        "use_source_id": True,
        "use_source_kind": True,
        "use_url_domain": False,
        "use_summary_length": True,
        "use_mission_count": True,
    },
    "model": {
        "type": "logistic_regression",
        "C": 1.0,
        "class_weight": "balanced",
    },
    "label": {
        "treat_soft_as": "positive",
        "gold_weight": 3.0,
    },
}


def load_dataset(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_config(path: Optional[str]) -> dict:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_CONFIG


def build_labels(rows: list[dict], config: dict) -> np.ndarray:
    """Binary labels: positive (gold+soft) vs negative."""
    label_cfg = config.get("label", {})
    treat_soft = label_cfg.get("treat_soft_as", "positive")

    labels = []
    for r in rows:
        if r["label"] == "gold":
            labels.append(1)
        elif r["label"] == "soft":
            labels.append(1 if treat_soft == "positive" else 0)
        else:
            labels.append(0)
    return np.array(labels)


def build_sample_weights(rows: list[dict], config: dict) -> np.ndarray:
    """Gold items get higher weight than soft positives."""
    gold_weight = config.get("label", {}).get("gold_weight", 3.0)
    weights = []
    for r in rows:
        if r["label"] == "gold":
            weights.append(gold_weight)
        else:
            weights.append(1.0)
    return np.array(weights)


def build_features(
    rows,
    config,
    tfidf=None,
    fit=True,
    category_vocab=None,
):
    """Build sparse feature matrix from text + categorical + numeric features.

    Returns (X, tfidf, feature_names, category_vocab).
    Pass category_vocab from training into test to ensure consistent dimensions.
    """
    tfidf_cfg = config.get("tfidf", {})
    feat_cfg = config.get("features", {})

    # Text features
    texts = [f"{r['title_text']} {r['summary_text']}" for r in rows]

    if tfidf is None:
        tfidf = TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 500),
            min_df=tfidf_cfg.get("min_df", 2),
            max_df=tfidf_cfg.get("max_df", 0.95),
            ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        )

    if fit:
        text_features = tfidf.fit_transform(texts)
    else:
        text_features = tfidf.transform(texts)

    feature_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    parts = [text_features]

    # Build or reuse category vocabulary
    if category_vocab is None:
        category_vocab = {}
        if feat_cfg.get("use_source_id", True):
            category_vocab["source_ids"] = sorted(set(r["source_id"] for r in rows))
        if feat_cfg.get("use_source_kind", True):
            category_vocab["source_kinds"] = sorted(set(r["source_kind"] for r in rows))

    # Categorical: sourceId one-hot (fixed vocabulary)
    if feat_cfg.get("use_source_id", True):
        source_ids = category_vocab["source_ids"]
        source_id_map = {s: i for i, s in enumerate(source_ids)}
        source_mat = np.zeros((len(rows), len(source_ids)))
        for i, r in enumerate(rows):
            idx = source_id_map.get(r["source_id"])
            if idx is not None:
                source_mat[i, idx] = 1.0
        parts.append(csr_matrix(source_mat))
        feature_names.extend(f"source_id_{s}" for s in source_ids)

    # Categorical: sourceKind one-hot (fixed vocabulary)
    if feat_cfg.get("use_source_kind", True):
        source_kinds = category_vocab["source_kinds"]
        kind_map = {k: i for i, k in enumerate(source_kinds)}
        kind_mat = np.zeros((len(rows), len(source_kinds)))
        for i, r in enumerate(rows):
            idx = kind_map.get(r["source_kind"])
            if idx is not None:
                kind_mat[i, idx] = 1.0
        parts.append(csr_matrix(kind_mat))
        feature_names.extend(f"source_kind_{k}" for k in source_kinds)

    # Numeric: summary_length (log-scaled)
    if feat_cfg.get("use_summary_length", True):
        lengths = np.array([np.log1p(float(r["summary_length"])) for r in rows]).reshape(-1, 1)
        parts.append(csr_matrix(lengths))
        feature_names.append("summary_length_log")

    # Numeric: mission_count
    if feat_cfg.get("use_mission_count", True):
        missions = np.array([float(r["mission_count"]) for r in rows]).reshape(-1, 1)
        parts.append(csr_matrix(missions))
        feature_names.append("mission_count")

    X = hstack(parts, format="csr")
    return X, tfidf, feature_names, category_vocab


def build_model(config: dict) -> LogisticRegression:
    model_cfg = config.get("model", {})
    return LogisticRegression(
        C=model_cfg.get("C", 1.0),
        class_weight=model_cfg.get("class_weight", "balanced"),
        max_iter=1000,
        solver="lbfgs",
    )


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Fraction of positives in the top-k scored items."""
    if k >= len(y_true):
        return float(y_true.sum() > 0)
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    return y_true[top_k_indices].sum() / max(y_true.sum(), 1)


def precision_at_recall(
    y_true: np.ndarray, y_scores: np.ndarray, target_recall: float
) -> float:
    """Precision at the threshold where recall >= target_recall."""
    positives = y_true.sum()
    if positives == 0:
        return 0.0

    sorted_indices = np.argsort(y_scores)[::-1]
    tp = 0
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            tp += 1
        current_recall = tp / positives
        if current_recall >= target_recall:
            return tp / (i + 1)
    return tp / len(y_true)


def evaluate_cv(rows: list[dict], config: dict, n_splits: int = 5) -> dict:
    """Run stratified cross-validation and return metrics."""
    y = build_labels(rows, config)
    sample_weights = build_sample_weights(rows, config)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ks = [100, 200, 300, 400, 672]

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(rows, y)):
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = sample_weights[train_idx]

        X_train, tfidf, feature_names, cat_vocab = build_features(train_rows, config, fit=True)
        X_test, _, _, _ = build_features(test_rows, config, tfidf=tfidf, fit=False, category_vocab=cat_vocab)

        model = build_model(config)
        model.fit(X_train, y_train, sample_weight=w_train)

        y_scores = model.predict_proba(X_test)[:, 1]

        fold_result = {"fold": fold}

        # recall@K — scale K proportionally to test set size
        scale = len(test_idx) / len(rows)
        for k in ks:
            scaled_k = max(1, int(k * scale))
            fold_result[f"recall@{k}"] = recall_at_k(y_test, y_scores, scaled_k)

        fold_result["precision@recall_0.95"] = precision_at_recall(y_test, y_scores, 0.95)

        if len(set(y_test)) > 1:
            fold_result["auc_roc"] = roc_auc_score(y_test, y_scores)
        else:
            fold_result["auc_roc"] = float("nan")

        fold_metrics.append(fold_result)

    # Average across folds
    avg = {}
    metric_keys = [k for k in fold_metrics[0] if k != "fold"]
    for key in metric_keys:
        vals = [fm[key] for fm in fold_metrics if not (isinstance(fm[key], float) and np.isnan(fm[key]))]
        avg[key] = float(np.mean(vals)) if vals else float("nan")

    return {"fold_metrics": fold_metrics, "average": avg}


def train_full(rows: list[dict], config: dict):
    """Train on full dataset for export."""
    y = build_labels(rows, config)
    sample_weights = build_sample_weights(rows, config)
    X, tfidf, feature_names, cat_vocab = build_features(rows, config, fit=True)

    model = build_model(config)
    model.fit(X, y, sample_weight=sample_weights)

    return model, tfidf, feature_names, cat_vocab, X, y


def format_report(cv_results: dict, config: dict) -> str:
    avg = cv_results["average"]
    lines = [
        "=" * 60,
        "SCORING MODEL BASELINE REPORT",
        "=" * 60,
        "",
        "Config:",
        f"  TF-IDF: max_features={config['tfidf']['max_features']}, "
        f"ngram_range={config['tfidf']['ngram_range']}",
        f"  Features: {', '.join(k for k, v in config['features'].items() if v)}",
        f"  Model: {config['model']['type']} (C={config['model']['C']})",
        f"  Labels: soft_as={config['label']['treat_soft_as']}, "
        f"gold_weight={config['label']['gold_weight']}",
        "",
        "Cross-Validated Metrics (5-fold):",
        "",
        f"  {'Metric':<25} {'Model':>10} {'Keyword':>10} {'Delta':>10}",
        f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}",
    ]

    for k in [100, 200, 300, 400, 672]:
        model_val = avg.get(f"recall@{k}", 0)
        keyword_val = KEYWORD_BASELINES.get(k, 0)
        delta = model_val - keyword_val
        marker = " ✓" if delta > 0 else ""
        lines.append(
            f"  Recall@{k:<19} {model_val:>9.1%} {keyword_val:>9.1%} {delta:>+9.1%}{marker}"
        )

    lines.extend([
        "",
        f"  Precision@Recall=0.95: {avg.get('precision@recall_0.95', 0):.1%}",
        f"  AUC-ROC:               {avg.get('auc_roc', 0):.3f}",
        "",
        "Gate: recall@200 > 38.6% (keyword baseline)",
        f"Result: {avg.get('recall@200', 0):.1%} — "
        f"{'PASS' if avg.get('recall@200', 0) > 0.386 else 'FAIL'}",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train scoring model baseline")
    parser.add_argument("--config", help="Path to scoring config JSON")
    parser.add_argument("--dataset", default=os.path.join(SCRIPT_DIR, "dataset.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = load_dataset(args.dataset)
    print(f"Loaded {len(rows)} items from {args.dataset}")

    print("Running 5-fold cross-validation...")
    cv_results = evaluate_cv(rows, config)

    report = format_report(cv_results, config)
    print(report)

    # Save results
    results_path = os.path.join(SCRIPT_DIR, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump({"config": config, "cv_results": cv_results}, f, indent=2)
    print(f"Saved metrics to {results_path}")

    report_path = os.path.join(SCRIPT_DIR, "model_comparison.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
