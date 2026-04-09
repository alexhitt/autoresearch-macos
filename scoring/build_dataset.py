"""
Build labeled dataset from Intelligence Pipeline data.

Joins items.json + claims.json to produce per-item labels based on
downstream disposition outcomes. Outputs dataset.csv + dataset_stats.txt.

Usage: python scoring/build_dataset.py [--signals-dir PATH]
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from urllib.parse import urlparse

DEFAULT_SIGNALS_DIR = os.path.expanduser("~/.lead-command/signals")


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def compute_item_labels(claims: list[dict]) -> dict[str, str]:
    """Assign each item a label based on its best claim disposition.

    Priority: gold (apply_now/queue_research) > soft (candidate_insight) > negative (discard).
    """
    item_disps: dict[str, set[str]] = defaultdict(set)
    for c in claims:
        d = c.get("disposition")
        if d:
            item_disps[c["signalItemId"]].add(d)

    labels = {}
    for iid, disps in item_disps.items():
        if "apply_now" in disps or "queue_research" in disps:
            labels[iid] = "gold"
        elif "candidate_insight" in disps:
            labels[iid] = "soft"
        elif "discard" in disps:
            labels[iid] = "negative"
    return labels


def extract_url_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return "unknown"


def build_dataset(signals_dir: str) -> tuple[list[dict], dict[str, int]]:
    items = load_json(os.path.join(signals_dir, "items.json"))
    claims = load_json(os.path.join(signals_dir, "claims.json"))

    labels = compute_item_labels(claims)
    items_by_id = {i["id"]: i for i in items}

    rows = []
    label_counts = defaultdict(int)

    for iid, label in labels.items():
        item = items_by_id.get(iid)
        if not item:
            continue

        title = item.get("title", "")
        summary = item.get("summary", "")[:2000]
        source_id = item.get("sourceId", "unknown")
        source_kind = item.get("sourceKind", "unknown")
        url_domain = extract_url_domain(item.get("url", ""))
        summary_length = len(item.get("summary", ""))
        mission_count = len(item.get("matchedMissions") or [])

        # Current keyword score for baseline comparison
        keyword_score = (item.get("score") or {}).get("overall", 0) or 0

        rows.append({
            "id": iid,
            "label": label,
            "source_id": source_id,
            "source_kind": source_kind,
            "url_domain": url_domain,
            "summary_length": summary_length,
            "mission_count": mission_count,
            "keyword_score": keyword_score,
            "title_text": title,
            "summary_text": summary,
        })
        label_counts[label] += 1

    return rows, dict(label_counts)


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_stats(rows: list[dict], label_counts: dict[str, int], path: str) -> None:
    total = len(rows)
    summary_lengths = [r["summary_length"] for r in rows]
    mission_counts = [r["mission_count"] for r in rows]
    source_ids = set(r["source_id"] for r in rows)
    source_kinds = set(r["source_kind"] for r in rows)
    url_domains = set(r["url_domain"] for r in rows)

    lines = [
        f"Dataset Statistics",
        f"==================",
        f"Total items: {total}",
        f"",
        f"Label distribution:",
        f"  gold (apply_now/queue_research): {label_counts.get('gold', 0)}",
        f"  soft (candidate_insight):        {label_counts.get('soft', 0)}",
        f"  negative (discard only):         {label_counts.get('negative', 0)}",
        f"",
        f"Binary framing (gold+soft vs negative):",
        f"  positive: {label_counts.get('gold', 0) + label_counts.get('soft', 0)}",
        f"  negative: {label_counts.get('negative', 0)}",
        f"",
        f"Features:",
        f"  Unique sourceIds: {len(source_ids)}",
        f"  Unique sourceKinds: {len(source_kinds)}",
        f"  Unique URL domains: {len(url_domains)}",
        f"  Summary length: min={min(summary_lengths)} mean={sum(summary_lengths)//total} max={max(summary_lengths)}",
        f"  Mission count: min={min(mission_counts)} mean={sum(mission_counts)/total:.1f} max={max(mission_counts)}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build scoring model dataset")
    parser.add_argument(
        "--signals-dir",
        default=DEFAULT_SIGNALS_DIR,
        help="Path to signals directory containing items.json and claims.json",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "dataset.csv")
    stats_path = os.path.join(out_dir, "dataset_stats.txt")

    print(f"Loading data from {args.signals_dir}...")
    rows, label_counts = build_dataset(args.signals_dir)

    print(f"Built dataset: {len(rows)} items")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    write_stats(rows, label_counts, stats_path)
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
