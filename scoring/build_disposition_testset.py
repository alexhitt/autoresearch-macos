"""
Build ground truth test set for disposition prompt optimization.

Extracts claims with human-verified applicationStatus labels, grouped by
source item with abstracts. Used by evaluate_disposition.py.

Ground truth label mapping:
  applied / implemented -> apply_now (human acted on it)
  skipped -> not_apply_now (human saw it, chose not to act)

Usage: python scoring/build_disposition_testset.py [--signals-dir PATH]
"""

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIGNALS_DIR = os.path.join(SCRIPT_DIR, "disposition_data")
TESTSET_PATH = os.path.join(SCRIPT_DIR, "disposition_testset.json")


def main():
    parser = argparse.ArgumentParser(description="Build disposition test set")
    parser.add_argument(
        "--signals-dir",
        default=DEFAULT_SIGNALS_DIR,
        help="Path to directory with claims.json and items.json",
    )
    parser.add_argument(
        "--output",
        default=TESTSET_PATH,
    )
    args = parser.parse_args()

    claims = json.load(open(os.path.join(args.signals_dir, "claims.json")))
    items = json.load(open(os.path.join(args.signals_dir, "items.json")))
    items_by_id = {i["id"]: i for i in items}

    # Filter to ground truth claims (have applicationStatus)
    gt_claims = [c for c in claims if c.get("applicationStatus")]

    # Group by source item
    by_item = {}
    for c in gt_claims:
        by_item.setdefault(c["signalItemId"], []).append(c)

    # Build test set entries
    test_items = []
    total_apply_now = 0
    total_not_apply_now = 0

    for item_id, item_claims in by_item.items():
        item = items_by_id.get(item_id)
        if not item:
            print(f"  WARNING: item {item_id} not found in items.json, skipping")
            continue

        test_claims = []
        for c in item_claims:
            status = c["applicationStatus"]
            # applied/implemented = truly actionable, skipped = not
            ground_truth = "apply_now" if status in ("applied", "implemented") else "not_apply_now"
            if ground_truth == "apply_now":
                total_apply_now += 1
            else:
                total_not_apply_now += 1

            test_claims.append({
                "id": c["id"],
                "claim": c["claim"],
                "category": c.get("category", "unknown"),
                "original_disposition": c.get("disposition"),
                "application_status": status,
                "ground_truth": ground_truth,
            })

        test_items.append({
            "item_id": item_id,
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "claims": test_claims,
        })

    # Sort by item_id for deterministic ordering
    test_items.sort(key=lambda x: x["item_id"])

    output = {
        "version": 1,
        "total_claims": sum(len(t["claims"]) for t in test_items),
        "total_items": len(test_items),
        "label_distribution": {
            "apply_now": total_apply_now,
            "not_apply_now": total_not_apply_now,
        },
        "items": test_items,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Built disposition test set:")
    print(f"  Items: {len(test_items)}")
    print(f"  Claims: {output['total_claims']}")
    print(f"  apply_now: {total_apply_now}")
    print(f"  not_apply_now: {total_not_apply_now}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
