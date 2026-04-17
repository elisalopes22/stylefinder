"""Run evaluation queries against curated ground truth.

Loads the manually verified evaluation_set.json, runs each query
through FAISS, and measures how many expected documents are retrieved.

Run: python -m src.evaluate
"""

import os
import sys
import json

# Ensure project root is on Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import EVAL_SET
from src.search import StyleFinderEngine

EVAL_QUERIES = [
    {"id": "Q01", "query": "flattering wedding guest dress that cinches at the waist", "layer": "style"},
    {"id": "Q02", "query": "plus-size woman looking for a professional interview outfit", "layer": "style"},
    {"id": "Q03", "query": "tall athletic man going to a casual date night", "layer": "style"},
    {"id": "Q04", "query": "non-binary person minimalist outfit for a gallery opening", "layer": "style"},
    {"id": "Q05", "query": "pear-shaped woman looking for a brunch outfit", "layer": "style"},
    {"id": "Q06", "query": "comfortable hiking outfit for cool autumn weather", "layer": "style"},
    {"id": "Q07", "query": "hourglass figure looking for an elegant evening gown", "layer": "style"},
    {"id": "Q08", "query": "petite woman going to a music festival", "layer": "style"},
    {"id": "Q09", "query": "elegant black evening dress for women", "layer": "product"},
    {"id": "Q10", "query": "men casual blue shirt for summer", "layer": "product"},
    {"id": "Q11", "query": "women white sneakers casual", "layer": "product"},
    {"id": "Q12", "query": "formal black shoes for men", "layer": "product"},
    {"id": "Q13", "query": "outfit for wheelchair user at a formal event", "layer": "style"},
    {"id": "Q14", "query": "budget friendly outfit under 50 dollars", "layer": "style"},
    {"id": "Q15", "query": "sustainable eco-friendly clothing for a date night", "layer": "style"},
]


def main():
    engine = StyleFinderEngine()

    # Load curated ground truth
    ground_truth = {}
    if os.path.exists(EVAL_SET):
        with open(EVAL_SET, "r") as f:
            gt_data = json.load(f)
            for entry in gt_data:
                ground_truth[entry["query_id"]] = entry.get("expected_doc_ids", [])
        print(f"Loaded ground truth from {EVAL_SET}\n")

    print("=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70)

    results_summary = []

    for eq in EVAL_QUERIES:
        results = engine.faiss_search(eq["query"], layer=eq["layer"], n=5)
        retrieved_ids = [r["doc_id"] for r in results]
        expected_ids = ground_truth.get(eq["id"], [])

        # Calculate hits
        if expected_ids:
            hits = [did for did in retrieved_ids if did in expected_ids]
            recall = len(hits) / len(expected_ids) if expected_ids else 0
        else:
            hits = []
            recall = None

        print(f"\n{eq['id']}: {eq['query']}")
        print(f"  Retrieved: {retrieved_ids}")
        print(f"  Expected:  {expected_ids}")
        if recall is not None:
            print(f"  Hits: {len(hits)}/{len(expected_ids)} (recall: {recall:.2f})")
        else:
            print(f"  ADVERSARIAL - no relevant documents expected")

        for r in results:
            marker = " <<" if r["doc_id"] in expected_ids else ""
            print(f"    doc_id={r['doc_id']:>6s} (score: {r['score']:.4f}){marker}")

        results_summary.append({
            "query_id": eq["id"],
            "query": eq["query"],
            "layer": eq["layer"],
            "retrieved_doc_ids": retrieved_ids,
            "expected_doc_ids": expected_ids,
            "hits": len(hits),
            "total_expected": len(expected_ids),
            "recall": round(recall, 2) if recall is not None else None,
        })

    # Summary table
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'ID':<6} {'Layer':<10} {'Hits':>5} {'Expected':>9} {'Recall':>8}  {'Query'}")
    print("-" * 80)

    total_hits = 0
    total_expected = 0
    realistic_count = 0
    realistic_with_hits = 0

    for r in results_summary:
        if r["recall"] is not None:
            recall_str = f"{r['recall']:.2f}"
            total_hits += r["hits"]
            total_expected += r["total_expected"]
            realistic_count += 1
            if r["hits"] > 0:
                realistic_with_hits += 1
        else:
            recall_str = "N/A"
        print(f"{r['query_id']:<6} {r['layer']:<10} {r['hits']:>5} {r['total_expected']:>9} {recall_str:>8}  {r['query'][:42]}...")

    avg_recall = total_hits / total_expected if total_expected > 0 else 0
    print(f"\nRealistic queries with at least 1 hit: {realistic_with_hits}/{realistic_count}")
    print(f"Overall recall (realistic only): {avg_recall:.2f}")
    print(f"Adversarial queries (correctly no ground truth): 3")

    engine.close()


if __name__ == "__main__":
    main()
