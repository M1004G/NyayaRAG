"""
Run the full QA set through naive vs. validity-aware retrieval and compute
Statutory Era Accuracy for each -- this produces your first real number.

Statutory Era Accuracy = % of queries where the TOP-1 retrieved section
exactly matches the gold section for that query's date. Computed by exact
string match against the gold_section_no field generate_qa.py already
derived mechanically from the trusted mapping table -- no manual judgment
involved on either side of this comparison.

Run from the project root, after build_index.py:
    python scripts/run_baseline_vs_validity.py

Output printed to console, and saved to:
    data/processed/eval_results.json
"""
import json
from datetime import date
from pathlib import Path

from retrieve import Retriever

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def normalize(s: str) -> str:
    return s.replace(" ", "").upper().split("(")[0]


def main():
    qa_items = json.loads((PROCESSED_DIR / "qa_items.json").read_text())
    retriever = Retriever()

    results = []
    for item in qa_items:
        query_date = date.fromisoformat(item["query_date"])
        gold = normalize(item["gold_section_no"])
        gold_statute = item["gold_statute"]

        naive_top = retriever.retrieve(item["query"], query_date, mode="naive", top_k=1)
        validity_top = retriever.retrieve(item["query"], query_date, mode="validity_aware", top_k=1)

        naive_doc = naive_top[0][0] if naive_top else None
        validity_doc = validity_top[0][0] if validity_top else None

        naive_correct = bool(naive_doc) and normalize(naive_doc["section_no"]) == gold and naive_doc["statute"] == gold_statute
        validity_correct = bool(validity_doc) and normalize(validity_doc["section_no"]) == gold and validity_doc["statute"] == gold_statute

        results.append({
            "id": item["id"],
            "query": item["query"],
            "query_date": item["query_date"],
            "gold": f"{gold_statute} {item['gold_section_no']}",
            "naive_top1": f"{naive_doc['statute']} {naive_doc['section_no']}" if naive_doc else None,
            "naive_correct": naive_correct,
            "validity_top1": f"{validity_doc['statute']} {validity_doc['section_no']}" if validity_doc else None,
            "validity_correct": validity_correct,
        })

    n = len(results)
    naive_acc = sum(r["naive_correct"] for r in results) / n
    validity_acc = sum(r["validity_correct"] for r in results) / n

    print(f"Total QA items: {n}")
    print(f"Naive (no date awareness)     Statutory Era Accuracy: {naive_acc:.1%}")
    print(f"Validity-aware (era-filtered) Statutory Era Accuracy: {validity_acc:.1%}")
    print(f"Delta: {validity_acc - naive_acc:+.1%}")

    (PROCESSED_DIR / "eval_results.json").write_text(
        json.dumps({
            "n_items": n,
            "naive_accuracy": naive_acc,
            "validity_aware_accuracy": validity_acc,
            "per_item_results": results,
        }, indent=2, ensure_ascii=False)
    )
    print(f"\nFull per-item results -> data/processed/eval_results.json")


if __name__ == "__main__":
    main()
