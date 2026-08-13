"""
Metric suite, proposal Section 6.1 -- the metrics that are pure
computation (no GPU, no API call required to COMPUTE them, though
Statutory Era Accuracy / Precedent Validity Rate need retrieval RESULTS
to compute over, which is where the API cost lives upstream).

Metrics implemented here:
  - Recall@k, MRR@10, nDCG@10          -- standard IR, against gold_evidence_ids
  - Statutory Era Accuracy (novel)      -- % answers citing era-correct provision
  - Precedent Validity Rate (novel)     -- % cited judgments good-law for the query date
  - Hallucination / trap-set metrics    -- string-match against the corpus index, per
                                            Section 6.1's "hard, objective, CPU-only check"

Deliberately NOT implemented here (need a live LLM-judge or human raters,
out of scope for this module -- see eval/citation_faithfulness.py for the
one exception that gets a CPU-only proxy):
  RAGAS/ARES-style faithfulness, LLM-judge answer quality, Krippendorff's
  alpha expert agreement.
"""
import math
from datetime import date


def recall_at_k(retrieved_ids: list, gold_ids: set, k: int) -> float:
    if not gold_ids:
        return float("nan")  # undefined for trap items -- caller should exclude these, not average them in as 0
    top_k = set(retrieved_ids[:k])
    return len(top_k & gold_ids) / len(gold_ids)


def mrr_at_k(retrieved_ids: list, gold_ids: set, k: int = 10) -> float:
    if not gold_ids:
        return float("nan")
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, gold_ids: set, k: int = 10) -> float:
    if not gold_ids:
        return float("nan")
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in gold_ids else 0.0
        dcg += rel / math.log2(rank + 1)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else float("nan")


def statutory_era_accuracy(predicted_statute: str, applicable_era: str) -> float:
    """1.0 if the retrieved/cited top document's statute matches the
    gold applicable_era for the query date, else 0.0. NaN if the item has
    no applicable_era (precedent/trap/multihop items where this metric
    doesn't apply)."""
    if not applicable_era:
        return float("nan")
    return 1.0 if predicted_statute == applicable_era else 0.0


def precedent_validity_rate(cited_judgment_ids: list, query_date: date, graph: dict) -> float:
    """% of cited judgments that are good law as of query_date. NaN if no
    judgments were cited (item has no precedent component)."""
    if not cited_judgment_ids:
        return float("nan")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "precedent"))
    from build_overruling_graph import good_law_as_of  # noqa: E402

    good = sum(1 for jid in cited_judgment_ids if good_law_as_of(jid, query_date, graph)[0])
    return good / len(cited_judgment_ids)


def fabricated_citation_rate(cited_doc_ids: list, valid_doc_ids: set) -> float:
    """% of cited doc_ids that do NOT exist in the real corpus index --
    the objective, string-match hallucination check (Section 6.1).
    0.0 = no fabrication. NaN if nothing was cited."""
    if not cited_doc_ids:
        return float("nan")
    fabricated = sum(1 for did in cited_doc_ids if did not in valid_doc_ids)
    return fabricated / len(cited_doc_ids)


def trap_false_answer_rate(trap_results: list) -> float:
    """trap_results: list of bool, True if the system answered instead of
    abstaining on a trap_qa item (wrong behavior). Returns the fraction
    that got it wrong -- lower is better, 0.0 is perfect abstention."""
    if not trap_results:
        return float("nan")
    return sum(trap_results) / len(trap_results)


def aggregate(per_item_scores: list) -> dict:
    """Mean +/- basic spread over a list of floats, skipping NaN (items
    the metric doesn't apply to) rather than treating them as zero, which
    would silently penalize a system for questions the metric was never
    meant to score."""
    valid = [s for s in per_item_scores if not math.isnan(s)]
    if not valid:
        return {"mean": float("nan"), "n": 0, "n_excluded_nan": len(per_item_scores)}
    mean = sum(valid) / len(valid)
    variance = sum((v - mean) ** 2 for v in valid) / len(valid) if len(valid) > 1 else 0.0
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "n": len(valid),
        "n_excluded_nan": len(per_item_scores) - len(valid),
    }
