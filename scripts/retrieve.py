"""
Retrieval: BM25 over the combined corpus, with naive and validity-aware
modes. This is the ONLY thing that differs between the two modes -- same
index, same query, same top-k candidate pool. The validity layer
(era_filter.py) is applied only in "validity_aware" mode.

Run standalone for a quick manual test:
    python scripts/retrieve.py "punishment for murder" 2025-01-01

Used as a module by run_baseline_vs_validity.py for the full evaluation.
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

from rank_bm25 import BM25Okapi

from era_filter import apply_validity_filter

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TOP_K = 10

# Standard BM25 practice: filter common stopwords before scoring. Without
# this, generic query phrasing (e.g. "What is the applicable law for...")
# lets filler words with coincidentally high IDF (rare-but-irrelevant across
# the corpus) dominate over the actual content words -- confirmed this was
# happening: "applicable"/"for"/"law" were outscoring "murder" itself.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "this", "that", "these", "those", "for", "of", "to", "in", "on",
    "at", "by", "with", "and", "or", "but", "if", "then", "so", "as", "it",
    "its", "be", "been", "being", "do", "does", "did", "has", "have", "had",
    "applicable", "law",
}


def tokenize(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever:
    def __init__(self):
        corpus_path = PROCESSED_DIR / "corpus.json"
        if not corpus_path.exists():
            raise SystemExit("[error] Run build_index.py first.")
        self.corpus = json.loads(corpus_path.read_text())
        tokenized = [tokenize(f"{d['marginal_note']} {d['text']}") for d in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, query_date: date, mode: str = "validity_aware", top_k: int = TOP_K):
        """mode: 'naive' (no date awareness) or 'validity_aware' (demotes wrong-era docs)."""
        scores = self.bm25.get_scores(tokenize(query))
        scored_docs = list(zip(self.corpus, scores))
        scored_docs.sort(key=lambda pair: pair[1], reverse=True)
        # Wide FIXED pool before era adjustment, not proportional to top_k --
        # if top_k=1 and the pool were only e.g. 3 candidates, the era filter
        # could have nothing correct-era left to promote even when one exists
        # further down the raw ranking. Confirmed this was silently breaking
        # validity-aware accuracy when called with top_k=1 (the eval script's
        # exact use case).
        POOL_SIZE = 50
        scored_docs = scored_docs[:POOL_SIZE]

        filter_mode = "off" if mode == "naive" else "demote"
        adjusted = apply_validity_filter(scored_docs, query_date, mode=filter_mode)
        return adjusted[:top_k]


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/retrieve.py \"<query>\" <YYYY-MM-DD> [naive|validity_aware]")
        sys.exit(1)

    query = sys.argv[1]
    query_date = date.fromisoformat(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "validity_aware"

    retriever = Retriever()
    results = retriever.retrieve(query, query_date, mode=mode, top_k=5)

    print(f"Query: {query!r}  Date: {query_date}  Mode: {mode}\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [{doc['statute']} {doc['section_no']}] score={score:.2f}")
        print(f"   {doc['text'][:150]}")


if __name__ == "__main__":
    main()
