"""
Reranking stage: a small CPU-runnable cross-encoder over the top
RERANK_POOL_SIZE candidates from hybrid retrieval, per proposal 4.2(b):
"a small CPU-runnable cross-encoder (e.g. ms-marco-MiniLM-L-6) restricted
to the top-20 candidates per query, which is tractable on CPU at
benchmark scale."

Applied AFTER the validity layer, not before -- reranking should refine
the ordering among already-validity-adjusted candidates, not fight the
validity demotion by pulling a bad-law precedent back up on pure semantic
relevance. (An ablation with rerank BEFORE validity would be a legitimate
alternative design to test later; this pipeline picks after-validity as
the default and states it explicitly rather than leaving the order
implicit.)

Falls back to a no-op (returns input order unchanged) if
sentence-transformers isn't installed, so the rest of the pipeline never
hard-fails on this optional stage -- callers should check
`reranker.available` if they need to know whether reranking actually ran
(e.g. for the ablation grid, which needs to record it as an honest "off"
rather than silently double-counting a no-op run as "on").

Run standalone for a quick manual test:
    python scripts/retrieval/rerank.py "punishment for cheating" 2023-01-01
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RERANK_MODE, RERANK_POOL_SIZE  # noqa: E402

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, mode: str = RERANK_MODE):
        self.mode = mode
        self.available = False
        self._model = None
        if mode == "cross_encoder" and _CROSS_ENCODER_AVAILABLE:
            try:
                self._model = CrossEncoder(MODEL_NAME)
                self.available = True
            except Exception as e:  # model download can fail offline -- degrade, don't crash the whole pipeline
                print(f"[warn] Could not load cross-encoder ({e}); reranking disabled for this run.")
        elif mode == "cross_encoder" and not _CROSS_ENCODER_AVAILABLE:
            print("[warn] sentence-transformers not installed; reranking disabled. "
                  "pip install sentence-transformers to enable.")

    def rerank(self, query: str, scored_docs: list, top_k: int = 10):
        """scored_docs: list of (doc, score) tuples, already validity-
        adjusted and sorted by hybrid_retrieve.py. Reranks only the top
        RERANK_POOL_SIZE of them (per proposal 4.2(b)); anything beyond
        that pool keeps its incoming relative order, appended after."""
        if not self.available or self.mode == "off":
            return scored_docs[:top_k]

        pool = scored_docs[:RERANK_POOL_SIZE]
        rest = scored_docs[RERANK_POOL_SIZE:]

        pairs = [(query, doc["text"]) for doc, _score in pool]
        ce_scores = self._model.predict(pairs)

        reranked_pool = sorted(zip([d for d, _ in pool], ce_scores), key=lambda x: x[1], reverse=True)
        return (reranked_pool + rest)[:top_k]


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/retrieval/rerank.py \"<query>\" <YYYY-MM-DD>")
        sys.exit(1)

    from hybrid_retrieve import HybridRetriever, _describe

    query = sys.argv[1]
    query_date = date.fromisoformat(sys.argv[2])

    hr = HybridRetriever()
    candidates = hr.retrieve(query, query_date, retriever="hybrid", validity_mode="on", top_k=RERANK_POOL_SIZE)

    reranker = Reranker()
    print(f"Cross-encoder available: {reranker.available}\n")
    results = reranker.rerank(query, candidates, top_k=5)

    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {_describe(doc)} score={score:.4f}")
        print(f"   {doc['text'][:150]}")


if __name__ == "__main__":
    main()
