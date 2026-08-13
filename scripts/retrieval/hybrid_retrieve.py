"""
Hybrid retrieval over the combined statute+judgment index: BM25 (sparse) +
dense (API embeddings) fused with Reciprocal Rank Fusion, per proposal
4.2. This is the retrieval layer NyayaBench's "retriever" ablation axis
(bm25 / dense / hybrid) selects between -- see eval/run_ablation.py.

BOTH validity axes are applied here, composed independently:
  - statute docs -> era_filter.apply_validity_filter (unchanged, existing)
  - judgment docs -> overruling-graph demotion (new, this module)
This mirrors the original retrieve.py's naive/validity_aware split, with
one addition: validity-aware mode now also demotes bad-law precedent, not
just wrong-era statutes.

Run standalone for a quick manual test:
    python scripts/retrieval/hybrid_retrieve.py "punishment for cheating" 2025-01-01
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, JUDGMENTS_PROCESSED_DIR  # noqa: E402
from era_filter import apply_validity_filter, valid_statute_for_date  # noqa: E402
from retrieve import tokenize  # noqa: E402 -- reuse the exact same tokenizer/stopword list as the statute-only pipeline, so BM25 behaves identically on statute text in both pipelines
from embed_api import embed_one  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "precedent"))
from build_overruling_graph import good_law_as_of, PRECEDENT_DEMOTION_FACTOR  # noqa: E402

RRF_K = 60  # standard RRF constant (Cormack et al. 2009 default); not tuned, since tuning it is not one of this proposal's ablation axes
POOL_SIZE = 50  # candidate pool before validity adjustment -- same reasoning as retrieve.py: must be a fixed size independent of top_k, or the validity layer can run out of same-validity candidates to promote


class HybridRetriever:
    def __init__(self):
        docs_path = DATA_PROCESSED / "dense_docs.json"
        npz_path = DATA_PROCESSED / "dense_index.npz"
        graph_path = JUDGMENTS_PROCESSED_DIR / "overruling_graph.json"
        if not docs_path.exists() or not npz_path.exists():
            raise SystemExit("[error] Run build_dense_index.py first.")

        self.docs = json.loads(docs_path.read_text(encoding="utf-8"))
        self.doc_by_id = {d["doc_id"]: d for d in self.docs}

        npz = np.load(npz_path)
        self.vectors = npz["vectors"]
        self.doc_ids = [str(x) for x in npz["doc_ids"]]
        assert self.doc_ids == [d["doc_id"] for d in self.docs], \
            "dense_index.npz and dense_docs.json are out of sync -- rebuild both together"
        self.index_of = {did: i for i, did in enumerate(self.doc_ids)}

        tokenized = [tokenize(d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

        self.graph = {}
        if graph_path.exists():
            self.graph = json.loads(graph_path.read_text(encoding="utf-8")).get("invalidating_edges", {})

    def _bm25_ranked(self):
        return self.bm25

    def _dense_ranked(self, query: str):
        qvec = np.array(embed_one(query), dtype=np.float32)
        # vectors are L2-normalized at embed time (see embed_api._offline_embed;
        # real providers' embeddings are also normalized here for consistency)
        # so a plain dot product IS cosine similarity -- no faiss dependency
        # required for exact search at this corpus size (see module docstring
        # of build_dense_index.py for the scale argument).
        qnorm = qvec / (np.linalg.norm(qvec) or 1.0)
        sims = self.vectors @ qnorm
        order = np.argsort(-sims)
        return [(self.doc_ids[i], float(sims[i])) for i in order]

    def _apply_precedent_validity(self, scored_docs, query_date: date, mode: str):
        if mode == "off":
            return scored_docs
        adjusted = []
        for doc, score in scored_docs:
            if doc["source"] == "judgment":
                ok, _reason = good_law_as_of(doc["judgment_id"], query_date, self.graph)
                if not ok:
                    score = score * PRECEDENT_DEMOTION_FACTOR
            adjusted.append((doc, score))
        adjusted.sort(key=lambda pair: pair[1], reverse=True)
        return adjusted

    def _apply_statute_validity(self, scored_docs, query_date: date, mode: str):
        if mode == "off":
            return scored_docs
        # era_filter.apply_validity_filter expects dicts with a "statute"
        # key, present on statute docs but not judgment docs -- judgment
        # docs pass through untouched (their score is compared against
        # each other and against statutes in the SAME units, so this is
        # safe: only statute docs get demoted, judgment docs are handled
        # by _apply_precedent_validity instead).
        statute_only = [(d, s) for d, s in scored_docs if d["source"] == "statute"]
        other = [(d, s) for d, s in scored_docs if d["source"] != "statute"]
        adjusted_statutes = apply_validity_filter(statute_only, query_date, mode="demote")
        merged = adjusted_statutes + other
        merged.sort(key=lambda pair: pair[1], reverse=True)
        return merged

    def retrieve(self, query: str, query_date: date, retriever: str = "hybrid",
                 validity_mode: str = "on", top_k: int = 10):
        """retriever: 'bm25' | 'dense' | 'hybrid'. validity_mode: 'on' | 'off'."""
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranked = sorted(range(len(self.docs)), key=lambda i: bm25_scores[i], reverse=True)[:POOL_SIZE]
        bm25_rank_of = {self.doc_ids[i]: rank for rank, i in enumerate(bm25_ranked)}

        dense_ranked = self._dense_ranked(query)[:POOL_SIZE]
        dense_rank_of = {doc_id: rank for rank, (doc_id, _score) in enumerate(dense_ranked)}

        if retriever == "bm25":
            scored = [(self.doc_by_id[self.doc_ids[i]], float(bm25_scores[i])) for i in bm25_ranked]
        elif retriever == "dense":
            scored = [(self.doc_by_id[did], score) for did, score in dense_ranked]
        elif retriever == "hybrid":
            all_ids = set(bm25_rank_of) | set(dense_rank_of)
            rrf_scores = {}
            for did in all_ids:
                s = 0.0
                if did in bm25_rank_of:
                    s += 1.0 / (RRF_K + bm25_rank_of[did] + 1)
                if did in dense_rank_of:
                    s += 1.0 / (RRF_K + dense_rank_of[did] + 1)
                rrf_scores[did] = s
            scored = [(self.doc_by_id[did], s) for did, s in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)]
        else:
            raise ValueError(f"Unknown retriever: {retriever!r}")

        scored = self._apply_statute_validity(scored, query_date, "on" if validity_mode == "on" else "off")
        scored = self._apply_precedent_validity(scored, query_date, "on" if validity_mode == "on" else "off")

        return scored[:top_k]


def _describe(doc: dict) -> str:
    if doc["source"] == "statute":
        return f"[{doc['statute']} {doc['section_no']}]"
    return f"[{doc['case_name']} / {doc['rhetorical_unit']}]"


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/retrieval/hybrid_retrieve.py \"<query>\" <YYYY-MM-DD> [bm25|dense|hybrid] [on|off]")
        sys.exit(1)

    query = sys.argv[1]
    query_date = date.fromisoformat(sys.argv[2])
    retriever = sys.argv[3] if len(sys.argv) > 3 else "hybrid"
    validity_mode = sys.argv[4] if len(sys.argv) > 4 else "on"

    hr = HybridRetriever()
    results = hr.retrieve(query, query_date, retriever=retriever, validity_mode=validity_mode, top_k=5)

    print(f"Query: {query!r}  Date: {query_date}  Retriever: {retriever}  Validity: {validity_mode}\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {_describe(doc)} score={score:.4f}")
        print(f"   {doc['text'][:150]}")


if __name__ == "__main__":
    main()
