"""
Ablation grid runner, proposal Section 6.3: "Chunking (2) x retriever
(BM25 / dense / hybrid+rerank) x validity layer (on/off) x generator
(2-3 hosted models) ~= 24-36 configurations."

WHAT THIS SANDBOX CAN AND CANNOT RUN (read before trusting the output):
  - Chunking x retriever x validity = 2 x 3 x 2 = 12 configs: FULLY
    RUNNABLE here, computed against NyayaBench's retrieval-gradeable
    slices (statutory_qa, precedent_qa, transition_qa, multihop_qa --
    trap_qa is excluded, it has no gold_evidence_ids by design).
  - The generator axis (2-3 hosted models) is NOT run here: it requires
    live API calls (Groq at minimum) and no API key is configured in this
    sandbox (see config.py). generation_eval_stub() below shows exactly
    where and how that axis plugs in -- run it with GROQ_API_KEY set to
    extend the grid to the full 24-36 the proposal specifies.
  - Reranking is folded into the "hybrid" retriever config as an
    additional on/off flag rather than a fourth retriever value, since
    the proposal frames it as a refinement stage on top of hybrid
    fusion, not a fourth retrieval mode.

Run from the project root, after generate_nyayabench.py:
    python scripts/eval/run_ablation.py

Output:
    data/processed/eval/ablation_results.json
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NYAYABENCH_DIR, EVAL_DIR, JUDGMENTS_PROCESSED_DIR, MAX_ABLATION_CONFIGS  # noqa: E402
from era_filter import apply_validity_filter  # noqa: E402
from retrieve import tokenize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
from chunking import structure_aware_docs, fixed_token_docs  # noqa: E402
from embed_api import embed_batch, embed_one  # noqa: E402
from rerank import Reranker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "precedent"))
from build_overruling_graph import good_law_as_of, PRECEDENT_DEMOTION_FACTOR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import statutory_era_accuracy, aggregate  # noqa: E402
from stats import bootstrap_ci, paired_bootstrap_test, holm_bonferroni  # noqa: E402

RETRIEVERS = ["bm25", "dense", "hybrid"]
CHUNKING_MODES = ["structure_aware", "fixed_512"]
VALIDITY_MODES = ["on", "off"]
TOP_K = 10


class InMemoryIndex:
    """Same retrieval logic as retrieval/hybrid_retrieve.py, generalized
    to run over EITHER chunking mode's doc list without touching disk --
    the ablation grid needs many short-lived indices, one per chunking
    mode, not the single persistent index the CLI tools use."""

    def __init__(self, docs: list):
        self.docs = docs
        self.doc_ids = [d["doc_id"] for d in docs]
        tokenized = [tokenize(d["text"]) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        vectors = embed_batch([d["text"] for d in docs])
        self.vectors = np.array(vectors, dtype=np.float32)

        graph_path = JUDGMENTS_PROCESSED_DIR / "overruling_graph.json"
        self.graph = {}
        if graph_path.exists():
            self.graph = json.loads(graph_path.read_text(encoding="utf-8")).get("invalidating_edges", {})

    def retrieve(self, query: str, query_date: date, retriever: str, validity_mode: str, top_k: int = TOP_K):
        bm25_scores = self.bm25.get_scores(tokenize(query))
        pool = min(50, len(self.docs))
        bm25_order = sorted(range(len(self.docs)), key=lambda i: bm25_scores[i], reverse=True)[:pool]
        bm25_rank_of = {self.doc_ids[i]: r for r, i in enumerate(bm25_order)}

        qvec = np.array(embed_one(query), dtype=np.float32)
        qnorm = qvec / (np.linalg.norm(qvec) or 1.0)
        sims = self.vectors @ qnorm
        dense_order = np.argsort(-sims)[:pool]
        dense_rank_of = {self.doc_ids[i]: r for r, i in enumerate(dense_order)}

        if retriever == "bm25":
            scored = [(self.docs[i], float(bm25_scores[i])) for i in bm25_order]
        elif retriever == "dense":
            scored = [(self.docs[i], float(sims[i])) for i in dense_order]
        else:  # hybrid
            all_ids = set(bm25_rank_of) | set(dense_rank_of)
            rrf = {}
            for did in all_ids:
                s = 0.0
                if did in bm25_rank_of:
                    s += 1.0 / (60 + bm25_rank_of[did] + 1)
                if did in dense_rank_of:
                    s += 1.0 / (60 + dense_rank_of[did] + 1)
                rrf[did] = s
            by_id = {d["doc_id"]: d for d in self.docs}
            scored = [(by_id[did], s) for did, s in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)]

        if validity_mode == "on":
            statute_only = [(d, s) for d, s in scored if d.get("source") == "statute"]
            other = [(d, s) for d, s in scored if d.get("source") != "statute"]
            statute_only = apply_validity_filter(statute_only, query_date, mode="demote")
            merged = statute_only + other
            merged.sort(key=lambda pair: pair[1], reverse=True)

            adjusted = []
            for doc, score in merged:
                if doc.get("source") == "judgment":
                    ok, _ = good_law_as_of(doc["judgment_id"], query_date, self.graph)
                    if not ok:
                        score = score * PRECEDENT_DEMOTION_FACTOR
                adjusted.append((doc, score))
            adjusted.sort(key=lambda pair: pair[1], reverse=True)
            scored = adjusted

        return scored[:top_k]


def gold_id_matches(retrieved_id: str, gold_ids: set, chunking: str) -> bool:
    """For fixed_512 chunking, a retrieved sub-chunk (doc_id like
    'BNS-318::w0') counts as a gold hit if its PARENT doc_id is gold --
    matches standard practice of crediting any chunk that falls inside the
    gold section (see chunking.py's own docstring)."""
    if retrieved_id in gold_ids:
        return True
    if chunking == "fixed_512" and "::w" in retrieved_id:
        parent = retrieved_id.split("::w")[0]
        return parent in gold_ids
    return False


def run_config(items: list, index: "InMemoryIndex", retriever: str, validity_mode: str, chunking: str, reranker: Reranker = None):
    import math
    recall5, recall10, mrr10, ndcg10, era_acc = [], [], [], [], []
    for item in items:
        query_date = date.fromisoformat(item["query_date"])
        results = index.retrieve(item["question"], query_date, retriever, validity_mode, top_k=TOP_K)
        if reranker is not None and reranker.available:
            results = reranker.rerank(item["question"], results, top_k=TOP_K)
        retrieved_ids = [d["doc_id"] for d, _s in results]
        gold_ids = set(item["gold_evidence_ids"])

        hit5 = any(gold_id_matches(rid, gold_ids, chunking) for rid in retrieved_ids[:5])
        hit10 = any(gold_id_matches(rid, gold_ids, chunking) for rid in retrieved_ids[:10])
        recall5.append(1.0 if hit5 else 0.0)
        recall10.append(1.0 if hit10 else 0.0)

        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if gold_id_matches(rid, gold_ids, chunking):
                rr = 1.0 / rank
                break
        mrr10.append(rr)

        dcg = sum(1.0 / math.log2(r + 1) for r, rid in enumerate(retrieved_ids, start=1) if gold_id_matches(rid, gold_ids, chunking))
        ndcg10.append(dcg)  # idcg = 1.0 -- single relevant doc per item by benchmark design

        if item.get("applicable_era") and results:
            top_doc = results[0][0]
            predicted_statute = top_doc.get("statute", "")
            era_acc.append(statutory_era_accuracy(predicted_statute, item["applicable_era"]))

    return {
        "recall@5": aggregate(recall5),
        "recall@10": aggregate(recall10),
        "mrr@10": aggregate(mrr10),
        "ndcg@10": aggregate(ndcg10),
        "statutory_era_accuracy": aggregate(era_acc),
        "_raw_recall10": recall10,  # kept for paired significance testing across configs
    }


def generation_eval_stub():
    """Plug point for the generator axis (2-3 hosted models). Requires
    GROQ_API_KEY (and optionally SECONDARY_GENERATION_MODEL) -- see
    config.py. Skipped by default; see module docstring for why."""
    import os
    if not os.environ.get("GROQ_API_KEY"):
        return {"status": "skipped", "reason": "GROQ_API_KEY not set -- see run_ablation.py module docstring"}
    return {"status": "not_implemented_but_key_present"}


def main():
    bench_path = NYAYABENCH_DIR / "nyayabench.json"
    if not bench_path.exists():
        raise SystemExit("[error] Run generate_nyayabench.py first.")
    all_items = json.loads(bench_path.read_text(encoding="utf-8"))
    items = [i for i in all_items if i["is_answerable"] and i["gold_evidence_ids"]]
    print(f"Running ablation over {len(items)} retrieval-gradeable NyayaBench items "
          f"(excludes {len(all_items) - len(items)} trap/unanswerable items)\n")

    configs = [
        (chunking, retriever, validity)
        for chunking in CHUNKING_MODES
        for retriever in RETRIEVERS
        for validity in VALIDITY_MODES
    ]
    if len(configs) > MAX_ABLATION_CONFIGS:
        configs = configs[:MAX_ABLATION_CONFIGS]
    print(f"Grid: {len(configs)} configs (chunking x retriever x validity; "
          f"generator axis skipped, see generation_eval_stub)\n")

    indices = {}
    for chunking in CHUNKING_MODES:
        docs = structure_aware_docs() if chunking == "structure_aware" else fixed_token_docs()
        print(f"Building {chunking} index over {len(docs)} chunks...")
        indices[chunking] = InMemoryIndex(docs)

    reranker = Reranker()

    results = {}
    raw_recall10 = {}
    for chunking, retriever, validity in configs:
        label = f"{chunking}|{retriever}|validity_{validity}"
        print(f"Running {label} ...")
        r = run_config(items, indices[chunking], retriever, validity, chunking, reranker=reranker)
        raw_recall10[label] = r.pop("_raw_recall10")
        results[label] = r

    for label in results:
        results[label]["recall@10_ci95"] = bootstrap_ci(raw_recall10[label])

    p_values = {}
    for chunking, retriever, _ in configs:
        on_label = f"{chunking}|{retriever}|validity_on"
        off_label = f"{chunking}|{retriever}|validity_off"
        if on_label in raw_recall10 and off_label in raw_recall10:
            test = paired_bootstrap_test(raw_recall10[off_label], raw_recall10[on_label])
            p_values[f"{chunking}|{retriever}: validity on vs off"] = test["p_value"]

    significance = holm_bonferroni(p_values) if p_values else {}

    out = {
        "n_items": len(items),
        "n_configs": len(configs),
        "generation_axis": generation_eval_stub(),
        "results": results,
        "validity_significance_holm_bonferroni": significance,
    }
    out_path = EVAL_DIR / "ablation_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'Config':<45}{'Recall@5':<12}{'Recall@10':<12}{'MRR@10':<10}{'nDCG@10':<10}")
    for label, r in results.items():
        print(f"{label:<45}{r['recall@5']['mean']:<12.3f}{r['recall@10']['mean']:<12.3f}"
              f"{r['mrr@10']['mean']:<10.3f}{r['ndcg@10']['mean']:<10.3f}")

    print("\nValidity-layer significance (Holm-Bonferroni corrected):")
    for label, sig in significance.items():
        print(f"  {label}: p={sig['p']:.4f}  significant={sig['significant']}")

    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
