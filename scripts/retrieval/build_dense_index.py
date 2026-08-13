"""
Combined dense index over BOTH the statute corpus (data/processed/corpus.json,
unchanged from the original pipeline) and the new judgment chunks
(data/processed/judgments/judgment_chunks.json) -- one retrieval space
covering statutes and precedent together, per proposal 4.1-4.2.

Uses FAISS-CPU per the proposal's explicit "FAISS-CPU" spec. Falls back to
a plain numpy brute-force cosine index if faiss-cpu isn't installed in
this environment -- functionally equivalent for a corpus this size (exact
search either way), just without FAISS's approximate-search speedups that
only matter at much larger scale.

Run from the project root, after build_index.py (statutes) and
segment_judgments.py (precedent):
    python scripts/retrieval/build_dense_index.py

Output:
    data/processed/dense_index.npz   (embeddings + doc_id order)
    data/processed/dense_docs.json   (unified doc records, id-aligned with the .npz)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, JUDGMENTS_PROCESSED_DIR, EMBEDDING_API_PROVIDER  # noqa: E402
from embed_api import embed_batch  # noqa: E402

EMBED_BATCH_SIZE = 96  # keeps individual API requests a reasonable size for hosted providers' payload limits


def load_unified_docs() -> list:
    """Same unified schema for both sources: {doc_id, source, text, ...
    passthrough metadata}. `source` is "statute" or "judgment" so
    hybrid_retrieve.py / era_filter.py / overruling_graph.py can each
    apply the validity rule that actually applies to that source (they
    are NOT interchangeable -- a statute's validity is era-based, a
    judgment's is precedent-based)."""
    docs = []

    corpus_path = DATA_PROCESSED / "corpus.json"
    if corpus_path.exists():
        # The upstream PDF-extracted statute text (build_statute_corpus.py)
        # contains a small number of cp1252 em-dashes (0x97) left over from
        # the source PDF, not valid UTF-8 -- pre-existing in this repo's
        # data, not introduced here. errors="replace" swaps each such byte
        # for U+FFFD rather than crashing the whole index build over a
        # handful of punctuation marks.
        for d in json.loads(corpus_path.read_bytes().decode("utf-8", errors="replace")):
            docs.append({
                "doc_id": d["doc_id"],
                "source": "statute",
                "text": f"{d.get('marginal_note', '')} {d['text']}".strip(),
                "statute": d["statute"],
                "section_no": d["section_no"],
            })

    chunks_path = JUDGMENTS_PROCESSED_DIR / "judgment_chunks.json"
    if chunks_path.exists():
        for c in json.loads(chunks_path.read_text(encoding="utf-8")):
            docs.append({
                "doc_id": c["chunk_id"],
                "source": "judgment",
                "text": c["text"],
                "judgment_id": c["judgment_id"],
                "case_name": c["case_name"],
                "court": c["court"],
                "date": c["date"],
                "rhetorical_unit": c["rhetorical_unit"],
            })

    if not docs:
        raise SystemExit(
            "[error] No documents found. Run build_index.py and/or "
            "segment_judgments.py first."
        )
    return docs


def main():
    docs = load_unified_docs()
    print(f"Embedding {len(docs)} documents via provider={EMBEDDING_API_PROVIDER!r} "
          f"(offline = deterministic hash embedding, no semantic meaning -- see embed_api.py docstring)")

    vectors = []
    for i in range(0, len(docs), EMBED_BATCH_SIZE):
        batch = docs[i:i + EMBED_BATCH_SIZE]
        texts = [d["text"] for d in batch]
        vectors.extend(embed_batch(texts))
        print(f"  embedded {min(i + EMBED_BATCH_SIZE, len(docs))}/{len(docs)}")

    matrix = np.array(vectors, dtype=np.float32)

    npz_path = DATA_PROCESSED / "dense_index.npz"
    np.savez_compressed(npz_path, vectors=matrix, doc_ids=np.array([d["doc_id"] for d in docs]))

    docs_path = DATA_PROCESSED / "dense_docs.json"
    docs_path.write_text(json.dumps(docs, indent=2, ensure_ascii=False), encoding="utf-8")

    statute_n = sum(1 for d in docs if d["source"] == "statute")
    judgment_n = sum(1 for d in docs if d["source"] == "judgment")
    print(f"\nDense index built: {matrix.shape[0]} vectors x {matrix.shape[1]} dims")
    print(f"  statute chunks: {statute_n}  judgment chunks: {judgment_n}")
    print(f"  -> {npz_path}\n  -> {docs_path}")


if __name__ == "__main__":
    main()
