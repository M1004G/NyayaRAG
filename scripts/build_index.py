"""
Build a BM25 index over the combined IPC + BNS section corpus.

Uses rank_bm25 (pure Python, pip-installable, no Java):
The retrieval logic is the same as Pyserini.

Run from the project root, after build_statute_corpus.py:
    pip install rank_bm25
    python scripts/build_index.py

Output:
    data/processed/corpus.json       -- combined, unified-schema corpus
    (the BM25 index itself is rebuilt in-memory each time retrieve.py runs,
    since rank_bm25 doesn't persist to disk -- rebuilding from corpus.json
    takes well under a second for a corpus this size)
"""
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_corpus():
    ipc = json.loads((PROCESSED_DIR / "ipc_sections.json").read_text(encoding="utf-8"))
    bns = json.loads((PROCESSED_DIR / "bns_sections.json").read_text(encoding="utf-8"))

    corpus = []
    for s in ipc:
        corpus.append({
            "doc_id": f"IPC-{s['section_no']}",
            "statute": "IPC",
            "section_no": s["section_no"],
            "marginal_note": s.get("marginal_note", ""),
            "text": s["text"],
        })
    for s in bns:
        corpus.append({
            "doc_id": f"BNS-{s['section_no']}",
            "statute": "BNS",
            "section_no": s["section_no"],
            "marginal_note": s.get("marginal_note", ""),
            "text": s["text"],
        })
    return corpus


def main():
    corpus = build_corpus()
    out_path = PROCESSED_DIR / "corpus.json"
    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Combined corpus: {len(corpus)} documents -> {out_path}")
    #count how many documents have statute == "IPC"
    ipc_count = sum(1 for d in corpus if d["statute"] == "IPC")

    bns_count = sum(1 for d in corpus if d["statute"] == "BNS")
    print(f"  IPC: {ipc_count}  BNS: {bns_count}")


if __name__ == "__main__":
    main()
