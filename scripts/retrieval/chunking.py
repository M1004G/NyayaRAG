"""
The two chunking strategies for the ablation grid's chunking axis, proposal
4.2: "fixed-token (512) vs. structure-aware chunks (statute section;
judgment rhetorical unit) -- two strategies instead of the original
three, to keep the ablation CPU-tractable."

structure_aware_docs() is just dense_docs.json as already built by
build_dense_index.py -- statute sections and judgment rhetorical units ARE
the structure-aware chunks by construction, nothing new needed here.

fixed_token_docs() re-splits that SAME underlying text into ~512-word
windows, ignoring section/rhetorical-unit boundaries, tagged with the
parent_doc_id they came from -- so gold-evidence matching in
eval/run_ablation.py can credit a fixed-token chunk as a "hit" whenever it
falls inside the gold section/chunk, exactly mirroring how retrieval
evaluation is done against fixed windows in the literature this proposal
cites (LegalBench-RAG).

NOTE: because almost every statute section and judgment rhetorical unit in
this corpus is well under 512 words on its own, this strategy mostly
degenerates to structure_aware for the SAMPLE data -- the two strategies
only meaningfully diverge on the handful of longer REASONING/FACTS
judgment units. This is an honest property of a small, mostly-short-form
corpus, not a bug: the divergence would be much larger over the real
5k-8k judgment corpus, where full judgment texts (not pre-segmented) are
common.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED  # noqa: E402

import json  # noqa: E402

FIXED_WINDOW_WORDS = 512


def structure_aware_docs() -> list:
    path = DATA_PROCESSED / "dense_docs.json"
    if not path.exists():
        raise SystemExit("[error] Run build_dense_index.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_token_docs(window: int = FIXED_WINDOW_WORDS) -> list:
    base_docs = structure_aware_docs()
    out = []
    for doc in base_docs:
        words = re.findall(r"\S+", doc["text"])
        if len(words) <= window:
            # short enough that fixed-512 chunking is a no-op for this doc
            out.append({**doc, "chunk_doc_id": doc["doc_id"], "parent_doc_id": doc["doc_id"]})
            continue
        for i in range(0, len(words), window):
            sub_text = " ".join(words[i:i + window])
            out.append({
                **doc,
                "doc_id": f"{doc['doc_id']}::w{i // window}",
                "chunk_doc_id": f"{doc['doc_id']}::w{i // window}",
                "parent_doc_id": doc["doc_id"],
                "text": sub_text,
            })
    return out
