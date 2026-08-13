# NyayaRAG

A validity-aware retrieval-augmented generation (RAG) system for Indian
criminal law. It tracks the July 1, 2024 transition from the Indian Penal
Code (IPC) to the Bharatiya Nyaya Sanhita (BNS), so that a legal query
gets an answer citing whichever statute was actually in force on the date
in question — instead of whichever section a plain keyword search happens
to rank first.

## Why this exists

An AI system with no awareness of the IPC→BNS transition will answer "what's
the punishment for cheating in 2023" and "in 2025" the same way, using
whichever section its retriever ranks highest — which is often wrong. This
is the exact class of mistake behind real-world sanctions against lawyers
who filed AI-generated citations to outdated law. NyayaRAG's core
contribution is a **validity layer** that demotes (not deletes) wrong-era
statute sections at retrieval time, so the top result is always era-correct
when a correct one exists.

## Pipeline

Built and tested in this order — each stage's output feeds the next:

1. **Section mapping** (`parse_source_official.py`) — parses the official
   NCRB IPC↔BNS concordance table into structured `{ipc_section,
   bns_section}` pairs.
2. **Trust scoring** (`merge_and_validate.py`) — turns raw parsed mappings
   into a trusted table (currently backed by the single official NCRB
   source; a weighted-voting scheme exists for combining additional
   sources later, but is unused infrastructure right now).
3. **Manual review** (`review_needs_review.py`, optional) — a small CLI to
   resolve the handful of disputed/ambiguous mappings by hand.
4. **Full section text extraction** (`build_statute_corpus.py`) — pulls the
   complete operative text of every IPC and BNS section out of the
   official PDFs via `pdfplumber`.
5. **QA benchmark generation** (`generate_qa.py`) — for every trusted,
   fully-extracted mapping pair, mechanically generates two test
   questions (one dated before the transition, gold = IPC; one dated
   after, gold = BNS). No person writes or judges these — correctness
   comes from the validated mapping table, not subject-matter judgment.
6. **Indexing** (`build_index.py`) — builds a combined IPC+BNS corpus for
   retrieval.
7. **Retrieval + validity layer** (`retrieve.py`, `era_filter.py`) — BM25
   keyword search (`rank_bm25`), with a `naive` mode (no date awareness)
   and a `validity_aware` mode that demotes wrong-era documents by a fixed
   factor before re-ranking.
8. **Retrieval evaluation** (`run_baseline_vs_validity.py`) — runs the full
   QA benchmark through both modes and computes **Statutory Era
   Accuracy**.
9. **Generation** (`generate.py`) — retrieval + an LLM (Llama 3.3 70B via
   Groq's free API) that must cite every claim as `[Doc N]` and must state
   any system-computed validity notice (e.g. "IPC Section 420 was repealed
   w.e.f. 2024-07-01; corresponding current provision is BNS Section
   318(4)") as a verified fact, not something left to its judgment.
10. **Generation evaluation** (`run_generation_eval.py`) — samples the QA
    benchmark through the full retrieve+generate pipeline and checks
    citation correctness and validity-notice surfacing.

## Results so far

**Statute layer** (`parse_source_official.py` → `build_statute_corpus.py`):

| | Count |
|---|---|
| Trusted IPC↔BNS section mappings | 498 |
| Mappings still needing manual review | 6 |
| IPC sections with full extracted text | 585 |
| BNS sections with full extracted text | 358 (= the correct total for BNS — a useful sanity check) |
| Combined retrieval corpus | 943 documents |
| Auto-generated QA benchmark items | 948 (474 mapping pairs × 2 dates) |
| Mapping pairs skipped (missing text on one side) | 24 |

**Retrieval — Statutory Era Accuracy** (`run_baseline_vs_validity.py`, full 948-item benchmark):

| Mode | Accuracy |
|---|---|
| Naive (no date awareness) | 37.6% |
| Validity-aware | 58.5% |

The more important number is in the *error breakdown*, not just the top
line: in naive mode, 50% of all errors are the system citing an entirely
wrong-era statute. In validity-aware mode, **0 of 948 items** fail for
that reason — the validity layer eliminates the whole error category it
was built to remove. What's left is a separate, already-diagnosed
problem: BM25 confusing numerically/topically adjacent sections with
similar wording (a retrieval-precision ceiling, not a validity failure).

**Generation** (`run_generation_eval.py`, 30-item sample):

| Metric | Result |
|---|---|
| Citation accuracy (cited the correct section) | 90.0% |
| No citation given at all | 6.7% |
| Model correctly abstained (docs didn't answer the question) | 6.7% |
| Validity notice correctly surfaced when one was computed | 100% |

## Repo structure

```
NyayaRAG/
├── data/
│   ├── raw/                        # manually-downloaded sources + parsed intermediate output
│   │   ├── ncrb_raw.html           # manual save of the official NCRB concordance table
│   │   ├── ipc_full.pdf            # manual download, India Code portal
│   │   ├── bns_full.pdf            # manual download, NCRB
│   │   └── source_official.json    # output of parse_source_official.py
│   └── processed/                  # every pipeline stage's output lands here
│       ├── section_mapping.json    # trusted IPC<->BNS mappings
│       ├── needs_review.json       # disputed/single-source mappings
│       ├── ipc_sections.json       # full IPC section text
│       ├── bns_sections.json       # full BNS section text
│       ├── corpus.json             # combined retrieval corpus
│       ├── qa_items.json           # auto-generated QA benchmark
│       ├── qa_skipped.json         # mapping pairs skipped (missing text)
│       ├── eval_results.json       # retrieval eval output
│       └── generation_eval_results.json  # generation eval output
├── scripts/
│   ├── parse_source_official.py    # NCRB HTML -> source_official.json
│   ├── merge_and_validate.py       # trust-scores raw mappings -> section_mapping.json / needs_review.json
│   ├── review_needs_review.py      # interactive CLI to manually resolve disputed mappings
│   ├── build_statute_corpus.py     # IPC/BNS PDFs -> full section text
│   ├── generate_qa.py              # trusted mapping -> QA benchmark
│   ├── build_index.py              # builds the combined BM25 corpus
│   ├── era_filter.py               # the validity (demotion) layer
│   ├── retrieve.py                 # BM25 retrieval, naive vs validity-aware
│   ├── run_baseline_vs_validity.py # full retrieval evaluation -> Statutory Era Accuracy
│   ├── generate.py                 # retrieval + cited LLM generation (Groq)
│   └── run_generation_eval.py      # samples QA set through generate.py, checks citations
├── requirements.txt
└── README.md
```

## Setup

1. Open the project folder in VS Code (or your editor of choice).
2. Create and activate a virtual environment:

   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Manually download three source files (both source sites disallow
   automated fetching via `robots.txt`, so this project fetches them the
   way a human visitor would — open the page, save it — rather than
   routing around that restriction):

   | File | Source | Save as |
   |---|---|---|
   | NCRB IPC↔BNS concordance table | `https://www.ncrb.gov.in/uploads/SankalanPortal/SectionTableBNS.html` (Ctrl+S → "Webpage, HTML only") | `data/raw/ncrb_raw.html` |
   | Full IPC text | `https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf` | `data/raw/ipc_full.pdf` |
   | Full BNS text | `https://www.ncrb.gov.in/uploads/SankalanPortal/DownloadPDF/BNS2023.pdf` | `data/raw/bns_full.pdf` |

5. For the generation layer only: get a free API key from
   [console.groq.com/keys](https://console.groq.com/keys), then create a
   `.env` file in the project root:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
   (`.env` is already git-ignored.)

## Running the pipeline

**Statute + retrieval layer** (in order):
```bash
python scripts/parse_source_official.py
python scripts/build_statute_corpus.py
python scripts/merge_and_validate.py
python scripts/review_needs_review.py    # optional — manually resolve disputed mappings
python scripts/generate_qa.py
python scripts/build_index.py
python scripts/run_baseline_vs_validity.py
```

**Quick manual retrieval test:**
```bash
python scripts/retrieve.py "punishment for murder" 2025-01-01
```

**Generation layer** (needs `GROQ_API_KEY`, requires the steps above to have run first):
```bash
python scripts/generate.py "What is the punishment for theft?" 2025-01-01
python scripts/run_generation_eval.py 30    # sample size, default 30
```

## Known limitations

- **Single-source trust.** The mapping is trusted based on one
  authoritative source (the official NCRB table). A weighted-voting
  scheme for combining multiple independent sources exists in
  `merge_and_validate.py` but is currently unused infrastructure.
- **~24 sections missing full text.** `pdfplumber` cannot cleanly extract
  a small number of sections from their specific PDF pages (see
  `data/processed/qa_skipped.json`). These are excluded from the QA
  benchmark rather than silently included with missing text.
- **Statute-wide era boundary.** All IPC sections are treated as repealed
  on one single date. A handful of BNS provisions have individually
  delayed commencement dates, which this does not model.
- **Document-level citations.** Generation citations are `[Doc N]`
  (section-level), not paragraph-level, because the corpus stores each
  section as a single text block.
- **Criminal law only.** Covers the IPC/BNS pair only — not the Code of
  Criminal Procedure/BNSS, Indian Evidence Act/BSA, or other central acts.

## Architecture v2: precedent layer, hybrid retrieval, NyayaBench, evaluation

Everything below was added on top of the phase-1 statute-only pipeline
above, following the CPU-only rescoped NyayaRAG proposal. **The phase-1
pipeline above is completely unchanged and still works standalone** —
nothing in `scripts/precedent/`, `scripts/retrieval/`, `scripts/benchmark/`,
or `scripts/eval/` is imported by `retrieve.py` / `generate.py` /
`build_index.py`.

### Read this first: what's real here vs. what's sample/stubbed

This was built in a sandbox with **no network access** to court-judgment
sources, HuggingFace, or any embedding/LLM API host other than Groq's, and
**no API keys** configured for OpenAI/Cohere/Voyage/etc. Concretely:

| Piece | Status here | To make it real |
|---|---|---|
| Judgment corpus | 18 **synthetic, hand-authored** sample judgments (`data/raw/judgments/sample_judgments.json`) — schema-accurate, not real case law | Drop a real export (matching the documented schema) into `data/raw/judgments/`, re-run `ingest_judgments.py` onward — nothing downstream changes |
| Dense embeddings | `EMBEDDING_API_PROVIDER=offline` by default: a deterministic hash embedding with **no semantic meaning**, used only to exercise the code path | Set `EMBEDDING_API_PROVIDER`/`EMBEDDING_API_KEY`/`EMBEDDING_MODEL` in `.env` — `openai`, `cohere`, and `voyage` are already wired in `embed_api.py` |
| Reranking | Disabled — `sentence-transformers` model weights aren't downloadable from this sandbox (no `huggingface.co` access) | `pip install sentence-transformers torch`; `rerank.py` will pick it up automatically, no code change |
| Generator ablation axis | Skipped — no `GROQ_API_KEY` configured here | Set `GROQ_API_KEY` in `.env`; `run_ablation.py`'s `generation_eval_stub()` is the documented plug point |
| NyayaBench annotation | Template-generated from trusted data (`judgments.json`, `section_mapping.json`), **not yet LLM-proposed + human-verified** per the proposal's Section 5 protocol | Route `nyayabench.json` through that protocol before treating it as gold |
| Citation faithfulness (proposition-level) | A transparent **lexical token-overlap proxy**, not real NLI | Swap in DeBERTa-v3-small at the documented point in `citation_faithfulness.py` once model weights are reachable |

Every module docstring repeats its own version of this table's relevant row
— the intent is that no script's output can be mistaken for a
publication-ready number without reading why it's scoped the way it is.

### New components

**Precedent / case-law layer** (`scripts/precedent/`) — the other half of
Novelty Claim #2 (statute-side validity was already `era_filter.py`):
- `ingest_judgments.py` — loads + validates raw judgment records
- `segment_judgments.py` — rule-based rhetorical segmentation
  (FACTS/ISSUE/REASONING/HOLDING), with a cue-word fallback classifier for
  judgments without explicit headings
- `extract_citations.py` — regex case-citation extraction + citator-signal
  classification (`overruled` / `reversed` / `distinguished` / `followed` /
  `cited`)
- `build_overruling_graph.py` — the validity graph + `good_law_as_of(judgment_id, query_date)`,
  used by retrieval exactly the way `era_filter.py` is: **demote, don't hard-filter**

**Hybrid retrieval** (`scripts/retrieval/`):
- `embed_api.py` — pluggable embedding client (openai/cohere/voyage/offline)
- `build_dense_index.py` — combined statute+judgment dense index (FAISS-CPU-equivalent; brute-force cosine via numpy, exact at this corpus scale)
- `chunking.py` — the two chunking strategies for the ablation grid (structure-aware vs. fixed-512-token)
- `hybrid_retrieve.py` — BM25 + dense fused via Reciprocal Rank Fusion, with **both** validity axes composed (statute era demotion + precedent overruling demotion)
- `rerank.py` — CPU cross-encoder reranking over the top candidates, with graceful no-op fallback

**NyayaBench** (`scripts/benchmark/`):
- `schema.py` — the shared `QAItem` dataclass (5 slices: statutory_qa, precedent_qa, transition_qa, trap_qa, multihop_qa)
- `generate_nyayabench.py` — builds all 5 slices; reuses the existing `qa_items.json` for statutory/transition, generates precedent/trap/multihop fresh from the judgment corpus

**Evaluation** (`scripts/eval/`):
- `metrics.py` — Recall@k, MRR@10, nDCG@10, Statutory Era Accuracy, Precedent Validity Rate, fabricated-citation rate, trap-abstention rate
- `citation_faithfulness.py` — proposition-level Citation Precision/Recall (lexical proxy, see table above)
- `stats.py` — paired bootstrap significance testing + Holm-Bonferroni correction, per proposal 6.2
- `run_ablation.py` — the chunking x retriever x validity ablation grid (12 configs, runnable here; generator axis is the documented plug point for the full 24-36)

### Running architecture v2

```bash
# Precedent layer (after the phase-1 statute pipeline has been run once)
python scripts/precedent/ingest_judgments.py
python scripts/precedent/segment_judgments.py
python scripts/precedent/extract_citations.py
python scripts/precedent/build_overruling_graph.py

# Hybrid retrieval
python scripts/retrieval/build_dense_index.py
python scripts/retrieval/hybrid_retrieve.py "punishment for cheating" 2025-01-01 hybrid on

# NyayaBench
python scripts/benchmark/generate_nyayabench.py

# Evaluation
python scripts/eval/run_ablation.py
```

On the sample data, the full ablation grid reproduces the phase-1 pipeline's
headline finding (validity-aware retrieval beats naive) **and extends it**:
the validity layer's Recall@10 improvement is statistically significant
(Holm-Bonferroni corrected, paired bootstrap) across every
chunking-x-retriever combination tested, not just BM25-over-statutes.
Numbers themselves are sample-scale and not to be read as the paper's
real results — see the scope table above.

### What's still not built

- Real LLM-augmented resolution of ambiguous citations (`extract_citations.py` writes these to `citations_ambiguous.json` rather than spending API budget automatically)
- RAGAS/ARES-style answer-level faithfulness and LLM-judge answer quality (need a live judge call)
- The expert-rater correlation study (Krippendorff's alpha) — needs human law-trained annotators
- Real generator-axis ablation (needs `GROQ_API_KEY` + a second hosted model)
- The manuscript itself
