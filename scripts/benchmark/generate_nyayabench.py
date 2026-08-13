"""
Build NyayaBench: merges the pre-existing statutory/transition QA
(data/processed/qa_items.json, already built by generate_qa.py -- unchanged)
with three NEW slices built here from the judgment corpus: precedent_qa,
trap_qa, multihop_qa. Output matches the schema in schema.py.

TARGET COUNTS vs. proposal Section 5 table (100/100/100/50/50 = 400):
this repo's judgment corpus is 18 SYNTHETIC sample records (see
data/raw/judgments/sample_judgments.json's docstring), not the real
5,000-8,000 judgment corpus, so precedent_qa/trap_qa/multihop_qa are
generated at whatever count the sample data actually supports -- reported
explicitly at the end of this script, not padded to the target with
duplicates or low-quality items. TARGET_COUNTS below documents the real
targets so scaling this up is a config change, not a rewrite, once a real
corpus is ingested.

Construction protocol (Section 5): the proposal specifies LLM-proposed
candidates verified by 2-3 law-trained annotators. That human annotation
step cannot happen in this sandbox -- the items generated here are
TEMPLATE-derived directly from the trusted, already-verified
judgments.json / section_mapping.json data (not LLM-proposed, not yet
human-verified) and should be treated as a scaffold for the real
annotation workflow, not as annotated gold data ready to publish.

Run from the project root, after build_overruling_graph.py:
    python scripts/benchmark/generate_nyayabench.py

Output:
    data/processed/nyayabench/nyayabench.json
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROCESSED, JUDGMENTS_PROCESSED_DIR, NYAYABENCH_DIR, BNS_COMMENCEMENT_DATE  # noqa: E402
from schema import QAItem, validate_items  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "precedent"))
from build_overruling_graph import good_law_as_of  # noqa: E402

TARGET_COUNTS = {
    "statutory_qa": 100, "precedent_qa": 100, "transition_qa": 100,
    "trap_qa": 50, "multihop_qa": 50,
}  # per proposal Section 5 -- see module docstring for why this repo doesn't hit them

RNG_SEED = 13  # fixed, for reproducible sampling of which pre-existing qa_items.json rows get used


def load_existing_statute_qa():
    path = DATA_PROCESSED / "qa_items.json"
    if not path.exists():
        return []
    return json.loads(path.read_bytes().decode("utf-8", errors="replace"))


def build_statutory_and_transition(existing_rows: list):
    """The existing generator already produces paired pre/post items per
    section (same section, two dates, gold statute flips IPC->BNS) -- that
    pairing IS the transition_qa design. statutory_qa reuses the "post"
    half only (a plain current-law question, no transition framing)."""
    statutory, transition = [], []
    for row in existing_rows:
        item = QAItem(
            item_id=f"nb_{row['id']}",
            slice="transition_qa",
            question=row["query"],
            query_date=row["query_date"],
            gold_answer=f"{row['gold_statute']} Section {row['gold_section_no']}",
            gold_evidence_ids=[f"{row['gold_statute']}-{row['gold_section_no']}"],
            applicable_era=row["gold_statute"],
        )
        transition.append(item)
        if row["id"].endswith("_post"):
            statutory.append(QAItem(
                item_id=f"nb_{row['id']}_stat",
                slice="statutory_qa",
                question=row["query"],
                query_date=row["query_date"],
                gold_answer=f"{row['gold_statute']} Section {row['gold_section_no']}",
                gold_evidence_ids=[f"{row['gold_statute']}-{row['gold_section_no']}"],
                applicable_era=row["gold_statute"],
            ))

    rng = random.Random(RNG_SEED)
    rng.shuffle(statutory)
    rng.shuffle(transition)
    return statutory[:TARGET_COUNTS["statutory_qa"]], transition[:TARGET_COUNTS["transition_qa"]]


def build_precedent_qa(judgments: list, chunks_by_judgment: dict, graph: dict):
    """One item per judgment: 'What did the court hold in <case>?', gold =
    the HOLDING chunk. Query date = 1 day after decision (asks 'is this
    good law right now', the proposal's own precedent_qa framing) --
    deliberately includes items whose gold precedent later becomes
    overruled (validity_flags records this), per the proposal's explicit
    'includes items whose naive top answer is an overruled case'."""
    items = []
    for j in judgments:
        holding_chunks = [c for c in chunks_by_judgment.get(j["judgment_id"], []) if c["rhetorical_unit"] == "HOLDING"]
        if not holding_chunks:
            continue
        from datetime import date, timedelta
        query_date = date.fromisoformat(j["date"]) + timedelta(days=1)
        ok, reason = good_law_as_of(j["judgment_id"], query_date, graph)
        items.append(QAItem(
            item_id=f"nb_prec_{j['judgment_id']}",
            slice="precedent_qa",
            question=f"What did the court hold in {j['case_name']}?",
            query_date=query_date.isoformat(),
            gold_answer=holding_chunks[0]["text"],
            gold_evidence_ids=[holding_chunks[0]["chunk_id"]],
            validity_flags={"good_law_at_query_date": ok, **({"invalidated_because": reason} if reason else {})},
        ))
    return items[:TARGET_COUNTS["precedent_qa"]]


def build_trap_qa():
    """Fabricated sections and fabricated case names -- correct system
    behavior is a grounded refusal, per proposal Section 5. is_answerable
    is set to False and gold_evidence_ids is deliberately empty: nothing
    in the corpus SHOULD be cited as an answer to these."""
    fabricated_sections = [
        ("IPC", "999"), ("IPC", "512"), ("BNS", "450"), ("BNS", "999A"), ("CrPC", "88B"),
    ]
    fabricated_cases = [
        "Rajendra Prasad v. Union Bank of Fiction (2021)",
        "Meenakshi Sundaram v. State of Neverland (2019)",
        "Bharat Aggregates Ltd. v. Ghost Litigants Association (2023)",
    ]
    items = []
    for statute, sec in fabricated_sections:
        items.append(QAItem(
            item_id=f"nb_trap_sec_{statute}_{sec}",
            slice="trap_qa",
            question=f"What is the punishment prescribed under Section {sec} of the {statute}?",
            query_date="2025-01-01",
            gold_answer="No such section exists in the corpus; correct behavior is a grounded refusal.",
            is_answerable=False,
        ))
    for case in fabricated_cases:
        items.append(QAItem(
            item_id=f"nb_trap_case_{len(items)}",
            slice="trap_qa",
            question=f"What did the Supreme Court hold in {case}?",
            query_date="2025-01-01",
            gold_answer="No such case exists in the corpus; correct behavior is a grounded refusal.",
            is_answerable=False,
        ))
    return items[:TARGET_COUNTS["trap_qa"]]


def build_multihop_qa(judgments: list, chunks_by_judgment: dict, mapping_by_ipc: dict, graph: dict):
    """Combines a statute lookup with an interpreting judgment, per
    proposal Section 5's multihop definition: 'What section currently
    governs the offence discussed in <case>, and is that case's holding
    still good law?' -- answering requires BOTH the IPC->BNS mapping table
    AND the overruling graph, not either alone."""
    items = []
    for j in judgments:
        ipc_sections = [s["section_no"] for s in j["sections_cited"] if s["statute"] == "IPC"]
        if not ipc_sections:
            continue
        sec = ipc_sections[0]
        bns_equiv = mapping_by_ipc.get(sec.replace(" ", ""))
        if not bns_equiv:
            continue
        from datetime import date, timedelta
        query_date = date.fromisoformat(j["date"]) + timedelta(days=730)  # 2 years out, well past BNS commencement for pre-2024 judgments -- forces the multihop
        query_date = max(query_date, BNS_COMMENCEMENT_DATE)
        ok, reason = good_law_as_of(j["judgment_id"], query_date, graph)
        holding_chunks = [c for c in chunks_by_judgment.get(j["judgment_id"], []) if c["rhetorical_unit"] == "HOLDING"]
        if not holding_chunks:
            continue
        items.append(QAItem(
            item_id=f"nb_multihop_{j['judgment_id']}",
            slice="multihop_qa",
            question=(f"What section currently governs the offence discussed in {j['case_name']}, "
                       f"and is that case's holding still good law?"),
            query_date=query_date.isoformat(),
            gold_answer=(f"BNS Section {bns_equiv} (successor to IPC Section {sec}); "
                         f"the holding in {j['case_name']} is "
                         f"{'still good law' if ok else f'no longer good law ({reason})'}."),
            gold_evidence_ids=[f"BNS-{bns_equiv}", holding_chunks[0]["chunk_id"]],
            applicable_era="BNS",
            validity_flags={"good_law_at_query_date": ok},
        ))
    return items[:TARGET_COUNTS["multihop_qa"]]


def main():
    judgments_path = JUDGMENTS_PROCESSED_DIR / "judgments.json"
    chunks_path = JUDGMENTS_PROCESSED_DIR / "judgment_chunks.json"
    graph_path = JUDGMENTS_PROCESSED_DIR / "overruling_graph.json"
    mapping_path = DATA_PROCESSED / "section_mapping.json"

    if not (judgments_path.exists() and chunks_path.exists() and graph_path.exists()):
        raise SystemExit("[error] Run the precedent pipeline (ingest -> segment -> citations -> graph) first.")

    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8")).get("invalidating_edges", {})
    mapping = json.loads(mapping_path.read_bytes().decode("utf-8", errors="replace")) if mapping_path.exists() else []
    mapping_by_ipc = {m["ipc_section"].replace(" ", ""): m["bns_section"] for m in mapping}

    chunks_by_judgment = {}
    for c in chunks:
        chunks_by_judgment.setdefault(c["judgment_id"], []).append(c)

    existing_rows = load_existing_statute_qa()
    statutory_qa, transition_qa = build_statutory_and_transition(existing_rows)
    precedent_qa = build_precedent_qa(judgments, chunks_by_judgment, graph)
    trap_qa = build_trap_qa()
    multihop_qa = build_multihop_qa(judgments, chunks_by_judgment, mapping_by_ipc, graph)

    all_items = statutory_qa + precedent_qa + transition_qa + trap_qa + multihop_qa
    problems = validate_items(all_items)
    if problems:
        raise SystemExit(f"[error] Benchmark validation failed:\n" + "\n".join(problems))

    out_path = NYAYABENCH_DIR / "nyayabench.json"
    out_path.write_text(json.dumps([i.to_dict() for i in all_items], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"NyayaBench built: {len(all_items)} items -> {out_path}\n")
    print(f"{'Slice':<15}{'Count':<8}{'Target':<8}")
    for slice_name in ("statutory_qa", "precedent_qa", "transition_qa", "trap_qa", "multihop_qa"):
        n = sum(1 for i in all_items if i.slice == slice_name)
        print(f"{slice_name:<15}{n:<8}{TARGET_COUNTS[slice_name]:<8}")
    print("\n(Counts below target are expected -- see module docstring: this repo's judgment "
          "corpus is 18 synthetic sample records, not the real 5k-8k corpus.)")


if __name__ == "__main__":
    main()
