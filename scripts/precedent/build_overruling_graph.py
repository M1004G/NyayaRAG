"""
Build the overruling graph and provide the precedent-validity lookup that
retrieval/hybrid_retrieve.py demotes invalid precedent with -- this is the
other half of Novelty Claim #2 (statute-side validity was already handled
by era_filter.py; this module is the case-law side).

Design mirrors era_filter.py deliberately (same DEMOTE-not-filter
philosophy, same "explicit query date required" rule) so the two validity
axes (statute era + precedent validity) compose predictably in
hybrid_retrieve.py instead of fighting each other.

VALIDITY RULE:
  A judgment is GOOD LAW as of query_date if no "overruled" or "reversed"
  edge targeting it has a citing_judgment whose date <= query_date.
  "distinguished" and "followed" edges never affect validity -- only
  overruled/reversed do (see extract_citations.py's signal taxonomy).
  A judgment overruled as of one date can be a hard problem for the SAME
  case at an EARLIER query date -- correctly, a judgment overruled in 2021
  was still good law in 2019, so a Transition-QA-style query dated 2019
  should NOT demote it. This is why validity is a function of query_date,
  not a static boolean.

Run from the project root, after extract_citations.py:
    python scripts/precedent/build_overruling_graph.py

Output:
    data/processed/judgments/overruling_graph.json
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import JUDGMENTS_PROCESSED_DIR  # noqa: E402

INVALIDATING_SIGNALS = {"overruled", "reversed"}

# Same philosophy as era_filter.DEMOTION_FACTOR: a bad-law precedent is
# ranked down, not made unretrievable -- "what did the court say before it
# was overruled" is a legitimate historical query (mirrors the Transition
# QA slice's own reasoning for statutes).
PRECEDENT_DEMOTION_FACTOR = 0.05


def build_graph(citations: list, judgments_by_id: dict) -> dict:
    """Returns {judgment_id: [invalidating_edge, ...]} -- only
    overruled/reversed edges are kept in the graph proper (distinguished/
    followed/cited are informational and stored separately for
    transparency but don't drive demotion)."""
    graph = {}
    informational = {}
    for c in citations:
        target = c["cited_judgment_id"]
        if target is None or target not in judgments_by_id:
            continue
        if c["signal"] in INVALIDATING_SIGNALS:
            graph.setdefault(target, []).append({
                "invalidated_by": c["citing_judgment_id"],
                "invalidated_on": judgments_by_id[c["citing_judgment_id"]]["date"],
                "signal": c["signal"],
                "context": c["context"],
            })
        else:
            informational.setdefault(target, []).append({
                "related_judgment": c["citing_judgment_id"],
                "on": judgments_by_id[c["citing_judgment_id"]]["date"],
                "signal": c["signal"],
            })
    return graph, informational


def good_law_as_of(judgment_id: str, query_date: date, graph: dict) -> tuple:
    """Returns (is_good_law: bool, reason: str | None). Used both here for
    reporting and imported directly by hybrid_retrieve.py at query time."""
    edges = graph.get(judgment_id, [])
    for edge in edges:
        if date.fromisoformat(edge["invalidated_on"]) <= query_date:
            return False, (
                f"{edge['signal']} by {edge['invalidated_by']} "
                f"on {edge['invalidated_on']}"
            )
    return True, None


def main():
    citations_path = JUDGMENTS_PROCESSED_DIR / "citations.json"
    judgments_path = JUDGMENTS_PROCESSED_DIR / "judgments.json"
    if not citations_path.exists() or not judgments_path.exists():
        raise SystemExit("[error] Run ingest_judgments.py and extract_citations.py first.")

    citations = json.loads(citations_path.read_text(encoding="utf-8"))
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    judgments_by_id = {j["judgment_id"]: j for j in judgments}

    graph, informational = build_graph(citations, judgments_by_id)

    out = {
        "invalidating_edges": graph,
        "informational_edges": informational,
        "demotion_factor": PRECEDENT_DEMOTION_FACTOR,
    }
    out_path = JUDGMENTS_PROCESSED_DIR / "overruling_graph.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Overruling graph: {len(graph)} judgment(s) with an invalidating edge -> {out_path}")
    for jid, edges in graph.items():
        case = judgments_by_id[jid]["case_name"]
        for e in edges:
            print(f"  [{e['signal']}] {case} ({jid}) <- {e['invalidated_by']} on {e['invalidated_on']}")

    # Quick sanity demo: same judgment, two query dates either side of the
    # overruling event, per the module docstring's worked example.
    if graph:
        sample_jid = next(iter(graph))
        for probe_date in (date(2019, 1, 1), date(2022, 1, 1)):
            ok, reason = good_law_as_of(sample_jid, probe_date, graph)
            print(f"\n  good_law_as_of({sample_jid}, {probe_date}) = {ok}"
                  + (f"  ({reason})" if reason else ""))


if __name__ == "__main__":
    main()
