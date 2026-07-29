"""
Loads every data/raw/source_*.json file present (auto-discovered, any
number), and would cross-check them against each other if there were more
than one. Currently only source_official.json exists, so every
trusted mapping comes from that single source. This is still the
"validation without reading law" step: we never judge whether a mapping is
legally correct, only whether the sources present agree.

Rule: the official NCRB source (source_official.json, from
parse_source_official.py) is weighted at 2 votes and is trusted on its own
as it's the primary government table. If additional secondary sources are
added later (e.g. from independent scrapers), they'd be weighted at 1 vote
each and need TWO of them to agree with each other to be trusted on a
section the official source doesn't cover. A single secondary source alone,
or a tie, goes to needs_review.json instead of being trusted automatically.

Current state: source_official.json alone populates section_mapping.json.
The secondary-source voting logic is unused infrastructure right now, not
an active cross-validation step right now.

Run from the project root, after parsing your source(s):
    python scripts/merge_and_validate.py

Outputs:
    data/processed/section_mapping.json   --> trusted, majority-agreeing mappings
    data/processed/needs_review.json      --> ties / disagreements / single-source-only

"""
import json
import sys
from collections import Counter #ython's standard library built specifically for counting
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BNS_COMMENCEMENT_DATE = "2024-07-01"
OFFICIAL_SOURCE_NAME = "ncrb.gov.in (official)"
OFFICIAL_SOURCE_WEIGHT = 2


def normalize_section(section: str) -> str:
    return section.replace(" ", "").upper()


def load_all_sources():
    source_files = sorted(RAW_DIR.glob("source_*.json")) #glob pattern matching: finds every file in data/raw/ whose name matches source_*.json
    if not source_files:
        print("[error] No data/raw/source_*.json files found. Run a scraper first.", file=sys.stderr)
        sys.exit(1)

    entries_by_ipc = {}  # ipc_key -> list of (bns_key, weight, raw_entry)
    for path in source_files:  #path = one file at a time
        entries = json.loads(path.read_text(encoding="utf-8"))
        for e in entries:
            ipc_key = normalize_section(e["ipc_section"])
            bns_key = normalize_section(e["bns_section"])
            weight = OFFICIAL_SOURCE_WEIGHT if e.get("source") == OFFICIAL_SOURCE_NAME else 1
            entries_by_ipc.setdefault(ipc_key, []).append((bns_key, weight, e))
    return entries_by_ipc, [p.name for p in source_files]


def main():
    entries_by_ipc, source_files = load_all_sources()
    print(f"Loaded sources: {', '.join(source_files)}")

    trusted = []
    needs_review = []

    for ipc_key, votes in sorted(entries_by_ipc.items()):
        tally = Counter()
        for bns_key, weight, _ in votes:
            tally[bns_key] += weight

        top_bns, top_score = tally.most_common(1)[0]
        total_score = sum(tally.values())

        distinct_sources = len(votes)
        is_tie = list(tally.values()).count(top_score) > 1

        # Trust if EITHER the official source alone backs this mapping (weight 2
        # on its own clears the threshold), OR two-or-more secondary sources agree
        # with each other (1+1 also clears it). A single secondary/blog source
        # alone (weight 1) does not clear it and goes to needs_review.
        if top_score >= OFFICIAL_SOURCE_WEIGHT and not is_tie and top_score > total_score / 2:
            winning_entry = next(e for bns_key, _, e in votes if bns_key == top_bns)
            all_sources = sorted(set(e.get("source", "?") for _, _, e in votes))
            trusted.append({
                "ipc_section": winning_entry["ipc_section"],
                "bns_section": winning_entry["bns_section"],
                "description": winning_entry["description"],
                "enactment_date": None,
                "commencement_date": None,
                "repeal_date": BNS_COMMENCEMENT_DATE,
                "sources_agreeing": all_sources,
                "vote_detail": dict(tally),
            })
        else:
            needs_review.append({
                "ipc_section": votes[0][2]["ipc_section"],
                "reason": "tie_or_single_source" if not is_tie else "tied_disagreement",
                "candidates": [
                    {"bns_section": e["bns_section"], "source": e.get("source", "?")}
                    for _, _, e in votes
                ],
            })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "section_mapping.json").write_text(
    json.dumps(trusted, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (PROCESSED_DIR / "needs_review.json").write_text(
        json.dumps(needs_review, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Trusted (majority agreement): {len(trusted)} -> data/processed/section_mapping.json")
    print(f"Needs review: {len(needs_review)} -> data/processed/needs_review.json")

    has_official = any(
        e.get("source") == OFFICIAL_SOURCE_NAME
        for votes in entries_by_ipc.values()
        for _, _, e in votes
    )
    if len(source_files) < 2 and not has_official:
        print(
            "\n[warn] Only one secondary (non-official) source loaded -- a single "
            "blog source alone isn't enough to trust anything, so everything "
            "landed in needs_review.json. Add source_official.json "
            "(parse_source_official.py) or a second secondary source "
            "(scrape_source_a.py / scrape_source_b.py) to start trusting entries.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()