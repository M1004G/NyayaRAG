"""
Auto-generate QA benchmark items from the trusted section_mapping.json --
but ONLY for section pairs where build_statute_corpus.py actually found real
text for BOTH the IPC and BNS side. Pairs missing text on either side are
skipped and reported, not silently included with an empty/missing body.

Why skip instead of chasing 100% extraction coverage: pdfplumber has a small
number of genuine extraction gaps in these specific PDFs (confirmed --
sometimes a whole section's text doesn't extract from its page, not just the
number marker). 

Run from the project root, after merge_and_validate.py AND
build_statute_corpus.py:
    python scripts/generate_qa.py

Output:
    data/processed/qa_items.json
"""
import json
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

BNS_COMMENCEMENT_DATE = date(2024, 7, 1)
BEFORE_DATE = BNS_COMMENCEMENT_DATE - timedelta(days=30)
AFTER_DATE = BNS_COMMENCEMENT_DATE + timedelta(days=365)


def make_query(description: str) -> str:
    return f"What is the applicable law for: {description}?"


def normalize(s: str) -> str:
    return s.replace(" ", "").upper().split("(")[0]  # ignore sub-clause parens for matching


def main():
    mapping_path = PROCESSED_DIR / "section_mapping.json"
    ipc_sections_path = PROCESSED_DIR / "ipc_sections.json"
    bns_sections_path = PROCESSED_DIR / "bns_sections.json"

    if not mapping_path.exists():
        raise SystemExit("[error] Run merge_and_validate.py first.")
    if not ipc_sections_path.exists() or not bns_sections_path.exists():
        raise SystemExit("[error] Run build_statute_corpus.py first.")

    mappings = json.loads(mapping_path.read_text())
    ipc_sections = json.loads(ipc_sections_path.read_text())
    bns_sections = json.loads(bns_sections_path.read_text())

    ipc_have = {normalize(s["section_no"]) for s in ipc_sections}
    bns_have = {normalize(s["section_no"]) for s in bns_sections}

    qa_items = []
    skipped = []

    for i, m in enumerate(mappings):
        ipc_ok = normalize(m["ipc_section"]) in ipc_have
        bns_ok = normalize(m["bns_section"]) in bns_have

        if not (ipc_ok and bns_ok):
            skipped.append({
                "ipc_section": m["ipc_section"],
                "bns_section": m["bns_section"],
                "ipc_text_available": ipc_ok,
                "bns_text_available": bns_ok,
            })
            continue

        base_id = f"stat_{i:04d}"
        qa_items.append({
            "id": f"{base_id}_pre",
            "query": make_query(m["description"]),
            "query_date": BEFORE_DATE.isoformat(),
            "slice": "statutory",
            "gold_section_no": m["ipc_section"],
            "gold_statute": "IPC",
        })
        qa_items.append({
            "id": f"{base_id}_post",
            "query": make_query(m["description"]),
            "query_date": AFTER_DATE.isoformat(),
            "slice": "statutory",
            "gold_section_no": m["bns_section"],
            "gold_statute": "BNS",
        })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "qa_items.json").write_text(
        json.dumps(qa_items, indent=2, ensure_ascii=False)
    )
    (PROCESSED_DIR / "qa_skipped.json").write_text(
        json.dumps(skipped, indent=2, ensure_ascii=False)
    )

    print(f"Generated {len(qa_items)} QA items ({len(qa_items) // 2} mapping pairs x 2) -> data/processed/qa_items.json")
    print(f"Skipped {len(skipped)} mapping pairs (missing text on one or both sides) -> data/processed/qa_skipped.json")


if __name__ == "__main__":
    main()