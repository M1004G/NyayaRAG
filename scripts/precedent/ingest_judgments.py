"""
Load raw judgment records (data/raw/judgments/*.json, matching the schema
documented in sample_judgments.json) and normalize/validate them into
data/processed/judgments/judgments.json.

This is the swap point for real data: drop one or more real export files
into data/raw/judgments/ (each a JSON object with a top-level "judgments"
list matching the documented schema) and re-run -- nothing downstream
changes. Multiple files are merged; duplicate judgment_id raises an error
rather than silently overwriting, since a silent overwrite would quietly
shrink the corpus with no signal.

Run from the project root:
    python scripts/precedent/ingest_judgments.py

Output:
    data/processed/judgments/judgments.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import JUDGMENTS_RAW_DIR, JUDGMENTS_PROCESSED_DIR  # noqa: E402

REQUIRED_FIELDS = {"judgment_id", "case_name", "court", "date", "bench",
                    "sections_cited", "text", "disposal"}
VALID_DISPOSALS = {"allowed", "dismissed", "partly allowed", "remanded"}


def validate(record: dict, source_file: str) -> list:
    """Returns a list of problems (empty if the record is clean)."""
    problems = []
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        problems.append(f"missing fields: {sorted(missing)}")
        return problems  # can't check further without the fields

    try:
        from datetime import date
        date.fromisoformat(record["date"])
    except (ValueError, TypeError):
        problems.append(f"unparseable date: {record.get('date')!r}")

    if record["disposal"] not in VALID_DISPOSALS:
        problems.append(f"disposal {record['disposal']!r} not in {sorted(VALID_DISPOSALS)}")

    if not isinstance(record["sections_cited"], list) or not record["sections_cited"]:
        problems.append("sections_cited must be a non-empty list")
    else:
        for s in record["sections_cited"]:
            if "statute" not in s or "section_no" not in s:
                problems.append(f"malformed sections_cited entry: {s}")

    if not record["text"].strip():
        problems.append("empty text field")

    return problems


def load_raw_files() -> list:
    files = sorted(JUDGMENTS_RAW_DIR.glob("*.json"))
    if not files:
        raise SystemExit(
            f"[error] No .json files found in {JUDGMENTS_RAW_DIR}. "
            "Add sample_judgments.json (included) or a real export matching its schema."
        )
    return files


def main():
    seen_ids = {}
    clean, rejected = [], []

    for path in load_raw_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("judgments", [])
        for rec in records:
            problems = validate(rec, path.name)
            jid = rec.get("judgment_id")
            if jid in seen_ids:
                problems.append(f"duplicate judgment_id, already seen in {seen_ids.get(jid)}")
            if problems:
                rejected.append({"judgment_id": jid, "source": path.name, "problems": problems})
                continue
            seen_ids[jid] = path.name
            clean.append(rec)

    out_path = JUDGMENTS_PROCESSED_DIR / "judgments.json"
    out_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")

    if rejected:
        rej_path = JUDGMENTS_PROCESSED_DIR / "ingest_rejected.json"
        rej_path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[warn] {len(rejected)} record(s) rejected -> {rej_path}")

    print(f"Ingested {len(clean)} judgment(s) -> {out_path}")
    courts = sorted({r["court"] for r in clean})
    print(f"  Courts: {', '.join(courts)}")
    print(f"  Date range: {min(r['date'] for r in clean)} to {max(r['date'] for r in clean)}")


if __name__ == "__main__":
    main()
