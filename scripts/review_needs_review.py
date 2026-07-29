"""
Fast manual review of data/processed/needs_review.json.
Basically, manual resolution for conflicting IPC -> BNS mappings.

only for reviewing the disputed or single-source entries,  for a handful of offences that's usually a small
list.

Run from the project root, after merge_and_validate.py:
    python scripts/review_needs_review.py
Input:
- needs_review.json → entries where sources disagree or only 1 source exists

Output:
- section_mapping.json → updated with manually approved mappings
- needs_review.json → reduced to only unresolved items

User actions:
- Enter index → accept that candidate
- 's' or Enter → skip
- 'q' → quit early

Accepted entries get appended to data/processed/section_mapping.json.
Skipped entries stay in needs_review.json untouched, safe to leave out of
your benchmark entirely if you don't have time to resolve them.
"""
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
NEEDS_REVIEW_PATH = PROCESSED_DIR / "needs_review.json"
TRUSTED_PATH = PROCESSED_DIR / "section_mapping.json"

BNS_COMMENCEMENT_DATE = "2024-07-01"


def main():
    if not NEEDS_REVIEW_PATH.exists():
        raise SystemExit("[error] Run merge_and_validate.py first.")

    needs_review = json.loads(NEEDS_REVIEW_PATH.read_text(encoding="utf-8"))
    trusted = json.loads(TRUSTED_PATH.read_text(encoding="utf-8")) if TRUSTED_PATH.exists() else []

    if not needs_review:
        print("Nothing to review -- needs_review.json is empty.")
        return

    print(f"{len(needs_review)} items to review. For each: pick a number, 's' to skip, 'q' to quit.\n")

    still_pending = []
    resolved_count = 0
    # Iterate through each unresolved IPC section
    for i, item in enumerate(needs_review):
        print(f"--- [{i + 1}/{len(needs_review)}] IPC {item['ipc_section']} ---")
        for idx, cand in enumerate(item["candidates"]):
            print(f"  {idx}) BNS {cand['bns_section']}   (source: {cand['source']})")

        choice = input("Pick number, [s]kip, [q]uit: ").strip().lower()

        if choice == "q":
            print("Stopping early -- unreviewed items remain in needs_review.json.")
            still_pending.extend(needs_review[i:])
            break
        elif choice == "s" or choice == "":
            still_pending.append(item)
            continue
        elif choice.isdigit() and int(choice) < len(item["candidates"]):
            picked = item["candidates"][int(choice)]
            trusted.append({
                "ipc_section": item["ipc_section"],
                "bns_section": picked["bns_section"],
                "description": "",  # not carried in needs_review.json -- fine, optional field
                "enactment_date": None,
                "commencement_date": None,
                "repeal_date": BNS_COMMENCEMENT_DATE,
                "sources_agreeing": [picked["source"]],
                "manually_reviewed": True,
            })
            resolved_count += 1
        else:
            print("  Not a valid choice, skipping this one.")
            still_pending.append(item)

    TRUSTED_PATH.write_text(json.dumps(trusted, indent=2, ensure_ascii=False), encoding="utf-8")
    NEEDS_REVIEW_PATH.write_text(json.dumps(still_pending, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResolved {resolved_count} -> added to section_mapping.json")
    print(f"Still pending: {len(still_pending)} -> needs_review.json")


if __name__ == "__main__":
    main()
