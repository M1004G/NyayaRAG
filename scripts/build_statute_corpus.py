"""
Extract full section text from the manually-downloaded IPC and BNS PDFs.

Prerequisite: download both official PDFs and save into data/raw/:
    IPC:  https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf
          -> save as data/raw/ipc_full.pdf
    BNS:  https://www.ncrb.gov.in/uploads/SankalanPortal/DownloadPDF/BNS2023.pdf
          -> save as data/raw/bns_full.pdf

Run from the project root:
    pip install pdfplumber
    python scripts/build_statute_corpus.py

Output:
    data/processed/ipc_sections.json
    data/processed/bns_sections.json

WHY TWO DIFFERENT SPLITTING RULES :
  IPC format:  "NUMBER. Title.--Body text"  (title/body separated by "--",
               and the whole thing appears TWICE per section -- once as a
               short running header, once as the real content. We match on
               the "--" marker since only the real content has it, and clean
               up the duplicated running-header text from the title.)
  BNS format:  "NUMBER. Body text" directly : no "--" marker, no separate
               title (we already have clean titles for every section from
               parse_source_official.py / section_mapping.json, so we don't
               need to re-extract them here).

KNOWN LIMITATION: a small number of sections (roughly 1-2%, e.g. IPC 338/339
in testing) have their section-number marker fail to extract from the PDF
itself, due to a PDF layout quirk on that specific page, the text content
is there, just not the number that marks where it starts. This script
reports which of your section_mapping.json entries ended up with no
matching text, so you know exactly which few sections need a manual review.

Output: 
data/processed/ipc_sections.json — 585 entries (one per IPC section)
data/processed/bns_sections.json — 358 entries (one per BNS section)
"""
import json
import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("[error] Run: pip install pdfplumber", file=sys.stderr)
    sys.exit(1)

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

IPC_PDF = DATA_RAW / "ipc_full.pdf"
BNS_PDF = DATA_RAW / "bns_full.pdf"

# Confirmed against the real files: IPC's real content starts at page 0;
# BNS's front matter (arrangement of sections + a correspondence table) runs
# through page 72, with real section text starting at page 73.
IPC_START_PAGE = 0
BNS_START_PAGE = 73

# Identify amendment-history lines (footnotes) so they can be removed/ignored
# during parsing of actual legal section content.
FOOTNOTE_LINE = re.compile(
    r"^\d{1,3}\.\s+(Subs|Ins|Rep|The words|The brackets|omitted|Cl\.|Sub-s|Added|Substituted|Inserted|The letters)",
    re.IGNORECASE,
    # ^\d{1,3}\. Line starts with 1–3 digits followed by a dot (e.g., "1.", "23.", "105.")
    # \s+   One or more spaces after the number
    # (...) One of several keywords commonly used in amendment notes:
    #                      Subs (Substituted), Ins (Inserted), Rep (Repealed), etc.
    # re.IGNORECASE Matches regardless of uppercase/lowercase
)

# Matches a line that ONLY contains a section number (no text)
BARE_SECTION_RESTART = re.compile(r"^\d{1,3}[A-Z]{0,2}\.?$")
# ^\d{1,3} -> Starts with 1–3 digits (section number)
# [A-Z]{0,2} -> Optional suffix like A, B, AA (e.g., 338A, 12B)
# \.? -> Optional dot at the end (e.g., "338.")
# $ -> End of line (ensures nothing else is present)

# Matches a divider line made of dashes (like "----------")
DIVIDER = re.compile(r"\n-{10,}\n")


def clean_page(raw: str) -> str:
    """Strip amendment-history footnote lines, but unlike a naive
    'cut everything after the divider' approach keep any real section
    content that resumes AFTER the footnote block on the same page.
    Confirmed necessary: IPC section 129's entire text was being silently
    discarded because it happened to follow a footnote block on page 56,
    with no divider between the footnotes and the real content resuming."""
    m = DIVIDER.search(raw)
    if not m:
        return raw
    before = raw[: m.start()]
    after = raw[m.end():]
    lines = after.split("\n")
    kept = []
    in_footnote_block = True
    for line in lines:
        if in_footnote_block:
            if BARE_SECTION_RESTART.match(line.strip()):
                in_footnote_block = False
                kept.append(line)
            # else: still inside the footnote block (a footnote line, or a
            # wrapped continuation of one) -- drop it
        else:
            kept.append(line)
    return before + "\n" + "\n".join(kept)


def extract_clean_pages(path: Path, start_page: int) -> str:
    """Extract text per page, removing amendment-history footnote lines
    while preserving real section content that may follow them."""
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for i in range(start_page, len(pdf.pages)):
            raw = pdf.pages[i].extract_text() or ""
            pages_text.append(clean_page(raw))
    return "\n".join(pages_text)


def split_ipc(text: str):
    # Some sections render the title/body separator as "--", others as a
    # single "-" (an inconsistency in how the source PDF's dash characters
    # got extracted, confirmed against the real file -- not a real
    # formatting difference in the Act itself). Accept either.
    pattern = re.compile(
        r"(?:^|\n)(?:\d*\*?\[)?(\d{1,3}[A-Z]{0,2})\.\s+(.{1,300}?)-{1,2}(?!-)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        sec_num = m.group(1)
        raw_title = " ".join(m.group(2).split()).strip().strip('"').strip()
        # Strip the duplicated running-header text (appears before the real
        # title) by keeping only what follows the LAST "NUM." occurrence.
        title = re.sub(rf"^.*{re.escape(sec_num)}\.\s*", "", raw_title)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = " ".join(text[start:end].strip().split())
        sections.append({
            "statute": "IPC",
            "section_no": sec_num,
            "marginal_note": title,
            "text": body,
            "needs_review": len(body) < 10,
        })
    return sections


def split_bns(text: str):
    pattern = re.compile(r'(?:^|\n)(\d{1,3}[A-Z]{0,2})\.\s*(?=[A-Z\(\u201c"])')
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        sec_num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = " ".join(text[start:end].strip().split())
        sections.append({
            "statute": "BNS",
            "section_no": sec_num,
            "marginal_note": "",  # already have clean titles in section_mapping.json
            "text": body,
            "needs_review": len(body) < 10,
        })
    return sections


def main():
    if not IPC_PDF.exists() or not BNS_PDF.exists():
        print(
            f"[error] Missing PDF(s). Expected:\n  {IPC_PDF}\n  {BNS_PDF}\n"
            "Download both (see this script's docstring for URLs) before running.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Extracting IPC...")
    ipc_text = extract_clean_pages(IPC_PDF, IPC_START_PAGE)
    ipc_sections = split_ipc(ipc_text)

    print("Extracting BNS...")
    bns_text = extract_clean_pages(BNS_PDF, BNS_START_PAGE)
    bns_sections = split_bns(bns_text)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "ipc_sections.json").write_text(
        json.dumps(ipc_sections, indent=2, ensure_ascii=False)
    )
    (DATA_PROCESSED / "bns_sections.json").write_text(
        json.dumps(bns_sections, indent=2, ensure_ascii=False)
    )

    print(f"\nIPC: {len(ipc_sections)} sections -> data/processed/ipc_sections.json")
    print(f"BNS: {len(bns_sections)} sections -> data/processed/bns_sections.json")

    # Cross-check against your trusted mapping table, if it exists, so you
    # know exactly which mapped sections still need text (rather than
    # discovering gaps later when retrieval silently returns nothing).
    mapping_path = DATA_PROCESSED / "section_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        ipc_have = {s["section_no"] for s in ipc_sections}
        bns_have = {s["section_no"] for s in bns_sections}

        def norm(s):
            return s.replace(" ", "").upper()

        missing_ipc = [m["ipc_section"] for m in mapping if norm(m["ipc_section"]) not in {norm(x) for x in ipc_have}]
        missing_bns = [m["bns_section"].split("(")[0] for m in mapping if norm(m["bns_section"]).split("(")[0] not in {norm(x) for x in bns_have}]

        print(f"\nOf your {len(mapping)} trusted mapped sections:")
        print(f"  IPC text missing for: {len(missing_ipc)} sections {missing_ipc[:15]}{'...' if len(missing_ipc) > 15 else ''}")
        print(f"  BNS text missing for: {len(missing_bns)} sections {missing_bns[:15]}{'...' if len(missing_bns) > 15 else ''}")
    else:
        print(
            "\n(section_mapping.json not found -- run merge_and_validate.py first "
            "to get a coverage report against your trusted mapping.)"
        )


if __name__ == "__main__":
    main()