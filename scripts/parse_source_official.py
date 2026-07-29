"""
Parse a MANUALLY SAVED copy of the official NCRB IPC<->BNS section table.

Why manual: https://www.ncrb.gov.in/uploads/SankalanPortal/SectionTableBNS.html
disallows automated fetching via robots.txt
Steps to follow: 
  1. Open the URL above in your own browser.
  2. Save the page: Ctrl+S -> "Webpage, HTML only" -> save as
     data/raw/ncrb_raw.html in this project.
  3. Run: python scripts/parse_source_official.py

REAL STRUCTURE OF THIS PAGE (confirmed via scripts/debug_ncrb_structure.py):
There are TWO <table> elements containing the SAME data, just with columns
in opposite order:
    Table 0: BNS column first, then IPC  ("Bharatiya Nyaya Sanhita... Indian
              Penal Code...")
    Table 1: IPC column first, then BNS  ("Indian Penal Code... Bharatiya
              Nyaya Sanhita...")
Using both (concatenated) corrupts the pairing wherever they meet and
double-counts every mapping. This script auto-detects and uses ONLY the
table whose first cell mentions "Indian Penal Code" (i.e. Table 1), so the
column order matches what the rest of the pipeline expects.

Within that one table, cells form a flat, alternating sequence: IPC cell,
BNS cell, IPC cell, BNS cell, ... paired up two at a time. Non-mapping rows
(chapter headers, "Deleted", "New Section", "Explanation" notes, blanks)
don't start with a section number and are automatically skipped.

Output:
    data/raw/source_official.json
"""
import json
import re 
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup #HTML-parsing library
except ImportError:
    print("[error] beautifulsoup4 not installed. Run: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)


#__file__ is a special variable that fills in with path of this script itself. 
#resolve helps to turn it into an absolute path
RAW_HTML_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ncrb_raw.html"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "source_official.json"

#using regex (regular expression) to describe patterns, it is compiled so it can be reused.
# Group 1 section number:  \d+ :one or more digits and [A-Za-z]{0,3}:optional 0–3 letters
# Group 2 sub-clauses: \s*:optional whitespace, \( ... \) : literal parentheses, [A-Za-z0-9]{1,5}: 1–5 alphanumeric chars inside
# (?:...) : non-capturing group (used for grouping only), * : allows multiple repetitions (e.g., (1)(a)(i))
# Group 3 description text : .* → any characters (greedy) and $ → match until end of string
CELL_PATTERN = re.compile(
    r"^(\d+[A-Za-z]{0,3})((?:\s*\([A-Za-z0-9]{1,5}\))*)\s*\.?\s*(.*)$"
)


def parse_cell(text: str):
    text = text.strip()
    if not text or not text[0].isdigit(): #if the cell is empty, or doesn't start with a digit, it's definitely not a section cell
        return None
    m = CELL_PATTERN.match(text) 
    if not m:
        return None
    base, subpart, rest = m.groups() #returns a tuple of all captured groups from regex. base is main section number
    #subpart is optional part, and rest has remaining text (description)
    section = base + (subpart.replace(" ", "") if subpart else "") 
    description = rest.strip().rstrip(".")
    return {"section": section, "description": description} #return dict


def select_ipc_first_table(tables):
    """Find the one table whose column order is (IPC, BNS), not (BNS, IPC)."""
    candidates = [] #empty list used for debugging in case the correct table isn't found
    for table in tables:
        cells = [c.get_text(strip=True) for c in table.find_all(["td", "th"])] #finds every table-cell (<td>) or table-header (<th>) element inside this one table, in document order.
        candidates.append(cells)
        if cells and "Indian Penal Code" in cells[0]: #checks the table isn't empty AND its very first cell contains the text "Indian Penal Code, helps to pick correct table
            return cells
    # Fallback: nothing matched the expected header text 
    print(
        "[error] Could not find a table starting with 'Indian Penal Code'. "
        f"Found {len(candidates)} table(s); first cells were: "
        f"{[c[0] if c else '(empty)' for c in candidates]}\n"
        "The page structure may have changed. inspect data/raw/ncrb_raw.html "
        "manually to see what changed",
        file=sys.stderr,
    )
    sys.exit(1)


def extract_mappings(flat_cells):
    mappings = []
    skipped = []
    for i in range(0, len(flat_cells) - 1, 2): #walks the flat cell list two at a time: cell 0 & 1 form the first pair, cell 2 & 3 the next pair, and so on.
        #This only works correctly because the table alternates IPC-cell, BNS-cell.
        ipc_raw, bns_raw = flat_cells[i], flat_cells[i + 1]
        ipc = parse_cell(ipc_raw)
        bns = parse_cell(bns_raw)
        if ipc and bns:
            mappings.append({
                "ipc_section": ipc["section"],
                "bns_section": bns["section"],
                "description": ipc["description"] or bns["description"], #if ipc["description"] is a non-empty string, use it or if it's empty, fall back to bns["description"]
                "source": "ncrb.gov.in (official)",
            })
        else:
            skipped.append((ipc_raw, bns_raw))
    return mappings, skipped


def main():
    if not RAW_HTML_PATH.exists():
        print(
            f"[error] {RAW_HTML_PATH} not found.\n"
            "Open https://www.ncrb.gov.in/uploads/SankalanPortal/SectionTableBNS.html "
            "in your browser, save it (Ctrl+S) as that exact path, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    html = RAW_HTML_PATH.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser") #"html.parser" is Python's built-in parser
    tables = soup.find_all("table")
    if not tables:
        print("[error] No <table> elements found in the saved HTML.", file=sys.stderr)
        sys.exit(1)

    flat_cells = select_ipc_first_table(tables)
    print(f"Using the IPC-first table: {len(flat_cells)} cells.")

    mappings, skipped = extract_mappings(flat_cells)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True) #check output folder exists before writing to it
    OUT_PATH.write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Extracted {len(mappings)} mappings to {OUT_PATH}")
    print(f"Skipped {len(skipped)} pairs (chapter headers, Deleted, New Section, Explanation, blanks as expected) ")
    print("\nFirst 5 extracted mappings:")
    for m in mappings[:5]:
        print(f"  IPC {m['ipc_section']} -> BNS {m['bns_section']}  |  {m['description'][:60]}")
    print("\nLast 5 extracted mappings:")
    for m in mappings[-5:]:
        print(f"  IPC {m['ipc_section']} -> BNS {m['bns_section']}  |  {m['description'][:60]}")

    if not (400 <= len(mappings) <= 550):
        print(
            f"\n[warn] {len(mappings)} mappings is outside the expected ~450-520 "
            "range for the full IPC (511 sections, minus genuinely deleted ones). "
            "Worth a manual spot-check before trusting this file.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()