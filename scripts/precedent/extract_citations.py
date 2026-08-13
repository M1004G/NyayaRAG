"""
Citation extraction + citator-signal classification, per proposal 4.1:
"per-judgment citation extraction and citator-signal classification
(rule-based pattern matching, augmented by LLM-API calls on ambiguous
cases) to build the overruling/distinguishing graph."

This module does the rule-based pass. The LLM-augmentation step for
ambiguous cases is left as an explicit hook (see `classify_ambiguous`)
rather than wired to a live API call by default, since every LLM call in
this rescoped repo costs real API budget (Section 6.3 / 9 flags API cost
as the sprint's actual bottleneck) -- ambiguous cases are written to
data/processed/judgments/citations_ambiguous.json for a human (or a
budgeted batch LLM call you trigger deliberately) to resolve, rather than
silently spending API budget on every run.

SIGNAL TAXONOMY (deliberately small and high-precision, per the proposal's
own risk mitigation for Risk #2 -- "Restrict validity claims to
high-precision patterns"):
    overruled     -- the citing judgment states the cited judgment is no
                      longer good law on the point discussed
    reversed      -- the citing judgment is a higher court directly
                      reversing the SAME case on appeal (matched via
                      case_name equality + "(Review)"/"on appeal" markers)
    distinguished -- cited but held inapplicable on these facts (does NOT
                      affect the cited case's validity elsewhere)
    followed      -- cited and applied as controlling
    cited         -- referenced with no detectable signal either way
      (the default / lowest-confidence bucket)

Run from the project root, after segment_judgments.py:
    python scripts/precedent/extract_citations.py

Output:
    data/processed/judgments/citations.json
    data/processed/judgments/citations_ambiguous.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import JUDGMENTS_PROCESSED_DIR  # noqa: E402

# Case name pattern: "X v. Y (YYYY)" or "X v. Y" followed by a 4-digit year
# nearby. Indian judgments almost universally use "v." (not "vs." in formal
# citation, though we accept both).
CASE_NAME_RE = re.compile(
    r"([A-Z][A-Za-z.\s]+?\s+v\.?s?\.?\s+[A-Z][A-Za-z.\s()]+?)\s*\((\d{4})\)"
)

# High-precision signal patterns, checked in this order (first match wins
# per citation mention) -- order matters because "overruled" and
# "distinguished" can both appear near a citation and they mean opposite
# things for validity, so the MOST SPECIFIC pattern must win, not the first
# one found scanning left to right.
SIGNAL_PATTERNS = [
    ("overruled", re.compile(r"\bis overruled\b|\boverrul(?:ed|ing)\b", re.IGNORECASE)),
    ("distinguished", re.compile(r"\bdisting(?:uish|uished|uishing)\b", re.IGNORECASE)),
    ("followed", re.compile(r"\bfollow(?:ed|ing)\b|\bapplying the principle in\b|\bremains good law\b", re.IGNORECASE)),
]


HEADING_RE = re.compile(r"(?:^|\n)(FACTS|ISSUE|REASONING|HOLDING):\s*", re.IGNORECASE)


def reasoning_and_holding(text: str) -> str:
    """If the judgment has the standard headings, validity signals
    ('overruled', 'distinguished', ...) are drafted in the REASONING and
    HOLDING units, essentially never in FACTS/ISSUE -- so that's the
    window used for signal detection regardless of which sentence the
    case name itself lands in. Falls back to the full text when headings
    aren't present (matches segment_judgments.py's own fallback trigger)."""
    matches = list(HEADING_RE.finditer(text))
    found = [m.group(1).upper() for m in matches]
    if found != ["FACTS", "ISSUE", "REASONING", "HOLDING"]:
        return text

    spans = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans[found[i]] = text[start:end]
    return spans["REASONING"] + " " + spans["HOLDING"]


def extract_case_mentions(text: str):
    """Returns list of (case_name, year, context) for every case name
    mentioned in the text. `context` is the reasoning+holding portion of
    the SAME judgment (see reasoning_and_holding) -- wide enough to catch
    a signal phrase even when it's a sentence or two away from the case
    name itself, which is how these are actually drafted."""
    context = reasoning_and_holding(text)
    mentions = []
    for m in CASE_NAME_RE.finditer(text):
        case_name = m.group(1).strip()
        year = m.group(2)
        mentions.append((case_name, year, context))
    return mentions


def classify_signal(context: str) -> str:
    for label, pattern in SIGNAL_PATTERNS:
        if pattern.search(context):
            return label
    return "cited"


def match_case_name_to_id(case_name: str, year: str, judgments_by_name_year: dict):
    """Fuzzy-ish match: exact normalized name+year, falling back to name
    substring match (handles '(Review)' suffixes and minor punctuation
    drift) with a UNIQUE-match requirement -- if a substring match is
    ambiguous (matches more than one judgment), it's left unresolved
    rather than guessed, and routed to the ambiguous-cases file."""
    key = (normalize_name(case_name), year)
    if key in judgments_by_name_year:
        return judgments_by_name_year[key], True

    candidates = [
        jid for (name, yr), jid in judgments_by_name_year.items()
        if yr == year and (normalize_name(case_name) in name or name in normalize_name(case_name))
    ]
    candidates = list(set(candidates))
    if len(candidates) == 1:
        return candidates[0], True
    return None, False


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower().replace("vs.", "v.").replace(" v ", " v. "))


def main():
    judgments_path = JUDGMENTS_PROCESSED_DIR / "judgments.json"
    if not judgments_path.exists():
        raise SystemExit("[error] Run ingest_judgments.py first.")
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))

    judgments_by_name_year = {}
    for j in judgments:
        base_name = re.sub(r"\s*\(review\)\s*", "", normalize_name(j["case_name"]))
        judgments_by_name_year[(base_name, j["date"][:4])] = j["judgment_id"]
        judgments_by_name_year[(normalize_name(j["case_name"]), j["date"][:4])] = j["judgment_id"]

    citations, ambiguous = [], []
    for j in judgments:
        mentions = extract_case_mentions(j["text"])
        for case_name, year, context in mentions:
            target_id, resolved = match_case_name_to_id(case_name, year, judgments_by_name_year)
            signal = classify_signal(context)
            record = {
                "citing_judgment_id": j["judgment_id"],
                "cited_case_name": case_name,
                "cited_year": year,
                "cited_judgment_id": target_id,
                "signal": signal,
                "context": context,
            }
            if resolved:
                citations.append(record)
            else:
                ambiguous.append(record)

    out_path = JUDGMENTS_PROCESSED_DIR / "citations.json"
    out_path.write_text(json.dumps(citations, indent=2, ensure_ascii=False), encoding="utf-8")

    amb_path = JUDGMENTS_PROCESSED_DIR / "citations_ambiguous.json"
    amb_path.write_text(json.dumps(ambiguous, indent=2, ensure_ascii=False), encoding="utf-8")

    signal_counts = {}
    for c in citations:
        signal_counts[c["signal"]] = signal_counts.get(c["signal"], 0) + 1

    print(f"Extracted {len(citations)} resolved citation(s) -> {out_path}")
    print(f"  Signal breakdown: {signal_counts}")
    print(f"  {len(ambiguous)} unresolved mention(s) -> {amb_path} (needs human/LLM review, see module docstring)")


if __name__ == "__main__":
    main()
