"""
The validity layer, the actual novel/research component of this project.
Everything else in this pipeline exists to feed this or measure its effect.

Design decisions made explicit here:

1. DEMOTE, not hard-filter. A repealed-era statute is never removed from
   the index or made unretrievable -- it's ranked lower. This matters
   because "what was the law before July 2024" is a legitimate query type
   (your Transition QA slice depends on being able to answer it), and a
   hard filter would make it unanswerable.

2. Statute-level era boundary, not per-section repeal dates. All IPC
   sections were repealed on the same day (2024-07-01) when BNS commenced.
   This is a simplification (a handful of BNS provisions have delayed
   commencement -- e.g. BNS 106(2) -- which this does NOT model).

3. Query date is REQUIRED and must be explicit (an ISO date string on the
   QA item). No date-inference from free text : that was deliberately cut
   from scope to avoid contaminating the Statutory Era Accuracy number with
   a second, unrelated source of error (see earlier discussion: if
   inference is noisy, a wrong answer could be a retrieval failure OR a
   date-inference failure, and you can't tell which).
"""
from datetime import date

BNS_COMMENCEMENT_DATE = date(2024, 7, 1)

# How much to multiply a wrong-era document's score by. Not zero, that
# would be a hard filter. Low enough that a same-era match will essentially
# always outrank a wrong-era one, but the wrong-era doc can still surface if
# genuinely nothing else in the index matches the query at all.
DEMOTION_FACTOR = 0.05


def valid_statute_for_date(query_date: date) -> str:
    """Which statute is in force on this date, at the statute-wide level."""
    return "IPC" if query_date < BNS_COMMENCEMENT_DATE else "BNS"


def apply_validity_filter(scored_docs, query_date: date, mode: str = "demote"):
    """
    scored_docs: list of (doc, score) tuples, already ranked by retrieval.
    query_date: the date the query is asking about.
    mode: "demote" (default, recommended) or "off" (naive -- no adjustment).

    Returns a new list of (doc, score) tuples, re-sorted by adjusted score.
    """
    if mode == "off":
        return scored_docs

    valid_statute = valid_statute_for_date(query_date)
    adjusted = []
    for doc, score in scored_docs:
        if doc["statute"] != valid_statute:
            score = score * DEMOTION_FACTOR
        adjusted.append((doc, score))

    adjusted.sort(key=lambda pair: pair[1], reverse=True)
    return adjusted
