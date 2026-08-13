"""
NyayaBench QA item schema, per proposal Section 5. One dataclass shared by
generate_nyayabench.py (writer) and everything in scripts/eval/ (readers),
so the schema is defined exactly once.

SLICE DEFINITIONS (proposal Section 5 table):
    statutory_qa   -- direct questions on current law; gold = in-force section text
    precedent_qa   -- answered by case law; gold = controlling judgment paragraphs;
                       includes items whose naive top answer is an overruled case
    transition_qa  -- paired/dated queries spanning IPC->BNS; gold era-specific answers
    trap_qa        -- premised on non-existent sections/fabricated cases; correct
                       behavior = grounded refusal
    multihop_qa    -- require combining statute + interpreting judgment
"""
from dataclasses import dataclass, field, asdict

VALID_SLICES = {"statutory_qa", "precedent_qa", "transition_qa", "trap_qa", "multihop_qa"}


@dataclass
class QAItem:
    item_id: str
    slice: str
    question: str
    query_date: str  # ISO date -- required, see era_filter.py's "no date-inference from free text" design rule, reused here
    gold_answer: str
    gold_evidence_ids: list = field(default_factory=list)  # doc_ids into dense_docs.json (statute doc_id or judgment chunk_id)
    applicable_era: str = ""  # "IPC" | "BNS" | "" (n/a for precedent/trap items)
    is_answerable: bool = True  # False only for trap_qa items -- correct system behavior is abstention
    validity_flags: dict = field(default_factory=dict)  # free-form: e.g. {"cited_precedent_overruled_as_of_query_date": true}

    def __post_init__(self):
        if self.slice not in VALID_SLICES:
            raise ValueError(f"slice must be one of {sorted(VALID_SLICES)}, got {self.slice!r}")
        if not self.is_answerable and self.gold_evidence_ids:
            raise ValueError(f"{self.item_id}: trap items must have empty gold_evidence_ids (nothing IS the gold answer)")

    def to_dict(self):
        return asdict(self)


def validate_items(items: list) -> list:
    """Returns a list of problems across the whole set (empty if clean).
    Checked once at benchmark-build time, not per-item at eval time."""
    problems = []
    ids_seen = set()
    for item in items:
        if item.item_id in ids_seen:
            problems.append(f"duplicate item_id: {item.item_id}")
        ids_seen.add(item.item_id)
        if item.slice != "trap_qa" and not item.gold_evidence_ids:
            problems.append(f"{item.item_id}: non-trap item has no gold_evidence_ids")
    return problems
