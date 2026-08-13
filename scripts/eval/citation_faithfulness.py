"""
Citation Precision/Recall at the PROPOSITION level, per Novelty Claim #3:
"each atomic legal claim in an answer is aligned to a retrieved paragraph
and verified" -- not answer-level faithfulness, per-sentence.

Proposal 6.1 specifies "claim decomposition + small CPU-runnable NLI model
(e.g. DeBERTa-v3-small)". This module does claim decomposition (sentence
splitting on the citation markers [Doc N] the generation prompt already
requires -- see generate.py's SYSTEM_PROMPT) for free, since every
sentence must already carry its citation by construction. What it does
NOT do is real NLI: loading DeBERTa-v3-small needs a HuggingFace model
download, and this sandbox cannot reach huggingface.co (see config.py's
top docstring). In its place, `_lexical_entailment_proxy` below is a
transparent, auditable token-overlap heuristic -- MUCH weaker than a
trained NLI model at catching subtle unsupported claims (e.g. negation
flips, hedged claims presented as certain), and every score this module
produces should be labeled "proxy" in any report, never presented as the
paper's real Citation Precision/Recall number.

SWAP POINT: replace `_lexical_entailment_proxy` with a real DeBERTa-v3-small
NLI call (premise=cited paragraph, hypothesis=claim sentence, entailment
score) once model weights are reachable -- the claim-decomposition and
aggregation logic around it does not need to change.

Run standalone for a quick manual test:
    python scripts/eval/citation_faithfulness.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CITATION_MARKER_RE = re.compile(r"\[Doc\s+(\d+)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "this", "that", "these",
    "those", "for", "of", "to", "in", "on", "at", "by", "with", "and",
    "or", "but", "if", "then", "as", "it", "its", "be", "been", "being",
    "do", "does", "did", "has", "have", "had", "not",
}


def decompose_claims(answer_text: str) -> list:
    """Splits the generator's answer into (claim_sentence, [doc_numbers])
    pairs, using the [Doc N] markers the generation prompt mandates."""
    sentences = SENTENCE_SPLIT_RE.split(answer_text)
    claims = []
    for sent in sentences:
        doc_nums = [int(n) for n in CITATION_MARKER_RE.findall(sent)]
        clean_sent = CITATION_MARKER_RE.sub("", sent).strip()
        if clean_sent:
            claims.append((clean_sent, doc_nums))
    return claims


def _tokenize(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


def _lexical_entailment_proxy(claim: str, cited_text: str) -> bool:
    """PROXY, not real NLI -- see module docstring. A claim is judged
    'supported' if a majority of its non-stopword tokens appear in the
    cited passage. This will over-credit claims that share vocabulary
    with the source but invert its meaning (e.g. negation), and
    under-credit claims that are correctly paraphrased with different
    words -- both are known, documented failure modes of a lexical proxy,
    not edge cases this implementation happens to miss."""
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return True  # nothing substantive to verify (e.g. a pure connective sentence)
    source_tokens = _tokenize(cited_text)
    overlap = len(claim_tokens & source_tokens)
    return (overlap / len(claim_tokens)) >= 0.5


def score_citation_faithfulness(answer_text: str, doc_number_to_text: dict) -> dict:
    """Returns proposition-level Citation Precision and Recall for one
    answer. Precision = of the claims that cited something, how many
    citations were actually supported by that doc. Recall here is scoped
    to "citation completeness": how many claims cited at least one doc at
    all (an uncited factual claim is a recall failure -- the proposition
    exists but points to nothing verifiable)."""
    claims = decompose_claims(answer_text)
    if not claims:
        return {"precision": float("nan"), "recall": float("nan"), "n_claims": 0}

    cited_claims = [(c, docs) for c, docs in claims if docs]
    supported = 0
    for claim, doc_nums in cited_claims:
        cited_text = " ".join(doc_number_to_text.get(n, "") for n in doc_nums)
        if _lexical_entailment_proxy(claim, cited_text):
            supported += 1

    precision = supported / len(cited_claims) if cited_claims else float("nan")
    recall = len(cited_claims) / len(claims)

    return {"precision": precision, "recall": recall, "n_claims": len(claims), "n_cited_claims": len(cited_claims)}


def main():
    # Worked example, mirrors generate.py's output_contract shape.
    answer = (
        "IPC Section 420 was repealed w.e.f. 01-07-2024 [Doc 1]. "
        "The corresponding provision is BNS Section 318(4) [Doc 2]. "
        "This is a general statement with no citation."
    )
    doc_texts = {
        1: "IPC Section 420 was repealed with effect from 1 July 2024.",
        2: "BNS Section 318(4) is the successor provision for cheating.",
    }
    result = score_citation_faithfulness(answer, doc_texts)
    print("Worked example (PROXY scoring, not real NLI -- see module docstring):")
    print(result)


if __name__ == "__main__":
    main()
