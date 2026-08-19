"""
Citation Precision/Recall at the PROPOSITION level, per Novelty Claim #3:
"each atomic legal claim in an answer is aligned to a retrieved paragraph
and verified" -- not answer-level faithfulness, per-sentence.

Proposal 6.1 specifies "claim decomposition + small CPU-runnable NLI model
(e.g. DeBERTa-v3-small)". This module does claim decomposition (sentence
splitting on the citation markers [Doc N] the generation prompt already
requires -- see generate.py's SYSTEM_PROMPT) for free, since every
sentence must already carry its citation by construction.

Two scoring backends, same pattern as retrieval/rerank.py:
  - Real NLI (`_nli_entailment`): cross-encoder/nli-deberta-v3-small via
    `transformers`, matching the proposal's model choice. Used
    automatically if `transformers` + model weights are available.
  - Lexical proxy (`_lexical_entailment_proxy`): transparent token-overlap
    heuristic, used as a fallback ONLY when the real model can't be
    loaded (no `transformers` install, or no network to fetch weights).
    MUCH weaker at catching subtle unsupported claims (negation flips,
    hedged claims presented as certain) -- any score produced this way is
    labeled "proxy" in the result dict, and should never be reported as
    the paper's real Citation Precision/Recall number.

Which backend actually ran is always in the returned dict's "backend"
key ("nli" or "lexical_proxy") -- check this before trusting a number,
don't assume from context.

Run standalone for a quick manual test:
    python scripts/eval/citation_faithfulness.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"  # matches proposal 6.1's named model family

try:
    from sentence_transformers import CrossEncoder
    _nli_model = None  # lazy-loaded on first real use, not at import time -- importing this module shouldn't trigger a model download

    def _get_nli_model():
        global _nli_model
        if _nli_model is None:
            _nli_model = CrossEncoder(NLI_MODEL_NAME)
        return _nli_model
    _NLI_LIBS_AVAILABLE = True
except ImportError:
    _NLI_LIBS_AVAILABLE = False

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


def _nli_entailment(claim: str, cited_text: str) -> bool:
    """Real NLI: premise=cited passage, hypothesis=claim. Returns True if
    the entailment score beats contradiction/neutral. Truncates the
    premise to a reasonable length -- these are short statute/judgment
    chunks by construction (see build_dense_index.py), so this rarely
    matters, but a defensive cap avoids an unbounded call if it's ever
    fed a longer passage."""
    model = _get_nli_model()
    premise = cited_text[:2000]
    scores = model.predict([(premise, claim)])[0]  # [contradiction, entailment, neutral] for this model family
    contradiction, entailment, neutral = scores
    return entailment > max(contradiction, neutral)


def score_citation_faithfulness(answer_text: str, doc_number_to_text: dict) -> dict:
    """Returns proposition-level Citation Precision and Recall for one
    answer. Precision = of the claims that cited something, how many
    citations were actually supported by that doc. Recall here is scoped
    to "citation completeness": how many claims cited at least one doc at
    all (an uncited factual claim is a recall failure -- the proposition
    exists but points to nothing verifiable).

    Uses real NLI if transformers/model weights are available, otherwise
    falls back to the lexical proxy -- check the returned "backend" key,
    never assume which one ran."""
    claims = decompose_claims(answer_text)
    if not claims:
        return {"precision": float("nan"), "recall": float("nan"), "n_claims": 0, "backend": "n/a"}

    backend = "lexical_proxy"
    entailment_fn = _lexical_entailment_proxy
    if _NLI_LIBS_AVAILABLE:
        try:
            _get_nli_model()  # trigger the (possibly network-dependent) load here, fail fast before scoring anything
            backend = "nli"
            entailment_fn = _nli_entailment
        except Exception as e:
            print(f"[warn] Could not load {NLI_MODEL_NAME} ({e}); falling back to lexical proxy for this run.")

    cited_claims = [(c, docs) for c, docs in claims if docs]
    supported = 0
    for claim, doc_nums in cited_claims:
        cited_text = " ".join(doc_number_to_text.get(n, "") for n in doc_nums)
        if entailment_fn(claim, cited_text):
            supported += 1

    precision = supported / len(cited_claims) if cited_claims else float("nan")
    recall = len(cited_claims) / len(claims)

    return {"precision": precision, "recall": recall, "n_claims": len(claims),
            "n_cited_claims": len(cited_claims), "backend": backend}


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
    print(f"Worked example -- backend actually used: {result['backend']}")
    print(result)


if __name__ == "__main__":
    main()
