"""
Pluggable dense-embedding client. Proposal 4.2 specifies embeddings come
from a hosted API (OpenAI text-embedding-3-large, Cohere embed-v3, Voyage)
rather than a locally fine-tuned bi-encoder -- this module is that API
boundary, with THREE real providers wired plus one offline fallback.

READ THIS BEFORE TRUSTING ANY DENSE/HYBRID RETRIEVAL NUMBER:
EMBEDDING_API_PROVIDER defaults to "offline" (see config.py) because this
sandbox has no embedding API key configured. The offline embedder is a
deterministic hash-based bag-of-words vector -- it has NO semantic
understanding (synonyms, paraphrase, etc. will NOT be recognized as
similar). It exists ONLY so build_dense_index.py / hybrid_retrieve.py are
exercised end-to-end and are trivially swappable to a real provider by
setting three .env values -- it is not a stand-in for real retrieval
quality and any recall/nDCG numbers produced under "offline" mode should
be reported as pipeline-validity checks, not retrieval-quality results.

To use a real provider, set in .env:
    EMBEDDING_API_PROVIDER=openai      # or cohere / voyage
    EMBEDDING_API_KEY=...
    EMBEDDING_MODEL=text-embedding-3-large   # provider-appropriate model name

Run standalone for a quick manual test:
    python scripts/retrieval/embed_api.py "punishment for cheating"
"""
import hashlib
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    EMBEDDING_API_PROVIDER, EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM,
)


def _offline_embed(text: str, dim: int = EMBEDDING_DIM):
    """Deterministic hash-of-tokens bag-of-words vector. Same text always
    yields the same vector (needed for reproducible indexing); different
    text yields a decorrelated vector -- but with NO semantic content, see
    module docstring. Implementation: each token hashes to a dimension and
    a sign; vector is L2-normalized so cosine similarity is well-defined."""
    import re
    import math

    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _openai_embed(texts: list) -> list:
    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


def _cohere_embed(texts: list) -> list:
    resp = requests.post(
        "https://api.cohere.com/v1/embed",
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "texts": texts, "input_type": "search_document"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _voyage_embed(texts: list) -> list:
    resp = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]


PROVIDERS = {"openai": _openai_embed, "cohere": _cohere_embed, "voyage": _voyage_embed}


def embed_batch(texts: list) -> list:
    """Returns a list of embedding vectors, one per input text. Batches
    everything in a single API call for real providers (cheaper, fewer
    rate-limit hits) with no batching needed for the offline path."""
    if EMBEDDING_API_PROVIDER == "offline":
        return [_offline_embed(t) for t in texts]

    if EMBEDDING_API_PROVIDER not in PROVIDERS:
        raise SystemExit(
            f"[error] Unknown EMBEDDING_API_PROVIDER={EMBEDDING_API_PROVIDER!r}. "
            f"Use one of: offline, {', '.join(PROVIDERS)}"
        )
    if not EMBEDDING_API_KEY:
        raise SystemExit(
            f"[error] EMBEDDING_API_PROVIDER={EMBEDDING_API_PROVIDER!r} but "
            "EMBEDDING_API_KEY is not set in .env."
        )
    return PROVIDERS[EMBEDDING_API_PROVIDER](texts)


def embed_one(text: str) -> list:
    return embed_batch([text])[0]


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/retrieval/embed_api.py \"<text>\"")
        sys.exit(1)
    text = sys.argv[1]
    vec = embed_one(text)
    print(f"Provider: {EMBEDDING_API_PROVIDER}")
    print(f"Dim: {len(vec)}")
    print(f"First 8 components: {[round(v, 4) for v in vec[:8]]}")


if __name__ == "__main__":
    main()
