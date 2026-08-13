"""
Central config for everything added on top of the original statute-only
pipeline (precedent layer, hybrid retrieval, NyayaBench, evaluation).

Kept deliberately separate from era_filter.py / retrieve.py so the original
statute pipeline keeps working completely unchanged -- nothing below is
imported by the pre-existing scripts.

WHERE THIS DIVERGES FROM THE PROPOSAL, AND WHY (read this before running
anything in scripts/retrieval or scripts/eval):

  This sandbox cannot reach ncrb.gov.in, indiacode.nic.in, HuggingFace, or
  any embedding/LLM API host other than Groq's, and has no API keys
  configured for OpenAI/Cohere/Voyage. So:
    - data/raw/judgments/sample_judgments.json is a small HAND-AUTHORED
      set of realistic-but-synthetic judgment records (not a real corpus)
      standing in for the 5k-8k judgment corpus the proposal specifies.
      ingest_judgments.py reads real data too, if you point
      JUDGMENTS_RAW_DIR at a real download -- the schema is documented
      there.
    - EMBEDDING_API_PROVIDER defaults to "offline", a deterministic hash
      embedding with NO semantic meaning -- it exists only so the dense/
      hybrid retrieval code path is exercised end-to-end without an API
      key. Swap in a real key (see below) before trusting any dense or
      hybrid retrieval NUMBER.
    - GENERATION_API_PROVIDER defaults to "groq", matching the existing
      generate.py. A second hosted model for the open-vs-closed ablation
      axis is left as a config slot, unset by default.
"""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

JUDGMENTS_RAW_DIR = DATA_RAW / "judgments"
JUDGMENTS_PROCESSED_DIR = DATA_PROCESSED / "judgments"
NYAYABENCH_DIR = DATA_PROCESSED / "nyayabench"
EVAL_DIR = DATA_PROCESSED / "eval"

for d in (JUDGMENTS_PROCESSED_DIR, NYAYABENCH_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Same date used by era_filter.py -- repeated here (not imported) so this
# file has zero dependency on the original scripts and can be read/reused
# standalone.
BNS_COMMENCEMENT_DATE = date(2024, 7, 1)

# --- Embedding API (dense retrieval) ---------------------------------------
# "offline" = deterministic hash embedding, no network/key required, no
#             semantic meaning -- pipeline-plumbing only, see module docstring.
# "openai"  / "cohere" / "voyage" = real hosted embedding APIs. Set
#             EMBEDDING_API_KEY and EMBEDDING_MODEL in .env to use one.
EMBEDDING_API_PROVIDER = os.environ.get("EMBEDDING_API_PROVIDER", "offline")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "256"))  # small on purpose for the offline mode; real APIs override via their own response shape

# --- Generation API (unchanged default: Groq, matches generate.py) --------
GENERATION_API_PROVIDER = os.environ.get("GENERATION_API_PROVIDER", "groq")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# Second hosted model slot for the open-vs-closed generator ablation axis
# (Section 4.3 / 4.4 of the proposal). Left unset by default -- fill in a
# second Groq-hosted model name (e.g. "llama-3.1-8b-instant") or wire up
# another provider's client in scripts/retrieval / scripts/eval as needed.
SECONDARY_GENERATION_MODEL = os.environ.get("SECONDARY_GENERATION_MODEL")

# --- Reranking ---------------------------------------------------------
# "cross_encoder" = local CPU cross-encoder (sentence-transformers,
#                    ms-marco-MiniLM-L-6-v2) over the top RERANK_POOL_SIZE
#                    candidates, per proposal 4.2(b). Falls back to "off"
#                    automatically if sentence-transformers isn't installed.
# "off"           = skip reranking, hybrid fusion order stands.
RERANK_MODE = os.environ.get("RERANK_MODE", "cross_encoder")
RERANK_POOL_SIZE = int(os.environ.get("RERANK_POOL_SIZE", "20"))

# --- Ablation grid budget caps (Section 6.3) --------------------------
MAX_ABLATION_CONFIGS = int(os.environ.get("MAX_ABLATION_CONFIGS", "36"))
