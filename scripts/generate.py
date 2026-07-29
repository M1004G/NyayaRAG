"""
Generation layer: retrieval (validity-aware) + LLM answer with mandatory
citations AND guaranteed validity notices, matching the proposal's output
contract: "answer + cited paragraphs + validity notices (e.g., 'IPC section
420 repealed w.e.f. 01-07-2024; corresponding provision BNS section 318(4)')."

Uses Groq's free API (OpenAI-compatible request/response format, serving
Llama models).

Get a free API key: https://console.groq.com/keys

Set it EITHER as an environment variable:
    export GROQ_API_KEY=gsk_...        (macOS/Linux)
    setx GROQ_API_KEY "gsk_..."        (Windows, then restart terminal)

OR (simpler, recommended) create a file named .env in the project root
(same folder as requirements.txt) containing one line:
    GROQ_API_KEY=gsk_your_key_here
This gets loaded automatically -- no terminal restart needed. IMPORTANT:
add ".env" to a .gitignore file so you never accidentally commit your key
if you put this project under version control.

Run standalone for a quick manual test:
    python scripts/generate.py "What is the punishment for theft?" 2025-01-01

Used as a module by run_generation_eval.py for batch evaluation.

KNOWN LIMITATION vs. the proposal: citations here are DOCUMENT-level
([Doc N]), not paragraph-level ([Doc i, para j]) as the proposal specifies,
because the corpus stores each section as a single text block rather than
paragraph-segmented. This matches your architecture doc's own phasing
though -- the full Citation Precision/Recall metric (which needs paragraph
granularity) was already deferred to Phase 3, after the thin-slice validity
result. Worth stating explicitly as a scoped limitation, not silently
presented as full parity with the proposal.
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

from retrieve import Retriever
from era_filter import BNS_COMMENCEMENT_DATE, valid_statute_for_date

load_dotenv()  # reads .env in the project root if present; harmless no-op if not

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

API_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile: best quality Groq offers free, good for a
# citation-following task like this. llama-3.1-8b-instant is faster/cheaper
# per call if you're running a large batch eval and want to trade quality
# for speed -- swap this if you hit rate limits on the 70b model.
MODEL = "llama-3.3-70b-versatile"

TOP_K_FOR_GENERATION = 5

SYSTEM_PROMPT = """You are a legal research assistant answering questions about \
Indian criminal law. You will be given numbered source documents (statute \
sections), possibly some SYSTEM-VERIFIED VALIDITY NOTICES, and a question \
with a specific date.

RULES -- follow exactly:
1. Answer using ONLY the information in the numbered documents below. Do not \
use outside knowledge of Indian law.
2. Every factual claim in your answer MUST be followed by a citation in the \
form [Doc N], where N is the document number it came from.
3. If any SYSTEM-VERIFIED VALIDITY NOTICES are provided, you MUST state them \
plainly in your answer -- they are verified facts, not suggestions, and you \
must never contradict them.
4. If none of the documents actually answer the question, say so explicitly \
("The provided documents do not contain an answer to this question") instead \
of guessing. Do not fabricate a citation.
5. Be concise -- a few sentences, not an essay."""


def load_supersession_map() -> dict:
    """IPC section -> its current BNS replacement, from the trusted mapping
    table -- used to generate notices like the proposal's own example:
    'IPC 420 repealed w.e.f. 01-07-2024; corresponding provision BNS 318(4)'."""
    path = PROCESSED_DIR / "section_mapping.json"
    if not path.exists():
        return {}
    mapping = json.loads(path.read_text())
    return {m["ipc_section"].replace(" ", ""): m["bns_section"] for m in mapping}


def build_validity_notice(query_date: date, docs: list, supersession_map: dict) -> str:
    """Computed the way era_filter.py already knows the answer -- not left
    to the LLM's judgment:
    1. Safety notice: flags if a wrong-era document made it into the
       retrieved set at all (e.g. naive mode, or a validity-aware near-miss).
    2. Transparency notice (the proposal's literal example): whenever an IPC
       section is cited, states its current BNS replacement, regardless of
       whether IPC was the era-correct choice for this query."""
    valid_statute = valid_statute_for_date(query_date)
    notices = []
    for doc in docs:
        if doc["statute"] != valid_statute:
            notices.append(
                f"{doc['statute']} Section {doc['section_no']} was "
                f"{'repealed' if doc['statute'] == 'IPC' else 'not yet in force'} "
                f"as of {query_date.isoformat()} (BNS commencement date: "
                f"{BNS_COMMENCEMENT_DATE.isoformat()})."
            )
        elif doc["statute"] == "IPC":
            replacement = supersession_map.get(doc["section_no"].replace(" ", ""))
            if replacement:
                notices.append(
                    f"IPC Section {doc['section_no']} was repealed w.e.f. "
                    f"{BNS_COMMENCEMENT_DATE.isoformat()}; corresponding current "
                    f"provision is BNS Section {replacement}."
                )
    return "\n".join(notices)


def call_groq(system: str, user_message: str, api_key: str, max_retries: int = 5) -> str:
    """Retries on 429 (rate limit) with exponential backoff. Confirmed
    necessary: a first real batch run lost 13/30 calls to untreated 429s
    with only a flat 1-second delay between requests -- that delay alone
    isn't enough for Groq's free-tier limits, so failed calls need to
    actually retry rather than just being dropped."""
    delay = 3
    for attempt in range(max_retries):
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            },
            timeout=30,
        )
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            print(f"  [rate limited, retrying in {wait:.1f}s -- attempt {attempt + 1}/{max_retries}]")
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"Gave up after {max_retries} retries -- still rate limited.")


class Generator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise SystemExit(
                "[error] GROQ_API_KEY not set. Get a free key at "
                "https://console.groq.com/keys, then see this file's "
                "docstring for how to set it as an environment variable."
            )
        self.retriever = Retriever()
        self.supersession_map = load_supersession_map()

    def build_prompt(self, query: str, docs: list, query_date: date) -> str:
        doc_blocks = []
        for i, doc in enumerate(docs, 1):
            doc_blocks.append(f"[Doc {i}] {doc['statute']} Section {doc['section_no']}\n{doc['text']}")
        docs_text = "\n\n".join(doc_blocks)

        notice = build_validity_notice(query_date, docs, self.supersession_map)
        validity_section = (
            f"\n\nSYSTEM-VERIFIED VALIDITY NOTICES:\n{notice}" if notice else ""
        )

        return f"Documents:\n\n{docs_text}{validity_section}\n\nQuestion: {query}"

    def answer(self, query: str, query_date: date, mode: str = "validity_aware", top_k: int = TOP_K_FOR_GENERATION):
        results = self.retriever.retrieve(query, query_date, mode=mode, top_k=top_k)
        docs = [doc for doc, score in results]

        if not docs:
            return {"answer": "No relevant documents found.", "docs": [], "validity_notice": ""}

        notice = build_validity_notice(query_date, docs, self.supersession_map)
        prompt = self.build_prompt(query, docs, query_date)
        answer_text = call_groq(SYSTEM_PROMPT, prompt, self.api_key)

        return {
            "answer": answer_text,
            "docs": [{"n": i + 1, "statute": d["statute"], "section_no": d["section_no"]} for i, d in enumerate(docs)],
            "validity_notice": notice,
        }


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/generate.py \"<question>\" <YYYY-MM-DD> [naive|validity_aware]")
        sys.exit(1)

    query = sys.argv[1]
    query_date = date.fromisoformat(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "validity_aware"

    gen = Generator()
    result = gen.answer(query, query_date, mode=mode)

    print(f"Question: {query}")
    print(f"Date: {query_date}  Mode: {mode}\n")
    print("Retrieved documents:")
    for d in result["docs"]:
        print(f"  [Doc {d['n']}] {d['statute']} Section {d['section_no']}")
    if result["validity_notice"]:
        print(f"\nSystem-verified validity notice(s):\n{result['validity_notice']}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()