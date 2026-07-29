"""
Run a SAMPLE of your QA items through the full retrieve + generate pipeline,
and check whether the LLM's cited document actually corresponds to the gold
section. This is a citation-accuracy check, not a legal-correctness check --
we verify the citation number points to the right retrieved document, not
whether the LLM's prose is legally sound (that would need a human/expert
rater, per the original proposal's evaluation protocol).

Deliberately samples instead of running all 948 items: Groq's free tier has
rate limits, and a sample of ~30-50 is enough to see whether generation is
working correctly before committing your whole quota to a full run.

Run from the project root, after generate.py's GROQ_API_KEY is set:
    python scripts/run_generation_eval.py [sample_size]

Output:
    data/processed/generation_eval_results.json
"""
import json
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

from generate import Generator

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DEFAULT_SAMPLE_SIZE = 30
CITATION_PATTERN = re.compile(r"\[Doc (\d+)\]")


def normalize(s: str) -> str:
    return s.replace(" ", "").upper().split("(")[0]


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SAMPLE_SIZE

    qa_items = json.loads((PROCESSED_DIR / "qa_items.json").read_text())
    sample = random.sample(qa_items, min(sample_size, len(qa_items)))

    gen = Generator()
    results = []

    for i, item in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] {item['id']} ...")
        query_date = date.fromisoformat(item["query_date"])
        gold_statute = item["gold_statute"]
        gold_section = normalize(item["gold_section_no"])

        try:
            result = gen.answer(item["query"], query_date, mode="validity_aware")
        except Exception as exc:
            print(f"  [warn] generation failed: {exc}")
            time.sleep(2)
            continue

        cited_doc_numbers = {int(n) for n in CITATION_PATTERN.findall(result["answer"])}
        docs_by_n = {d["n"]: d for d in result["docs"]}

        cited_correct_gold = any(
            docs_by_n[n]["statute"] == gold_statute and normalize(docs_by_n[n]["section_no"]) == gold_section
            for n in cited_doc_numbers
            if n in docs_by_n
        )
        made_no_citation = len(cited_doc_numbers) == 0
        abstained = "do not contain an answer" in result["answer"].lower()

        # If generate.py computed a validity notice (a repealed/replaced
        # section is involved), check whether the LLM's answer actually
        # surfaced it -- crude proxy: does the answer mention the specific
        # section number the notice named? This tests the part of the
        # output contract that's a guaranteed system fact, not just
        # something we hope the model remembers to say.
        validity_notice = result.get("validity_notice", "")
        notice_expected = bool(validity_notice)
        notice_surfaced = False
        if notice_expected:
            # Check for the notice's actual signal language ("repealed" /
            # "not yet in force"), not just any section number it mentions --
            # an answer can cite the OLD section normally (as part of a
            # correct citation) without ever communicating that it's
            # repealed, and checking for the number alone gave false
            # positives on exactly that case (confirmed while testing this
            # script offline before shipping it).
            answer_lower = result["answer"].lower()
            notice_surfaced = "repeal" in answer_lower or "not yet in force" in answer_lower or "not in force" in answer_lower

        results.append({
            "id": item["id"],
            "query": item["query"],
            "gold": f"{gold_statute} {item['gold_section_no']}",
            "answer": result["answer"],
            "retrieved_docs": result["docs"],
            "cited_doc_numbers": sorted(cited_doc_numbers),
            "cited_correct_gold": cited_correct_gold,
            "made_no_citation": made_no_citation,
            "abstained": abstained,
            "validity_notice": validity_notice,
            "notice_expected": notice_expected,
            "notice_surfaced_in_answer": notice_surfaced,
        })

        time.sleep(2.5)  # base spacing between calls; call_groq() retries with backoff on top of this if still rate limited

    n = len(results)
    if n == 0:
        print("[error] No results -- all generations failed. Check your API key and connection.")
        sys.exit(1)

    correct = sum(r["cited_correct_gold"] for r in results)
    no_citation = sum(r["made_no_citation"] for r in results)
    abstained = sum(r["abstained"] for r in results)
    notices_expected = [r for r in results if r["notice_expected"]]
    notices_surfaced = sum(r["notice_surfaced_in_answer"] for r in notices_expected)

    print(f"\nSampled {n} items:")
    print(f"  Cited the correct gold document: {correct}/{n} ({correct/n:.1%})")
    print(f"  Gave no citation at all: {no_citation}/{n} ({no_citation/n:.1%})")
    print(f"  Abstained (said docs don't answer it): {abstained}/{n} ({abstained/n:.1%})")
    if notices_expected:
        print(
            f"  Validity notice was expected on {len(notices_expected)} items; "
            f"LLM actually surfaced it in {notices_surfaced}/{len(notices_expected)} "
            f"({notices_surfaced/len(notices_expected):.1%})"
        )
    else:
        print("  No items in this sample triggered a validity notice (try a larger sample).")

    (PROCESSED_DIR / "generation_eval_results.json").write_text(
        json.dumps({
            "sample_size": n,
            "citation_accuracy": correct / n,
            "no_citation_rate": no_citation / n,
            "abstention_rate": abstained / n,
            "notice_surfacing_rate": (notices_surfaced / len(notices_expected)) if notices_expected else None,
            "results": results,
        }, indent=2, ensure_ascii=False)
    )
    print(f"\nFull results -> data/processed/generation_eval_results.json")


if __name__ == "__main__":
    main()