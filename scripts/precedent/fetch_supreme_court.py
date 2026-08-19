"""
Processes Indian Supreme Court judgments from the bulk tar+parquet source
(s3://indian-supreme-court-judgments), which is structurally much simpler
than the High Court dataset this repo used first: ONE tar file per YEAR
(not per court x bench x year), containing every English judgment for
that year already bundled -- a single large download instead of hundreds
of small per-file operations. That matters in practice: every large
single-file S3 transfer this session (metadata syncs) was fast and
reliable; the High Court path's per-file downloads were the actual
source of most of the pain (see fetch_real_judgments.py's docstring and
git history for the full diagnosis trail).

Also a better fit for the proposal's own emphasis: Supreme Court
judgments are what create binding national precedent, more central to
the overruling-graph work than scattered High Court benches, and
downloading 2023/2024/2025 separately gives a clean pre-BNS /
transition / post-BNS split that the High Court sampling never cleanly
achieved.

PowerShell steps (see this module's own printed instructions too):
    aws s3 ls --no-sign-request s3://indian-supreme-court-judgments/data/tar/
    aws s3 cp --no-sign-request s3://indian-supreme-court-judgments/data/tar/year=2024/english/english.tar .
    aws s3 cp --no-sign-request s3://indian-supreme-court-judgments/metadata/parquet/year=2024/metadata.parquet .
    tar -xf english.tar -C sc_extracted_2024
    python scripts\\precedent\\fetch_supreme_court.py --year 2024 --tar-dir sc_extracted_2024 --parquet metadata.parquet

Metadata columns (per the dataset's own README, different from the High
Court dataset -- NOT reusing that script's COLUMN_MAP):
    title, petitioner, respondent, description, judge, author_judge,
    citation, case_id, cnr, decision_date, disposal_nature, court,
    available_languages, raw_html, path, nc_display, scraped_at, year
No is_final / judicial_section columns exist here -- Supreme Court
reported judgments don't need the same interim-order/civil-vs-criminal
filtering the High Court data needed; every row is already a disposed
Supreme Court matter. Sections-cited detection still does the real
filtering work (civil/constitutional/tax matters simply won't match the
IPC/BNS/CrPC/BNSS patterns and get skipped, same as before).
"""
import argparse
import json
import multiprocessing
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_real_judgments import (  # noqa: E402 -- reusing the same, already-tested regex/filter logic AND the process-based extraction timeout fix, rather than duplicating either
    extract_sections_cited, guess_disposal, is_likely_sensitive_category,
    extract_text_bounded, extract_texts_bounded_concurrent,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW  # noqa: E402


def build_pdf_index(tar_dir: Path) -> dict:
    """Called ONCE before the main loop, not per-item. The previous
    version called tar_dir.rglob() up to 3 times PER ITEM inside
    find_local_pdf whenever the direct filename guess missed -- each
    call walks the whole directory tree with no timeout at all, and one
    of those walks stalling indefinitely (observed: item 14/782 hung
    with no completing line and no [slow] flag from either of the OTHER
    two, properly-bounded checks, meaning the hang was here, not in
    extraction) explains the 5-6 minute freeze with zero diagnostic
    output. A single upfront index turns every lookup into an instant
    dict access -- no repeated filesystem walks, no way for a lookup
    itself to hang."""
    return {p.name: p for p in tar_dir.rglob("*.pdf")}


def normalize_date(raw_value) -> str:
    """The Supreme Court dataset's decision_date column is DD-MM-YYYY
    text (e.g. '25-09-2024'), confirmed against real ingest_rejected.json
    output -- ALL 207 kept judgments were silently rejected downstream
    because the old code assumed it was already ISO-formatted (true for
    the High Court dataset, false here) and just took the first 10
    characters as-is. Tries ISO first, then DD-MM-YYYY, so this is safe
    even if a future year mixes conventions; returns the original string
    unchanged (and lets ingest_judgments.py's own validator catch it) if
    neither parses -- never silently drops or guesses a date."""
    s = str(raw_value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s[:10] if fmt == "%Y-%m-%d" else s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def find_local_pdf(pdf_index: dict, path_value: str) -> Path:
    """Confirmed pattern (real data, year=2024): parquet 'path' column
    is a bare stem like '2024_10_108_125'; actual extracted filename is
    that plus '_EN.pdf'. Falls back to a couple of other plausible
    variants via the SAME index (still O(1) each, never a fresh scan)."""
    stem = str(path_value)
    for candidate_name in (f"{stem}_EN.pdf", stem, f"{stem}.pdf", Path(stem).name):
        if candidate_name in pdf_index:
            return pdf_index[candidate_name]
    return None


def process_year(year: int, tar_dir: Path, parquet_path: Path, n: int = None,
                  include_sensitive: bool = False):
    if not parquet_path.exists():
        raise SystemExit(f"[error] {parquet_path} not found -- download it first (see module docstring).")
    if not tar_dir.exists():
        raise SystemExit(f"[error] {tar_dir} not found -- extract the tar first (see module docstring).")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows of Supreme Court metadata for year={year}")

    # available_languages likely includes non-English entries even in the
    # "english" tar's matching metadata rows in some edge cases (regional
    # translations of the same judgment) -- if the column exists, prefer
    # rows that actually list English, but don't hard-fail if the column
    # or expected values look different than assumed, since this is a
    # best-effort narrowing, not a correctness-critical filter (a wrongly
    # -kept row just fails to find its PDF file and gets skipped below).
    if "available_languages" in df.columns:
        try:
            mask = df["available_languages"].astype(str).str.contains("english", case=False, na=False)
            if mask.any():
                df = df[mask]
        except Exception:
            pass

    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=42)
    print(f"Processing {len(df)} row(s)")

    print("Indexing local PDF filenames (one-time directory scan)...", flush=True)
    pdf_index = build_pdf_index(tar_dir)
    print(f"Indexed {len(pdf_index)} local PDF file(s)")

    records = []
    skipped = {"missing_local_file": 0, "text_extract_failed": 0, "too_short": 0,
               "no_sections_cited": 0, "sensitive_category_excluded": 0}
    bench_start = time.time()

    # PASS 1: resolve every row's local PDF path -- pure dict lookups,
    # no filesystem I/O, effectively instant even for thousands of rows.
    row_by_path = {}
    for row in df.itertuples():
        path_value = getattr(row, "path", None) or getattr(row, "nc_display", None)
        if not path_value:
            skipped["missing_local_file"] += 1
            continue
        pdf_path = find_local_pdf(pdf_index, path_value)
        if pdf_path is None:
            skipped["missing_local_file"] += 1
            continue
        row_by_path[pdf_path] = row
    print(f"Resolved {len(row_by_path)} local PDF path(s), {skipped['missing_local_file']} missing")

    # PASS 2: extract text from all resolved PDFs CONCURRENTLY instead of
    # one at a time. This is the actual fix for a slow corpus: a real
    # 150-page judgment was measured taking the full 45s timeout on its
    # own (genuinely slow, not a bug) -- sequential processing means
    # every such file pays its full cost serially (two back-to-back 45s
    # timeouts were observed on a real run). With 4 concurrent workers, a
    # slow file only occupies one of four slots while the other three
    # keep making progress -- confirmed ~2x+ real speedup on a
    # slow+fast mix in testing, and it scales further with more slow
    # files or more workers.
    #
    # CHECKPOINTED: extraction is the slow part (782 files took ~38
    # minutes in a real run) and previously lived only in memory until
    # the whole run finished -- a crash, an interrupt, or ANY hang (even
    # an already-fixed one) meant total loss. Now: results are written
    # to a checkpoint file every 20 completions, and reloaded on startup
    # so an interrupted run resumes instead of restarting from zero.
    checkpoint_path = DATA_RAW / "judgments" / f"_extraction_checkpoint_{year}.json"
    text_results = {}
    if checkpoint_path.exists():
        try:
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            text_results = {Path(k): v for k, v in raw.items()}
            print(f"Resuming from checkpoint: {len(text_results)} PDF(s) already extracted -> {checkpoint_path}")
        except Exception as e:
            print(f"[warn] Could not read checkpoint ({e}), starting extraction fresh.")

    already_done = set(text_results.keys())
    pending_paths = [p for p in row_by_path.keys() if p not in already_done]
    print(f"{len(pending_paths)} PDF(s) still need extraction ({len(already_done)} resumed from checkpoint)")

    def _save_checkpoint():
        serializable = {str(k): v for k, v in text_results.items()}
        checkpoint_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")

    def _progress(done, total, partial_results):
        # partial_results is the SAME dict extract_texts_bounded_concurrent
        # is building internally (passed by reference) -- merge it into
        # OUR persistent text_results (which may already have checkpoint-
        # resumed entries) on every callback, not just at the end, so the
        # checkpoint save below actually has something new to write.
        text_results.update(partial_results)
        if done % 10 == 0 or done == total:
            elapsed = time.time() - bench_start
            print(f"  extracted {done}/{total}  [{elapsed:.0f}s elapsed]")
        if done % 20 == 0 or done == total:
            _save_checkpoint()

    if pending_paths:
        print(f"Extracting text from {len(pending_paths)} PDFs (4 concurrent workers)...", flush=True)
        extract_texts_bounded_concurrent(
            pending_paths, timeout=45, max_workers=4, progress_callback=_progress
        )
        _save_checkpoint()
    else:
        print("All PDFs already extracted (from checkpoint) -- skipping straight to filtering.")

    # PASS 3: apply citation extraction / filters to whatever text came
    # back -- fast, CPU-light, no more waiting on anything.
    for i, (pdf_path, row) in enumerate(row_by_path.items(), start=1):
        text = text_results.get(pdf_path)
        if text is None:
            skipped["text_extract_failed"] += 1
            continue

        if len(text.strip()) < 500:
            skipped["too_short"] += 1
            continue

        title = str(getattr(row, "title", "") or "")
        petitioner = str(getattr(row, "petitioner", "") or "")
        respondent = str(getattr(row, "respondent", "") or "")
        case_name = title or (f"{petitioner} v. {respondent}" if petitioner and respondent else f"Case {i}")

        if not include_sensitive and is_likely_sensitive_category(case_name, text):
            skipped["sensitive_category_excluded"] += 1
            continue

        sections = extract_sections_cited(text)
        if not sections:
            skipped["no_sections_cited"] += 1
            continue

        judge_raw = str(getattr(row, "judge", "") or getattr(row, "author_judge", "") or "")
        bench_names = [j.strip() for j in judge_raw.split(",") if j.strip()]
        cnr = str(getattr(row, "cnr", "") or getattr(row, "case_id", "") or f"SC-{year}-{i}")
        date_val = normalize_date(getattr(row, "decision_date", ""))

        records.append({
            "judgment_id": f"SC-{cnr}",
            "case_name": case_name,
            "court": "Supreme Court of India",
            "date": date_val,
            "bench": bench_names,
            "sections_cited": sections,
            "text": text,
            "disposal": guess_disposal(text),
        })

        if i % 5 == 0:  # process-based extraction is now hard-bounded (see extract_text_bounded), so a coarser interval is safe again
            elapsed = time.time() - bench_start
            print(f"  processed {i}/{len(df)}  (kept {len(records)}, skipped {sum(skipped.values())})  [{elapsed:.0f}s elapsed]")

    out_path = DATA_RAW / "judgments" / f"real_supremecourt_{year}.json"
    payload = {
        "_NOTE": (
            f"REAL Supreme Court judgments, year={year}, from the bulk tar source "
            f"(s3://indian-supreme-court-judgments). {len(records)} of {len(df)} rows kept. "
            f"disposal is a keyword-heuristic guess (see guess_disposal in fetch_real_judgments.py), "
            f"not verified against the dataset's own disposal_nature field."
        ),
        "judgments": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKept {len(records)}/{len(df)} judgments -> {out_path}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--tar-dir", required=True, help="Directory the year's english.tar was extracted into")
    parser.add_argument("--parquet", required=True, help="Path to that year's metadata.parquet")
    parser.add_argument("--n", type=int, default=None, help="Optional: process only a random sample of N rows instead of the whole year")
    parser.add_argument("--include-sensitive", action="store_true")
    args = parser.parse_args()

    process_year(args.year, Path(args.tar_dir), Path(args.parquet), n=args.n,
                 include_sensitive=args.include_sensitive)


if __name__ == "__main__":
    main()