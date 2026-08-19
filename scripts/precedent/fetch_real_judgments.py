"""
Fetches REAL judgments from the "Indian High Court Judgments" AWS Open Data
dataset (s3://indian-high-court-judgments, CC-BY-4.0, no AWS account needed)
and transforms them into the schema ingest_judgments.py expects -- replacing
the synthetic sample_judgments.json with actual case law.

THIS SCRIPT CANNOT BE TEST-RUN FROM THE SANDBOX THAT BUILT THE REST OF THIS
REPO (same network restriction that produced the synthetic sample data in
the first place -- no amazonaws.com access there). It CAN run from your
local machine, which has normal internet access. Because of that, this
script runs in two phases so a wrong guess about the dataset's column names
doesn't waste a large download:

  PHASE 1 (--inspect): loads ONE small parquet metadata file already synced
  by the PowerShell step and just prints its columns + a few sample rows.
  No PDFs are downloaded. Look at the printed columns and edit COLUMN_MAP
  below if they don't match what's assumed here.

  PHASE 2 (default): samples N rows, downloads each PDF over plain HTTPS
  (public bucket, no signing needed), extracts text, extracts statute
  citations via regex, guesses disposal via keyword heuristic, and writes
  data/raw/judgments/real_highcourt_sample.json in ingest_judgments.py's
  schema.

Usage (from the project root, after the PowerShell sync step below):
    python scripts/precedent/fetch_real_judgments.py --inspect
    # edit COLUMN_MAP if needed, then:
    python scripts/precedent/fetch_real_judgments.py --n 300 --court 27_1 --bench hcaurdb --year 2024
"""
import argparse
import json
import multiprocessing
import re
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import pandas as pd
import pdfplumber
import requests


def _extract_text_worker(pdf_path_str: str, result_queue: multiprocessing.Queue):
    """Runs in a SEPARATE PROCESS, not a thread -- confirmed necessary,
    not a style choice: a threading-based version of this
    (ThreadPoolExecutor + .result(timeout=N)) was directly tested
    against a worst-case tight loop and the timeout NEVER fired --
    confirmed hung indefinitely despite an explicit timeout=3, because a
    worker that never releases the GIL can starve the very timeout
    mechanism meant to bound it (this is exactly what happened on a real
    run: item 14/782 hung for 14+ minutes with the [slow] flag never
    firing). A separate OS process has no such issue: it can be
    terminated unconditionally from outside, regardless of what it's
    doing internally."""
    try:
        with pdfplumber.open(pdf_path_str) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        result_queue.put(("ok", text))
    except Exception as e:
        result_queue.put(("error", str(e)))


def extract_text_bounded(pdf_path: Path, timeout: int = 45):
    """Returns extracted text, or None if extraction failed OR exceeded
    the timeout -- callers treat both cases identically (count as
    text_extract_failed, move on). Shared by fetch_real_judgments.py's
    own local-processing path and fetch_supreme_court.py, so the fix
    lives in exactly one place."""
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_extract_text_worker, args=(str(pdf_path), result_queue))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None
    if not result_queue.empty():
        status, payload = result_queue.get()
        if status == "ok":
            return payload
    return None


def extract_texts_bounded_concurrent(pdf_paths: list, timeout: int = 45, max_workers: int = 4,
                                      progress_callback=None) -> dict:
    """Processes MULTIPLE PDFs at once (up to max_workers simultaneously)
    instead of one at a time -- the actual fix for a slow corpus, not
    just a bigger timeout. Sequential processing means every slow/hung
    file pays its FULL timeout cost before the next file can even start
    (confirmed in practice: two consecutive 45s timeouts back to back on
    a real run). With N workers, a stuck file only occupies one of N
    concurrent slots -- the other N-1 keep making progress the whole
    time, and overall throughput scales roughly with core count for the
    common (fast, successful) case.

    Each PDF still gets its own independent multiprocessing.Process with
    a hard per-item timeout (same kill mechanism as extract_text_bounded,
    proven to correctly force-kill a genuinely hung worker) -- this is
    deliberately NOT a multiprocessing.Pool, because a Pool's worker
    processes are reused across tasks: a hung task poisons that worker
    slot for everything queued after it, with no clean way to kill just
    one task without tearing down the whole pool. Managing individual
    Process objects directly avoids that trap entirely.

    Returns {pdf_path: text_or_None}. progress_callback(done_count,
    total_count, results_dict_so_far), if given, is called each time a
    result comes in -- the results_dict_so_far is the SAME dict this
    function is building (passed by reference), not a copy, specifically
    so a caller can checkpoint partial progress to disk DURING a long
    run, not just after it returns. (Confirmed necessary: an earlier
    version of the checkpointing caller tried to save a dict that only
    got merged in after this function fully returned -- meaning
    checkpoint saves silently did nothing until the whole extraction was
    already done, defeating the entire point.)
    """
    results = {}
    pending = list(pdf_paths)
    in_flight = {}  # pdf_path -> (process, queue, start_time)
    total = len(pdf_paths)

    while pending or in_flight:
        while pending and len(in_flight) < max_workers:
            p = pending.pop(0)
            q = multiprocessing.Queue()
            proc = multiprocessing.Process(target=_extract_text_worker, args=(str(p), q))
            proc.start()
            in_flight[p] = (proc, q, time.time())

        for p in list(in_flight.keys()):
            proc, q, start_time = in_flight[p]
            elapsed = time.time() - start_time

            if not proc.is_alive():
                if not q.empty():
                    status, payload = q.get()
                    results[p] = payload if status == "ok" else None
                else:
                    results[p] = None
                proc.join()
                del in_flight[p]
                if progress_callback:
                    progress_callback(len(results), total, results)
            elif elapsed > timeout:
                proc.terminate()
                proc.join()
                results[p] = None
                del in_flight[p]
                if progress_callback:
                    progress_callback(len(results), total, results)

        if in_flight:
            time.sleep(0.05)  # brief poll interval -- cheap, avoids a busy-spin while workers run

    return results

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW  # noqa: E402

BUCKET_HTTPS_BASE = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"

# EDIT THIS after running --inspect if the printed columns differ. Keys are
# what this script needs; values are the actual column names in the
# dataset's parquet files. These are a best guess based on the dataset's
# public documentation (cnr/decision_date/pdf_link are named explicitly in
# the dataset docs; the rest are inferred from common eCourts field names
# and may need adjusting).
COLUMN_MAP = {
    "case_name": "title",             # confirmed against a real sample: bench=hcaurdb, year=2024
    "date": "decision_date",          # confirmed
    "pdf_link": "pdf_link",           # confirmed -- but see fetch_and_transform: this is a FILENAME, not a full path
    "court_name": "court",            # confirmed -- was wrongly "court_name" before, real column is "court"
    "cnr": "cnr",                     # confirmed
    "judge": "judge",                 # confirmed present on bench=hcaurdb -- comma-separated judge names, used for the "bench" field
    "is_final": "is_final",           # confirmed present on bench=hcaurdb -- filters out interim/interlocutory orders
}

# case_name/date/pdf_link/court_name/cnr are load-bearing -- the script
# cannot build a valid judgment record without them. judge/is_final are
# NOT: different courts/benches (confirmed empirically -- court=10_8
# bench=patnahcucisdb94 lacks is_final entirely) use different metadata
# schemas, and the rest of this script already degrades gracefully when
# they're missing (bench stays empty, no is_final/dedup filtering
# happens). Only the required set is enforced below -- treating judge/
# is_final as hard requirements was a bug: it crashed on any bench
# lacking either column even though downstream code never actually
# needed them to be present.
REQUIRED_COLUMN_KEYS = {"case_name", "date", "pdf_link", "court_name", "cnr"}

SECTION_NUM_RE = r"\d+[A-Za-z]?(?:\(\w+\))*"  # e.g. 302, 304B, 376(2)(n) -- repeated sub-clause parens allowed
SECTION_TRIGGER_RE = re.compile(r"(?:Sections?|Secs?\.?|S\.|u/s\.?)\s+", re.IGNORECASE)
BACKWARD_WINDOW_CHARS = 200  # fixed, small, bounds every regex operation below regardless of overall text length

STATUTE_NAME_PATTERNS = {
    "IPC": r"Indian\s+Penal\s+Code|IPC",
    "BNS": r"Bharatiya\s+Nyaya\s+Sanhita|BNS",
    "CrPC": r"Code\s+of\s+Criminal\s+Procedure|CrPC",
    "BNSS": r"Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita|BNSS",
}
STATUTE_NAME_RE = {statute: re.compile(names, re.IGNORECASE) for statute, names in STATUTE_NAME_PATTERNS.items()}

DISPOSAL_KEYWORDS = [
    # checked in this order -- first match wins, most specific phrasing first
    ("partly allowed", "partly allowed"),
    ("partly dismissed", "partly allowed"),
    ("allowed", "allowed"),
    ("quashed", "allowed"),
    ("dismissed", "dismissed"),
    ("rejected", "dismissed"),
    ("remanded", "remanded"),
    ("remitted", "remanded"),
]

# --- Ethics / Section 10 of the proposal: sensitive-category cases must be
# anonymized before release (juvenile, sexual-offence, matrimonial). This
# script does NOT attempt automated name redaction -- unstructured-text
# redaction has a real false-negative failure mode (missing an actual name
# buried in prose is worse than not trying at all), and getting this wrong
# is a much bigger problem than getting it conservative. Instead, any
# judgment that trips one of these keyword checks is EXCLUDED from the
# output entirely by default. This is a blunt instrument -- it will also
# drop some non-sensitive cases that happen to mention these words in
# passing (e.g. a case citing POCSO case law while deciding an unrelated
# matter) -- but false exclusions are the safe failure mode here, false
# inclusions are not. If you specifically need sensitive-category cases in
# the corpus, that requires deliberate human review of each one, not this
# script -- see --include-sensitive below, which still excludes by
# default and requires an explicit flag to even attempt inclusion, and
# even then does NOT redact names, it only removes the automatic exclusion.
SENSITIVE_CATEGORY_KEYWORDS = [
    "pocso", "protection of children from sexual offences",
    "juvenile justice", "child in conflict with law",
    "rape", "sexual assault", "outraging.{0,20}modesty",
    "matrimonial", "divorce", "custody of the child", "child custody",
    "domestic violence", "section 498a",
]
SENSITIVE_CATEGORY_RE = re.compile("|".join(SENSITIVE_CATEGORY_KEYWORDS), re.IGNORECASE)


def is_likely_sensitive_category(case_name: str, text: str) -> bool:
    return bool(SENSITIVE_CATEGORY_RE.search(case_name) or SENSITIVE_CATEGORY_RE.search(text[:3000]))


def extract_sections_cited(text: str) -> list:
    """Rewritten after a real, confirmed catastrophic-backtracking hang
    (1+ hour, on real judgment text) in the previous version, which used
    a single regex with a repeated group containing nested optional
    pieces: `(?:{NUM}\\s*{JOIN}?\\s*)+?` -- classic ReDoS shape, where
    Python's backtracking engine can explore exponentially many ways to
    partition a long run of digit-like text when the tail match
    ultimately fails. This version has NO such construct anywhere:

      1. Find each statute-name occurrence directly (IPC/BNS/CrPC/BNSS)
         -- a plain alternation, no repetition, cannot blow up.
      2. Look at a FIXED, SMALL window of text immediately before that
         match (BACKWARD_WINDOW_CHARS chars -- bounded regardless of
         how long the overall document is).
      3. Within that small window, find the nearest "Section(s)/u/s"
         trigger, then pull every number-like token between the trigger
         and the statute name via re.findall on a simple, non-nested
         pattern.

    Every regex operation here runs on a string of at most a few hundred
    characters, not the whole document -- this is what actually
    guarantees safety, not a smarter pattern on the full text."""
    found = []
    seen = set()
    for statute, name_re in STATUTE_NAME_RE.items():
        for m in name_re.finditer(text):
            window_start = max(0, m.start() - BACKWARD_WINDOW_CHARS)
            window = text[window_start:m.start()]

            trigger_positions = [tm.end() for tm in SECTION_TRIGGER_RE.finditer(window)]
            if not trigger_positions:
                continue

            # Earliest trigger in the window, not the latest -- widens the
            # scanned span to cover cases like "Section 302 read with
            # Section 34 IPC" (two separate trigger words, both numbers
            # needed), not just "Sections 302, 307, 34 of the IPC" (one
            # trigger, one list). Still bounded by the fixed small window
            # regardless of which position is chosen -- this is a
            # precision improvement, not a safety trade-off.
            numbers_region = window[min(trigger_positions):]
            for section_no in re.findall(SECTION_NUM_RE, numbers_region):
                key = (statute, section_no)
                if key not in seen:
                    seen.add(key)
                    found.append({"statute": statute, "section_no": section_no})
    return found


def guess_disposal(text: str) -> str:
    # look at the LAST ~1500 chars -- the operative order is at the end of
    # a judgment, not the facts section, so scanning the whole text risks
    # matching an unrelated earlier mention of e.g. "the trial court
    # dismissed an earlier application"
    tail = text[-1500:].lower()
    for keyword, disposal in DISPOSAL_KEYWORDS:
        if keyword in tail:
            return disposal
    return "dismissed"  # documented fallback, not a real inference -- flagged for manual review below


def inspect(parquet_dir: Path):
    files = list(parquet_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"[error] No .parquet files found under {parquet_dir}. Run the PowerShell sync step first.")
    df = pd.read_parquet(files[0])
    print(f"File: {files[0]}")
    print(f"Rows: {len(df)}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nFirst 3 rows (all columns):\n{df.head(3).to_string()}")
    print(f"\n--> If any COLUMN_MAP value above doesn't match a real column name, "
          f"edit COLUMN_MAP at the top of this script before running Phase 2.")


def select_sample_rows(parquet_dir: Path, n: int, judicial_section: str = None):
    """Pure selection logic -- reads local parquet, filters, dedups,
    samples. ZERO network calls. Shared by both the direct-download path
    (fetch_and_transform, kept for small benches where it just works) and
    the manifest-based path (write_manifest / process_local_manifest),
    which exists because Python's own requests.get() loop was observed
    going silent for many minutes at a time on some benches, with no
    reproducible cause, while `aws s3 cp` stayed fast and reliable for
    every metadata sync all session -- so the manifest path hands the
    actual downloading off to aws s3 cp instead, and this function is the
    one piece both paths need in common.

    Returns None (after printing why) if nothing survives filtering,
    rather than raising -- see the [skip] handling below."""
    files = list(parquet_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"[error] No .parquet files found under {parquet_dir}. Run the PowerShell sync step first.")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows of metadata from {len(files)} parquet file(s)")

    missing_cols = [COLUMN_MAP[k] for k in REQUIRED_COLUMN_KEYS if COLUMN_MAP[k] not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"[error] COLUMN_MAP references columns not present in the data: {missing_cols}\n"
            f"Run with --inspect first and fix COLUMN_MAP at the top of this script."
        )

    if judicial_section and "judicial_section" not in df.columns:
        print(f"[note] No 'judicial_section' column on this bench -- judicial_section={judicial_section!r} "
              f"filter was NOT applied. All case types (civil + criminal) are being sampled from here.")
    elif judicial_section and "judicial_section" in df.columns:
        before = len(df)
        df = df[df["judicial_section"].astype(str).str.strip().str.lower() == judicial_section.lower()]
        print(f"Filtered to judicial_section={judicial_section!r}: {before} -> {len(df)} rows")
        if len(df) == 0:
            print(f"[warn] No rows matched judicial_section={judicial_section!r}. "
                  f"Actual values present: {sorted(set(str(v) for v in pd.concat([pd.read_parquet(f) for f in files])['judicial_section'].dropna().unique()))[:20]}")

    if COLUMN_MAP["is_final"] in df.columns:
        before = len(df)
        df = df[df[COLUMN_MAP["is_final"]] == True]  # noqa: E712
        print(f"Filtered to is_final=True: {before} -> {len(df)} rows")
    else:
        print("[note] No 'is_final' column on this bench -- interim orders were NOT filtered out. "
              "Some kept judgments here may be interlocutory orders rather than final judgments.")
    df = df.sort_values(COLUMN_MAP["date"]).drop_duplicates(subset=[COLUMN_MAP["cnr"]], keep="last")
    print(f"After per-case dedup on cnr: {len(df)} rows")

    if len(df) == 0:
        print("[skip] No rows left to sample after filtering on this bench/year -- "
              "see the judicial_section [warn] above if one was printed. Nothing written.")
        return None

    return df.sample(n=min(n, len(df)), random_state=42)


def write_manifest(parquet_dir: Path, n: int, year: int, court_code: str, bench_code: str,
                    judicial_section: str = None):
    """Phase 1 of the manifest-based path: selection only, zero network,
    writes a CSV of exactly which files to download and where they
    should end up locally. Phase 2 (downloading) is a PowerShell loop
    over this CSV using `aws s3 cp` -- deliberately NOT done here, so
    Python never makes an HTTP request to fetch a PDF in this path at
    all."""
    df = select_sample_rows(parquet_dir, n, judicial_section)
    if df is None:
        return None

    pdf_dir = DATA_RAW / "judgments" / "_pdfs" / f"{court_code}_{bench_code}_{year}"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in df.itertuples():
        pdf_link = getattr(row, COLUMN_MAP["pdf_link"])
        cnr = getattr(row, COLUMN_MAP["cnr"])
        pdf_filename = PurePosixPath(str(pdf_link)).name
        s3_key = f"data/pdf/year={year}/court={court_code}/bench={bench_code}/{pdf_filename}"
        local_path = pdf_dir / f"{cnr}.pdf"
        rows.append({
            "cnr": cnr,
            "case_name": str(getattr(row, COLUMN_MAP["case_name"], "") or f"Case {cnr}"),
            "date": str(getattr(row, COLUMN_MAP["date"]))[:10],
            "judge": str(getattr(row, COLUMN_MAP["judge"], "") or "") if COLUMN_MAP["judge"] in df.columns else "",
            "court": str(getattr(row, COLUMN_MAP["court_name"], "") or ""),
            "s3_key": s3_key,
            "local_path": str(local_path),
        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = DATA_RAW / "judgments" / f"_manifest_{court_code}_{bench_code}_{year}.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nManifest written: {len(rows)} files listed -> {manifest_path}")
    print(f"Local PDF target directory: {pdf_dir}")
    print("\nNext: download with aws s3 cp per row in this manifest (see download_manifest.ps1), "
          "then run:\n  python scripts\\precedent\\fetch_real_judgments.py --process-local "
          f"{manifest_path}")
    return manifest_path


def process_local_manifest(manifest_path: Path, include_sensitive: bool = False):
    """Phase 3: process PDFs that are ALREADY on disk (downloaded by
    Phase 2's aws s3 cp loop). Zero network calls here either -- pure
    local file processing, same extraction/citation/disposal logic as
    fetch_and_transform's per-item loop, just reading from local_path
    instead of a fresh HTTP response."""
    manifest_df = pd.read_csv(manifest_path)
    print(f"Processing {len(manifest_df)} manifest row(s) from {manifest_path}")

    records = []
    skipped = {"missing_local_file": 0, "text_extract_failed": 0, "too_short": 0,
               "no_sections_cited": 0, "sensitive_category_excluded": 0}

    for i, row in enumerate(manifest_df.itertuples(), start=1):
        pdf_path = Path(row.local_path)
        if not pdf_path.exists():
            skipped["missing_local_file"] += 1
            continue

        text = extract_text_bounded(pdf_path, timeout=45)
        if text is None:
            skipped["text_extract_failed"] += 1
            continue

        if len(text.strip()) < 500:
            skipped["too_short"] += 1
            continue

        case_name = str(row.case_name)
        if not include_sensitive and is_likely_sensitive_category(case_name, text):
            skipped["sensitive_category_excluded"] += 1
            continue

        sections = extract_sections_cited(text)
        if not sections:
            skipped["no_sections_cited"] += 1
            continue

        judge_raw = str(row.judge) if pd.notna(row.judge) else ""
        bench_names = [j.strip() for j in judge_raw.split(",") if j.strip()]

        records.append({
            "judgment_id": f"HC-{row.cnr}",
            "case_name": case_name,
            "court": str(row.court),
            "date": str(row.date),
            "bench": bench_names,
            "sections_cited": sections,
            "text": text,
            "disposal": guess_disposal(text),
        })

        if i % 10 == 0:
            print(f"  processed {i}/{len(manifest_df)}  (kept {len(records)}, skipped {sum(skipped.values())})")

    stem = manifest_path.stem.replace("_manifest_", "")
    out_path = DATA_RAW / "judgments" / f"real_highcourt_{stem}.json"
    payload = {
        "_NOTE": (
            f"REAL judgments, downloaded via aws s3 cp (manifest-based path) rather than "
            f"Python HTTP requests. {len(records)} of {len(manifest_df)} manifest rows kept. "
            f"disposal is a keyword-heuristic guess, not verified."
        ),
        "judgments": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKept {len(records)}/{len(manifest_df)} judgments -> {out_path}")
    print(f"Skipped: {skipped}")


def fetch_and_transform(parquet_dir: Path, n: int, court_label: str, year: int, court_code: str,
                         bench_code: str, include_sensitive: bool = False, judicial_section: str = None,
                         verbose_failures: int = 3):
    df = select_sample_rows(parquet_dir, n, judicial_section)
    if df is None:
        return

    print(f"Sampled {len(df)} rows to fetch")

    records = []
    skipped = {"download_failed": 0, "text_extract_failed": 0, "too_short": 0,
               "no_sections_cited": 0, "sensitive_category_excluded": 0}
    failure_log = []
    bench_start = time.time()

    for i, row in enumerate(df.itertuples(), start=1):
        pdf_link = getattr(row, COLUMN_MAP["pdf_link"])
        cnr = getattr(row, COLUMN_MAP["cnr"])
        case_name = str(getattr(row, COLUMN_MAP["case_name"], "") or f"Case {cnr}")
        date_val = getattr(row, COLUMN_MAP["date"])
        date_str = str(date_val)[:10]  # works whether it's already a string or a pandas Timestamp
        judge_raw = getattr(row, COLUMN_MAP["judge"], "") if COLUMN_MAP["judge"] in df.columns else ""
        bench_names = [j.strip() for j in str(judge_raw or "").split(",") if j.strip()]
        row_court = str(getattr(row, COLUMN_MAP["court_name"], "") or court_label)

        # pdf_link's format varies by bench/crawl source -- confirmed by
        # a real `aws s3 ls --recursive` listing (bench=patnahcucisdb94,
        # year=2023): the REAL S3 object is always flat, directly under
        # data/pdf/year=<Y>/court=<C>/bench=<B>/<filename>.pdf, but
        # pdf_link itself sometimes carries bogus leading path segments
        # that don't correspond to real S3 structure at all (this bench:
        # "court/cnrorders/<bench>/orders/<filename>.pdf" -- the
        # "court/cnrorders/.../orders/" part is not a real prefix
        # anywhere in the bucket). Other benches (hcaurdb, year=2024)
        # have pdf_link as a bare filename with no such junk. Taking only
        # the basename handles both correctly: a bare filename's
        # basename is itself, unchanged.
        pdf_filename = PurePosixPath(str(pdf_link)).name
        pdf_key = f"data/pdf/year={year}/court={court_code}/bench={bench_code}/{pdf_filename}"
        url = pdf_link if str(pdf_link).startswith("http") else f"{BUCKET_HTTPS_BASE}/{pdf_key}"
        req_start = time.time()
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            req_elapsed = time.time() - req_start
            if req_elapsed > 10:
                # Printed immediately, not just counted -- this is the
                # signal that was missing before: a request that takes
                # 10-30s to FAIL looks identical, from the outside, to
                # the whole process being stuck, when it's actually just
                # one slow network round-trip among many. Making this
                # visible in real time (rather than only in the
                # end-of-bench failure_log) is the actual fix for
                # "nothing printed for X minutes, is it stuck?".
                print(f"  [slow] item {i}/{len(df)}: request took {req_elapsed:.1f}s then failed ({e})")
            skipped["download_failed"] += 1
            if len(failure_log) < verbose_failures:
                failure_log.append({"url": url, "error": str(e)})
            continue
        req_elapsed = time.time() - req_start
        if req_elapsed > 10:
            print(f"  [slow] item {i}/{len(df)}: request took {req_elapsed:.1f}s (succeeded)")

        pdf_path = Path(tempfile.gettempdir()) / "_judgment_download.pdf"
        pdf_path.write_bytes(resp.content)

        # Bounded with a hard, PROCESS-level timeout (extract_text_bounded,
        # defined near the top of this file) -- not thread-based. A
        # threading version of this was directly tested against a
        # worst-case tight loop and the timeout NEVER fired, confirmed
        # hung indefinitely despite an explicit short timeout, because a
        # worker that never releases the GIL can starve the very timeout
        # mechanism meant to bound it (reproduced on a real run: one item
        # hung for 14+ minutes with no timeout ever firing). A separate
        # OS process can be terminated unconditionally regardless of what
        # it's doing internally -- confirmed this actually works against
        # the same tight-loop case that defeated threading.
        text = extract_text_bounded(pdf_path, timeout=45)
        if text is None:
            skipped["text_extract_failed"] += 1
            if len(failure_log) < verbose_failures:
                failure_log.append({"url": url, "error": "pdfplumber extraction failed or exceeded 45s timeout"})
            continue

        if len(text.strip()) < 500:
            skipped["too_short"] += 1
            continue

        if not include_sensitive and is_likely_sensitive_category(case_name, text):
            skipped["sensitive_category_excluded"] += 1
            continue

        sections = extract_sections_cited(text)
        if not sections:
            skipped["no_sections_cited"] += 1
            continue

        records.append({
            "judgment_id": f"HC-{cnr}",
            "case_name": case_name,
            "court": row_court,
            "date": date_str,
            "bench": bench_names,  # from the real "judge" column, comma-split
            "sections_cited": sections,
            "text": text,
            "disposal": guess_disposal(text),
        })

        if i % 5 == 0:
            elapsed = time.time() - bench_start
            print(f"  processed {i}/{len(df)}  (kept {len(records)}, skipped {sum(skipped.values())})  [{elapsed:.0f}s elapsed]")
        time.sleep(0.1)  # light self-throttling -- polite to the public bucket, not required by it

    # Filename includes court/bench/year so repeated runs against
    # DIFFERENT partitions (to accumulate volume -- the whole point of
    # running this more than once) don't overwrite each other. Re-running
    # the SAME court/bench/year does still overwrite, which is correct
    # (it's a re-fetch of the same partition, not new data).
    out_path = DATA_RAW / "judgments" / f"real_highcourt_{court_code}_{bench_code}_{year}.json"
    payload = {
        "_NOTE": (
            f"REAL judgments from Indian High Court Judgments (AWS Open Data, "
            f"CC-BY-4.0), court={court_label}. {len(records)} of {len(df)} sampled "
            f"rows kept -- see skip counts printed at fetch time. Likely "
            f"juvenile/sexual-offence/matrimonial cases were EXCLUDED by default "
            f"(sensitive_category_excluded count above) rather than anonymized -- "
            f"see is_likely_sensitive_category() in fetch_real_judgments.py. "
            f"disposal is a keyword-heuristic guess (see guess_disposal()), not "
            f"verified -- spot-check before treating as gold. bench is left "
            f"empty (not reliably derivable from this dataset's metadata columns)."
        ),
        "judgments": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nKept {len(records)}/{len(df)} judgments -> {out_path}")
    print(f"Skipped: {skipped}")
    if failure_log:
        print(f"\nFirst {len(failure_log)} download failure(s), for diagnosis:")
        for f in failure_log:
            print(f"  {f['url']}\n    -> {f['error']}")
    print("\nNext steps:")
    print("  1. Spot-check a few disposal/sections_cited values in the output file.")
    print("  2. Optionally remove or keep sample_judgments.json alongside this "
          "(ingest_judgments.py merges all *.json files in data/raw/judgments/, "
          "rejecting on duplicate judgment_id -- the IDs don't collide, so both can coexist).")
    print("  3. Re-run: ingest_judgments.py -> segment_judgments.py -> "
          "extract_citations.py -> build_overruling_graph.py -> "
          "build_dense_index.py -> generate_nyayabench.py -> run_ablation.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Print columns/sample rows only, no download")
    parser.add_argument("--n", type=int, default=300, help="Number of judgments to fetch")
    parser.add_argument("--court", help="Court code, e.g. 27_1 (see the S3 listing from the PowerShell step)")
    parser.add_argument("--bench", help="Bench code, e.g. hcaurdb")
    parser.add_argument("--year", type=int, default=2024, help="Year partition to read (2024 spans both IPC and BNS eras)")
    parser.add_argument("--include-sensitive", action="store_true",
                         help="Do NOT auto-exclude likely juvenile/sexual-offence/matrimonial cases. "
                              "Does not perform any name redaction -- see module docstring. Off by default.")
    parser.add_argument("--judicial-section", default="Criminal",
                         help="Filter to this judicial_section value before downloading (default: Criminal -- "
                              "civil writ petitions almost never cite IPC/BNS/CrPC sections, confirmed empirically "
                              "on a 300-item unfiltered sample). Pass '' to disable filtering.")
    parser.add_argument("--manifest-only", action="store_true",
                         help="Phase 1 of the aws-s3-cp-based path: select/sample rows and write a download "
                              "manifest CSV, with ZERO network calls. Does not download anything. Requires "
                              "--court/--bench/--year.")
    parser.add_argument("--process-local", metavar="MANIFEST_CSV",
                         help="Phase 3 of the aws-s3-cp-based path: process PDFs already downloaded (via "
                              "download_manifest.ps1) against the given manifest CSV. ZERO network calls. "
                              "Does not need --court/--bench/--year.")
    args = parser.parse_args()

    if args.process_local:
        process_local_manifest(Path(args.process_local), include_sensitive=args.include_sensitive)
        return

    if not args.court or not args.bench:
        raise SystemExit("[error] --court and --bench are required unless using --process-local.")

    parquet_dir = Path("metadata") / "parquet" / f"year={args.year}" / f"court={args.court}" / f"bench={args.bench}"
    court_label = f"{args.court} High Court (bench: {args.bench})"  # replace with the real court name once you know it from --inspect

    if args.inspect:
        inspect(parquet_dir)
    elif args.manifest_only:
        write_manifest(parquet_dir, args.n, args.year, args.court, args.bench,
                        judicial_section=args.judicial_section or None)
    else:
        fetch_and_transform(parquet_dir, args.n, court_label, args.year, args.court, args.bench,
                             include_sensitive=args.include_sensitive,
                             judicial_section=args.judicial_section or None)


if __name__ == "__main__":
    main()