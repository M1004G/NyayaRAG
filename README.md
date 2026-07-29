# NyayaRAG -- Phase 0 starter

Automated, computational pipeline for building the IPC<->BNS section
mapping and generating a QA benchmark from it, with zero hand-typed legal
data and zero manual legal judgment required.

## What's here

```
nyayarag/
├── data/
│   ├── raw/            # scraped, unvalidated output from each source
│   └── processed/      # cross-checked mapping + generated QA items
├── scripts/
│   ├── scrape_source_a.py     # scrapes thelawadvice.com
│   ├── scrape_source_b.py     # scrapes vakeel360.com
│   ├── merge_and_validate.py  # keeps only sections both sources agree on
│   └── generate_qa.py         # auto-generates QA items from the trusted mapping
├── requirements.txt
└── README.md
```

## Setup in VS Code

1. Open the `nyayarag` folder in VS Code (`File > Open Folder...`).
2. Open a terminal in VS Code: `` Ctrl+` `` (or `Terminal > New Terminal`).
3. Create and activate a virtual environment:

   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

   You'll know it worked because your terminal prompt gets a `(venv)` prefix.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. (Optional but recommended) In VS Code, install the Python extension
   (Microsoft) if you don't have it -- lets VS Code auto-detect the venv and
   gives you inline errors instead of only finding out when you run the file.

## Running the pipeline

Run these **in order** -- each one depends on the previous step's output:

```bash
python scripts/scrape_source_a.py
python scripts/scrape_source_b.py
python scripts/merge_and_validate.py
python scripts/generate_qa.py
```

After this, check:
- `data/processed/section_mapping.json` -- your trusted IPC<->BNS mapping table
- `data/processed/needs_review.json` -- sections the two sources disagreed on (fine to ignore for now, but worth a glance)
- `data/processed/qa_items.json` -- your auto-generated benchmark, ready to feed into retrieval

## Known limitation, on purpose

`source_b.json` will come out much smaller than `source_a.json` --
vakeel360's full table is behind a JS search widget that a plain scraper
can't execute, so `scrape_source_b.py` only pulls the handful of examples
that appear as static text on the page. That means right now only a small
number of sections have two-source agreement.

**Next step to grow this**, when you're ready: copy `scrape_source_a.py` as
a template, point it at a third site from the search results (e.g.
golegalconsultancy.com, judextutorials.com, or lawsikho.com), adjust the
regex `PATTERN` to match that site's wording, save it as
`scripts/scrape_source_c.py`, and add it to `merge_and_validate.py`'s inputs.
The merge logic already generalizes to any number of sources -- it just
keeps whatever they all agree on.

If you're using Claude to write that next scraper: paste it the relevant
raw HTML or text snippet from the target site and ask it to write the same
kind of regex-based extractor as `scrape_source_a.py`, rather than asking it
to write a "generic" scraper -- generic scrapers are exactly where an LLM
tends to hallucinate a plausible-looking pattern that silently matches zero
real rows.
