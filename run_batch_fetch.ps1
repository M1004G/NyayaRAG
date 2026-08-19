# run_batch_fetch.ps1
#
# Loops fetch_real_judgments.py across multiple court/bench/year
# combinations, auto-discovered from the S3 listing, until a target
# TOTAL kept-judgment count is reached (or it runs out of combinations
# to try). Stops early once the target is hit -- doesn't keep going
# past it.
#
# Run from the project root (same folder you've been running the other
# commands from), with your venv already activated.
#
# Usage:
#   .\run_batch_fetch.ps1
#   .\run_batch_fetch.ps1 -TargetKept 1000 -PerBenchN 300 -Years @(2023,2024,2025)
#
# What it does NOT do: it does not parallelize downloads (stays
# single-threaded, same as running fetch_real_judgments.py by hand
# repeatedly) -- the public bucket doesn't need multiple simultaneous
# clients hammering it, and fetch_real_judgments.py already
# self-throttles between individual PDF downloads.

param(
    [int]$TargetKept = 800,
    [int]$PerBenchN = 300,
    [int[]]$Years = @(2023, 2024, 2025),
    [int]$MaxBenchesPerCourt = 3   # caps how many benches of a single (usually large) court get tried before moving to the next court -- avoids one huge court eating the whole budget
)

$ErrorActionPreference = "Stop"
$totalKept = 0
$attempted = @()

# Resume support: TargetKept is TOTAL across all runs, not "how many
# more this invocation should fetch". On startup, scan every already-
# fetched real_highcourt_*.json and count its judgments toward the
# target -- this is what makes re-running this script (with the same or
# a higher -TargetKept) additive instead of redundant. Without this, a
# second run would re-walk the same courts/benches in the same order
# (fixed random_state=42 in fetch_real_judgments.py means identical
# sampling) and just overwrite the same files with the same data.
$judgmentsDir = "data\raw\judgments"
if (Test-Path $judgmentsDir) {
    $existingFiles = Get-ChildItem "$judgmentsDir\real_highcourt_*.json" -ErrorAction SilentlyContinue
    foreach ($f in $existingFiles) {
        try {
            $parsed = Get-Content $f.FullName -Raw | ConvertFrom-Json
            $count = $parsed.judgments.Count
            $totalKept += $count
        } catch {
            Write-Host "  (couldn't parse existing file $($f.Name), not counted: $_)" -ForegroundColor DarkYellow
        }
    }
    if ($existingFiles.Count -gt 0) {
        Write-Host "Resuming: found $($existingFiles.Count) existing file(s), $totalKept judgments already kept." -ForegroundColor Cyan
    }
}

function Get-S3Subfolders($prefix) {
    # Parses `aws s3 ls` PRE-folder output into just the folder names,
    # e.g. "court=27_1/" -> "27_1"
    $lines = aws s3 ls --no-sign-request $prefix
    $names = @()
    foreach ($line in $lines) {
        if ($line -match 'PRE\s+\S+=([^/]+)/') {
            $names += $Matches[1]
        }
    }
    return $names
}

Write-Host "=== Batch fetch: target $TargetKept kept judgments across years $($Years -join ', ') ===" -ForegroundColor Cyan

foreach ($year in $Years) {
    if ($totalKept -ge $TargetKept) { break }

    Write-Host "`n--- Year ${year}: listing courts ---" -ForegroundColor Yellow
    $courts = Get-S3Subfolders "s3://indian-high-court-judgments/metadata/parquet/year=$year/"
    Write-Host "Found $($courts.Count) courts"

    foreach ($court in $courts) {
        if ($totalKept -ge $TargetKept) { break }

        $benches = Get-S3Subfolders "s3://indian-high-court-judgments/metadata/parquet/year=$year/court=$court/"
        $benches = $benches | Select-Object -First $MaxBenchesPerCourt

        foreach ($bench in $benches) {
            if ($totalKept -ge $TargetKept) { break }

            $comboLabel = "year=$year court=$court bench=$bench"
            $outFile = "$judgmentsDir\real_highcourt_${court}_${bench}_${year}.json"
            if (Test-Path $outFile) {
                Write-Host "`n>>> $comboLabel (already fetched, skipping)" -ForegroundColor DarkGray
                continue
            }

            Write-Host "`n>>> $comboLabel" -ForegroundColor Green

            $localDir = "metadata\parquet\year=$year\court=$court\bench=$bench\"
            Write-Host "  syncing metadata..." -ForegroundColor DarkGray -NoNewline
            $syncStart = Get-Date
            try {
                # No longer silenced (was Out-Null before) -- a slow sync on
                # a large bench (Patna's metadata file alone was 135,913
                # rows) looked IDENTICAL to a hang with zero output for
                # 20+ minutes. Streaming aws's own per-file progress means
                # you can now tell "still working" from "actually stuck"
                # by whether new lines keep appearing.
                aws s3 sync --no-sign-request `
                    "s3://indian-high-court-judgments/metadata/parquet/year=$year/court=$court/bench=$bench/" `
                    $localDir
            } catch {
                Write-Host "`n  sync failed, skipping: $_" -ForegroundColor Red
                continue
            }
            $syncElapsed = (Get-Date) - $syncStart
            Write-Host ("  sync done ({0:N0}s)" -f $syncElapsed.TotalSeconds) -ForegroundColor DarkGray

            # Wrapped in its own try/catch, separate from the outer
            # $ErrorActionPreference = "Stop" -- a single bench's Python
            # process failing (missing columns on that bench's schema,
            # a transient network error, anything) should skip to the
            # next bench, not abort the whole multi-hour batch run.
            # Confirmed necessary: court=10_8 bench=patnahcucisdb94 hit
            # exactly this (missing is_final column, now fixed on the
            # Python side too, but other benches may fail for other
            # reasons later, so this stays defensive either way).
            try {
                # Streams live instead of buffering silently until the
                # whole 300-download batch finishes (the old `| Out-String`
                # version showed NOTHING for the full duration of each
                # bench -- indistinguishable from a hang, confirmed
                # confusing in practice). Tee-Object captures each line
                # into $outputLines for the Kept/skip regex parsing below
                # WHILE ALSO passing it straight to the console as it
                # arrives.
                $outputLines = @()
                # -u forces Python to run fully unbuffered instead of its
                # default block-buffering when stdout isn't a real
                # terminal (true here -- it's going through a pipe to
                # Tee-Object). Without -u, print() output can sit in
                # Python's internal buffer and not reach this pipeline at
                # all until the process exits or the buffer fills --
                # which would make the Tee-Object/Write-Host streaming
                # fix from before look like it's doing nothing, even
                # though the plumbing is correct. Well-documented,
                # especially on Windows; costs nothing if it wasn't
                # actually the cause of the multi-minute silences seen.
                python -u scripts\precedent\fetch_real_judgments.py --n $PerBenchN --court $court --bench $bench --year $year 2>&1 |
                    Tee-Object -Variable outputLines | Write-Host
                $output = $outputLines -join "`n"
            } catch {
                Write-Host "  Python script failed on this bench, skipping: $_" -ForegroundColor Red
                continue
            }

            if ($output -match 'Kept (\d+)/(\d+) judgments') {
                $keptThisRun = [int]$Matches[1]
                $totalKept += $keptThisRun
                $attempted += "$comboLabel -> kept $keptThisRun"
                Write-Host "  Running total: $totalKept / $TargetKept" -ForegroundColor Cyan
            } elseif ($output -match 'No rows left to sample') {
                Write-Host "  (no criminal rows in this bench, skipped)" -ForegroundColor DarkGray
            } else {
                Write-Host "  (couldn't parse a Kept line from this run -- check output above, this bench may have errored)" -ForegroundColor Red
            }
        }
    }
}

Write-Host "`n=== Done: $totalKept total kept across $($attempted.Count) bench(es) ===" -ForegroundColor Cyan
$attempted | ForEach-Object { Write-Host "  $_" }
Write-Host "`nNext: python scripts\precedent\ingest_judgments.py  (merges every real_highcourt_*.json file automatically)"