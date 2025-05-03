#!/usr/bin/env python3
"""add_cvss.py – append CVSS base scores to a CSV that already contains EPSS
----------------------------------------------------------------------------
This version adds **robust rate‑limit handling** for the NVD 2.0 API and a
couple of tweaks that should get rid of the 403 errors you just hit.

Highlights
~~~~~~~~~~
* **Custom *User‑Agent*** header – NVD sometimes blocks requests that arrive
  without one.
* **Exponential back‑off** on **403** as well as **429** responses, with the
  server‑supplied `Retry‑After` respected when present.
* Slightly **longer default delay (7 s)** between calls when no API key is
  provided, plus a random jitter so we don’t send requests on an exact beat.

Everything else (paths, progress saves, CSV handling) is unchanged – run it
with no CLI arguments and edit the constants up top if filenames change.
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

# ───────────────────────── configuration ──────────────────────────── #
INPUT_CSV  = Path("merged_cve_text_with_epss.csv")
OUTPUT_CSV = Path("merged_cve_text_with_scores.csv")
API_KEY    = "39064de0-4531-4313-b27f-f5d93959e850" 
PROGRESS_EVERY = 100                               # rows between progress saves

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PUBLIC_API_DELAY = 7       # seconds between unauth'd requests (was 6)
MAX_BACKOFF = 300          # cap exponential back‑off at 5 minutes

BASE_HEADERS = {
    # NVD recommends sending your own UA to avoid being mistaken for a bot
    "User-Agent": "cvss-fetcher/1.1 (https://github.com/your‑org)"
}
CACHE: Dict[str, Optional[float]] = {}


# ─────────────────────── networking helper ────────────────────────── #

def fetch_cvss_score(cve: str, api_key: Optional[str] = None) -> Optional[float]:
    """Return the latest CVSS base score for *cve* (v3.1 → v3.0 → v2).

    Implements in‑process caching and exponential back‑off for 403/429.
    """
    if cve in CACHE:
        return CACHE[cve]

    attempt = 0
    delay = 0
    headers = BASE_HEADERS.copy()
    if api_key:
        headers["apiKey"] = api_key  # per NVD docs, header key *apiKey*

    while True:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(NVD_URL, params={"cveId": cve}, headers=headers, timeout=15)
            if resp.status_code in (403, 429):
                # Rate limited or forbidden – respect Retry‑After if provided
                retry_after = int(resp.headers.get("Retry-After", 0))
                if retry_after <= 0:
                    retry_after = min((2 ** attempt) * 5, MAX_BACKOFF)
                attempt += 1
                print(f"{resp.status_code} for {cve}. Backing off {retry_after}s…", file=sys.stderr)
                delay = retry_after
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching CVSS for {cve}: {exc}", file=sys.stderr)
            CACHE[cve] = None
            return None
        break

    try:
        vulns = resp.json().get("vulnerabilities", [])
        if not vulns:
            CACHE[cve] = None
            return None
        metrics = vulns[0]["cve"].get("metrics", {})
        for ns in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if ns in metrics:
                score = metrics[ns][0]["cvssData"]["baseScore"]
                CACHE[cve] = float(score)
                return CACHE[cve]
    except Exception as exc:
        print(f"Error parsing CVSS for {cve}: {exc}", file=sys.stderr)

    CACHE[cve] = None
    return None


# ─────────────────────── CSV processing ───────────────────────────── #

def add_cvss_scores_to_csv(input_csv: Path, output_csv: Path, api_key: Optional[str] = None) -> None:
    df = pd.read_csv(input_csv)
    if "cve" not in df.columns:
        raise ValueError("Input CSV must contain a 'cve' column.")

    cvss_scores = []
    last_save = 0
    for idx, cve in enumerate(df["cve"], start=1):
        score = fetch_cvss_score(str(cve), api_key)
        cvss_scores.append(score)

        # obey public‑key rate limit if unauthenticated
        if not api_key:
            # add ±0.5 s jitter so we don’t hit the limit on exact boundaries
            time.sleep(PUBLIC_API_DELAY + random.uniform(-0.5, 0.5))

        if idx % PROGRESS_EVERY == 0:
            _save_partial(df, cvss_scores, idx, output_csv)
            last_save = idx

    if last_save != len(df):
        _save_partial(df, cvss_scores, len(df), output_csv)
    print("✓ Completed all rows!")


# ───────────────────── helper util ──────────────────────────────── #

def _save_partial(df: pd.DataFrame, scores: list[Optional[float]], upto: int, out: Path) -> None:
    part = df.iloc[:upto].copy()
    part["cvss_score"] = scores
    part.to_csv(out, index=False)
    print(f"Saved {upto} rows → {out}")


# ───────────────────────── main ─────────────────────────────────── #
if __name__ == "__main__":
    try:
        add_cvss_scores_to_csv(INPUT_CSV, OUTPUT_CSV, API_KEY)
    except KeyboardInterrupt:
        print("Interrupted by user – partial results (if any) are saved.")
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
