# #!/usr/bin/env python3
# """add_cvss.py – append CVSS base scores to a CSV that already contains EPSS
# ----------------------------------------------------------------------------
# This version adds **robust rate‑limit handling** for the NVD 2.0 API and a
# couple of tweaks that should get rid of the 403 errors you just hit.

# Highlights
# ~~~~~~~~~~
# * **Custom *User‑Agent*** header – NVD sometimes blocks requests that arrive
#   without one.
# * **Exponential back‑off** on **403** as well as **429** responses, with the
#   server‑supplied `Retry‑After` respected when present.
# * Slightly **longer default delay (7 s)** between calls when no API key is
#   provided, plus a random jitter so we don’t send requests on an exact beat.

# Everything else (paths, progress saves, CSV handling) is unchanged – run it
# with no CLI arguments and edit the constants up top if filenames change.
# """
# from __future__ import annotations

# import os
# import random
# import sys
# import time
# from pathlib import Path
# from typing import Dict, Optional

# import pandas as pd
# import requests

# # ───────────────────────── configuration ──────────────────────────── #
# INPUT_CSV  = Path("Data_Files/merged_cve_text_with_epss.csv")
# OUTPUT_CSV = Path("Data_Files/merged_cve_text_with_scores.csv")
# API_KEY    = "39064de0-4531-4313-b27f-f5d93959e850" 
# PROGRESS_EVERY = 100                               # rows between progress saves

# NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
# PUBLIC_API_DELAY = 7       # seconds between unauth'd requests (was 6)
# MAX_BACKOFF = 300          # cap exponential back‑off at 5 minutes

# BASE_HEADERS = {
#     # NVD recommends sending your own UA to avoid being mistaken for a bot
#     "User-Agent": "cvss-fetcher/1.1 (https://github.com/your‑org)"
# }
# CACHE: Dict[str, Optional[float]] = {}

# # ─────────────────────── networking helper ────────────────────────── #

# def fetch_cvss_score(cve: str, api_key: Optional[str] = None) -> Optional[float]:
#     """Return the latest CVSS base score for *cve* (v3.1 → v3.0 → v2).

#     Implements in‑process caching and exponential back‑off for 403/429.
#     """
#     if cve in CACHE:
#         return CACHE[cve]

#     attempt = 0
#     delay = 0
#     headers = BASE_HEADERS.copy()
#     if api_key:
#         headers["apiKey"] = api_key  # per NVD docs, header key *apiKey*

#     while True:
#         if delay:
#             time.sleep(delay)
#         try:
#             resp = requests.get(NVD_URL, params={"cveId": cve}, headers=headers, timeout=15)
#             if resp.status_code in (403, 429):
#                 # Rate limited or forbidden – respect Retry‑After if provided
#                 retry_after = int(resp.headers.get("Retry-After", 0))
#                 if retry_after <= 0:
#                     retry_after = min((2 ** attempt) * 5, MAX_BACKOFF)
#                 attempt += 1
#                 print(f"{resp.status_code} for {cve}. Backing off {retry_after}s…", file=sys.stderr)
#                 delay = retry_after
#                 continue
#             resp.raise_for_status()
#         except requests.RequestException as exc:
#             print(f"Error fetching CVSS for {cve}: {exc}", file=sys.stderr)
#             CACHE[cve] = None
#             return None
#         break

#     try:
#         vulns = resp.json().get("vulnerabilities", [])
#         if not vulns:
#             CACHE[cve] = None
#             return None
#         metrics = vulns[0]["cve"].get("metrics", {})
#         for ns in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
#             if ns in metrics:
#                 score = metrics[ns][0]["cvssData"]["baseScore"]
#                 CACHE[cve] = float(score)
#                 return CACHE[cve]
#     except Exception as exc:
#         print(f"Error parsing CVSS for {cve}: {exc}", file=sys.stderr)

#     CACHE[cve] = None
#     return None


# # ─────────────────────── CSV processing ───────────────────────────── #

# def add_cvss_scores_to_csv(input_csv: Path, output_csv: Path, api_key: Optional[str] = None) -> None:
#     df = pd.read_csv(input_csv)
#     if "cve" not in df.columns:
#         raise ValueError("Input CSV must contain a 'cve' column.")

#     cvss_scores = []
#     last_save = 0
#     for idx, cve in enumerate(df["cve"], start=1):
#         score = fetch_cvss_score(str(cve), api_key)
#         cvss_scores.append(score)

#         # obey public‑key rate limit if unauthenticated
#         if not api_key:
#             # add ±0.5 s jitter so we don’t hit the limit on exact boundaries
#             time.sleep(PUBLIC_API_DELAY + random.uniform(-0.5, 0.5))

#         if idx % PROGRESS_EVERY == 0:
#             _save_partial(df, cvss_scores, idx, output_csv)
#             last_save = idx

#     if last_save != len(df):
#         _save_partial(df, cvss_scores, len(df), output_csv)
#     print("✓ Completed all rows!")


# # ───────────────────── helper util ──────────────────────────────── #

# def _save_partial(df: pd.DataFrame, scores: list[Optional[float]], upto: int, out: Path) -> None:
#     part = df.iloc[:upto].copy()
#     part["cvss_score"] = scores
#     part.to_csv(out, index=False, encoding="utf-8")
#     print(f"Saved {upto} rows → {out}")


# # ───────────────────────── main ─────────────────────────────────── #
# if __name__ == "__main__":
#     try:
#         add_cvss_scores_to_csv(INPUT_CSV, OUTPUT_CSV, API_KEY)
#     except KeyboardInterrupt:
#         print("Interrupted by user – partial results (if any) are saved.")
#     except Exception as exc:
#         print(f"Failed: {exc}", file=sys.stderr)
#         sys.exit(1)


#=================================================================================================
# #!/usr/bin/env python3
# """add_cvss.py - append CVSS base scores to a CSV that already contains EPSS
# ----------------------------------------------------------------------------
# This version adds **robust rate-limit handling** for the NVD 2.0 API and a
# couple of tweaks that should get rid of the 403 errors you just hit.

# Highlights
# ~~~~~~~~~~
# * **Custom *User-Agent*** header - NVD sometimes blocks requests that arrive
#   without one.
# * **Exponential back-off** on **403** as well as **429** responses, with the
#   server-supplied `Retry-After` respected when present.
# * Slightly **longer default delay (7 s)** between calls when no API key is
#   provided, plus a random jitter so we don’t send requests on an exact beat.

# Everything else (paths, progress saves, CSV handling) is unchanged - run it
# with no CLI arguments and edit the constants up top if filenames change.
# """
# import io
# import os
# import random
# import sys
# import time
# from pathlib import Path
# from typing import Dict, Optional

# import pandas as pd
# import requests

# # Force UTF-8 for all stdout/stderr I/O
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# # ───────────────────────── configuration ──────────────────────────── #
# INPUT_CSV  = Path("Data_Files/merged_cve_text_with_epss.csv")
# OUTPUT_CSV = Path("Data_Files/merged_cve_text_with_scores.csv")
# API_KEY    = "39064de0-4531-4313-b27f-f5d93959e850"
# PROGRESS_EVERY = 100                               # rows between progress saves

# NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
# PUBLIC_API_DELAY = 7       # seconds between unauth'd requests (was 6)
# MAX_BACKOFF = 300          # cap exponential back-off at 5 minutes

# BASE_HEADERS = {
#     # NVD recommends sending your own UA to avoid being mistaken for a bot
#     "User-Agent": "cvss-fetcher/1.1 (https://github.com/your-org)"
# }
# CACHE: Dict[str, Optional[float]] = {}


# # ─────────────────────── networking helper ────────────────────────── #

# def fetch_cvss_score(cve: str, api_key: Optional[str] = None) -> Optional[float]:
#     """Return the latest CVSS base score for *cve* (v3.1 -> v3.0 -> v2).

#     Implements in-process caching and exponential back-off for 403/429.
#     """
#     if cve in CACHE:
#         return CACHE[cve]

#     attempt = 0
#     delay = 0
#     headers = BASE_HEADERS.copy()
#     if api_key:
#         headers["apiKey"] = api_key  # per NVD docs, header key *apiKey*

#     while True:
#         if delay:
#             time.sleep(delay)
#         try:
#             resp = requests.get(NVD_URL, params={"cveId": cve}, headers=headers, timeout=15)
#             if resp.status_code in (403, 429):
#                 # Rate limited or forbidden - respect Retry-After if provided
#                 retry_after = int(resp.headers.get("Retry-After", 0))
#                 if retry_after <= 0:
#                     retry_after = min((2 ** attempt) * 5, MAX_BACKOFF)
#                 attempt += 1
#                 print(f"{resp.status_code} for {cve}. Backing off {retry_after}s...", file=sys.stderr)
#                 delay = retry_after
#                 continue
#             resp.raise_for_status()
#         except requests.RequestException as exc:
#             print(f"Error fetching CVSS for {cve}: {exc}", file=sys.stderr)
#             CACHE[cve] = None
#             return None
#         break

#     try:
#         vulns = resp.json().get("vulnerabilities", [])
#         if not vulns:
#             CACHE[cve] = None
#             return None
#         metrics = vulns[0]["cve"].get("metrics", {})
#         for ns in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
#             if ns in metrics:
#                 score = metrics[ns][0]["cvssData"]["baseScore"]
#                 CACHE[cve] = float(score)
#                 return CACHE[cve]
#     except Exception as exc:
#         print(f"Error parsing CVSS for {cve}: {exc}", file=sys.stderr)

#     CACHE[cve] = None
#     return None


# # ─────────────────────── CSV processing ───────────────────────────── #

# def add_cvss_scores_to_csv(input_csv: Path, output_csv: Path, api_key: Optional[str] = None) -> None:
#     df = pd.read_csv(input_csv)
#     if "cve" not in df.columns:
#         raise ValueError("Input CSV must contain a 'cve' column.")

#     cvss_scores = []
#     last_save = 0
#     for idx, cve in enumerate(df["cve"], start=1):
#         score = fetch_cvss_score(str(cve), api_key)
#         cvss_scores.append(score)

#         # obey public-key rate limit if unauthenticated
#         if not api_key:
#             # add ±0.5 s jitter so we don’t hit the limit on exact boundaries
#             time.sleep(PUBLIC_API_DELAY + random.uniform(-0.5, 0.5))

#         if idx % PROGRESS_EVERY == 0:
#             _save_partial(df, cvss_scores, idx, output_csv)
#             last_save = idx

#     if last_save != len(df):
#         _save_partial(df, cvss_scores, len(df), output_csv)
#     print("Completed all rows!")


# # ───────────────────── helper util ──────────────────────────────── #

# def _save_partial(df: pd.DataFrame, scores: list[Optional[float]], upto: int, out: Path) -> None:
#     part = df.iloc[:upto].copy()
#     part["cvss_score"] = scores
#     part.to_csv(out, index=False, encoding="utf-8")
#     print(f"Saved {upto} rows -> {out}")


# # ───────────────────────── main ──────────────────────────────────── #
# if __name__ == "__main__":
#     try:
#         add_cvss_scores_to_csv(INPUT_CSV, OUTPUT_CSV, API_KEY)
#     except KeyboardInterrupt:
#         print("Interrupted by user - partial results (if any) are saved.")
#     except Exception as exc:
#         print(f"Failed: {exc}", file=sys.stderr)
#         sys.exit(1)

#==============================================================================
#!/usr/bin/env python3
"""add_cvss.py - append CVSS metrics and descriptions to a CSV that already contains EPSS
----------------------------------------------------------------------------
This version appends detailed CVSS fields and the English description for each CVE.
It retains robust rate-limit handling for the NVD 2.0 API (exponential back-off on 403/429,
custom User-Agent header, jittered delays, caching) and adds resume and progress bar support.
"""
import io
import sys
import time
import random
from pathlib import Path
from typing import Dict, Optional, Any, List

import pandas as pd
import requests
from tqdm import tqdm

# Force UTF-8 for all stdout/stderr I/O
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ───────────────────────── configuration ──────────────────────────── #
INPUT_CSV  = Path("Data_Files/merged_cve_text_with_epss.csv")
OUTPUT_CSV = Path("Data_Files/merged_cve_text_with_scores.csv")
API_KEY    = "39064de0-4531-4313-b27f-f5d93959e850"
PROGRESS_EVERY = 100  # rows between progress saves

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PUBLIC_API_DELAY = 7  # seconds between unauthenticated requests
MAX_BACKOFF = 300     # cap exponential back-off at 5 minutes
BASE_HEADERS = {
    "User-Agent": "cvss-fetcher/1.1 (https://github.com/your-org)"
}
# In-memory cache for current run
di_cache: Dict[str, Optional[Dict[str, Any]]] = {}

# Fields to append per CVE
APPEND_COLUMNS: List[str] = [
    "description", "cvss_version", "base_score", "attack_vector",
    "attack_complexity", "privileges_required", "user_interaction",
    "scope", "confidentiality_impact", "integrity_impact", "availability_impact"
]

# ─────────────────────── networking helper ────────────────────────── #

def fetch_cve_details(cve_id: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Return a dict of CVE details or None if unavailable.
    Implements caching and exponential back-off for 403/429.
    """
    if cve_id in di_cache:
        return di_cache[cve_id]

    attempt = 0
    delay = 0
    headers = BASE_HEADERS.copy()
    if api_key:
        headers["apiKey"] = api_key

    while True:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(NVD_URL, params={"cveId": cve_id}, headers=headers, timeout=15)
            if resp.status_code in (403, 429):
                ra = resp.headers.get("Retry-After")
                retry_after = int(ra) if ra and ra.isdigit() else min((2 ** attempt) * 5, MAX_BACKOFF)
                attempt += 1
                print(f"{resp.status_code} for {cve_id}, backing off {retry_after}s...", file=sys.stderr)
                delay = retry_after
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching details for {cve_id}: {e}", file=sys.stderr)
            di_cache[cve_id] = None
            return None
        break

    try:
        data = resp.json().get("vulnerabilities", [])
        if not data:
            di_cache[cve_id] = None
            return None
        item = data[0]["cve"]
        # English description
        desc = next((d.get("value") for d in item.get("descriptions", []) if d.get("lang") == "en"), None)

        details: Dict[str, Any] = {col: None for col in APPEND_COLUMNS}
        details["description"] = desc
        metrics = item.get("metrics", {})
        for ns, ver in (("cvssMetricV31", "3.1"), ("cvssMetricV30", "3.0"), ("cvssMetricV2", "2.0")):
            if ns in metrics:
                cv = metrics[ns][0]["cvssData"]
                details.update({
                    "cvss_version": ver,
                    "base_score": cv.get("baseScore"),
                    "attack_vector": cv.get("attackVector") or cv.get("accessVector"),
                    "attack_complexity": cv.get("attackComplexity") or cv.get("accessComplexity"),
                    "privileges_required": cv.get("privilegesRequired") or cv.get("authentication"),
                    "user_interaction": cv.get("userInteraction"),
                    "scope": cv.get("scope"),
                    "confidentiality_impact": cv.get("confidentialityImpact"),
                    "integrity_impact": cv.get("integrityImpact"),
                    "availability_impact": cv.get("availabilityImpact"),
                })
                break
        di_cache[cve_id] = details
        return details
    except Exception as e:
        print(f"Error parsing details for {cve_id}: {e}", file=sys.stderr)
        di_cache[cve_id] = None
        return None

# ─────────────────────── CSV processing ───────────────────────────── #

def add_cve_details_to_csv(input_csv: Path, output_csv: Path, api_key: Optional[str] = None) -> None:
    df = pd.read_csv(input_csv)
    if "cve" not in df.columns:
        raise ValueError("Input CSV must contain a 'cve' column.")

    # Prepare resume
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        processed = existing.shape[0]
        # ensure input and existing align on CVE
        if not df["cve"].iloc[:processed].equals(existing["cve"]):
            raise ValueError("Existing output CSV does not match input. Cannot resume.")
        details_list = existing[APPEND_COLUMNS].to_dict(orient="records")
        print(f"Resuming from CVE #{processed+1} out of {len(df)}")
    else:
        processed = 0
        details_list = []

    total = len(df)
    # iterate with progress bar
    for idx in tqdm(range(processed, total), desc="Fetching CVE details", unit="cve", total=total):
        cve_id = str(df.at[idx, "cve"])
        det = fetch_cve_details(cve_id, api_key)
        details_list.append(det if det else {col: None for col in APPEND_COLUMNS})

        if not api_key:
            time.sleep(PUBLIC_API_DELAY + random.uniform(-0.5, 0.5))

        # save periodically
        if (idx + 1) % PROGRESS_EVERY == 0:
            _save_partial(df, details_list, idx + 1, output_csv)

    # final save
    _save_partial(df, details_list, total, output_csv)
    print("✓ Completed all rows!")

# ───────────────────── helper util ──────────────────────────────── #

def _save_partial(df: pd.DataFrame, details: List[Dict[str, Any]], upto: int, out: Path) -> None:
    part = df.iloc[:upto].reset_index(drop=True)
    det_df = pd.DataFrame(details[:upto])
    merged = pd.concat([part, det_df], axis=1)
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {upto} rows → {out}")

# ───────────────────────── main ──────────────────────────────────── #
if __name__ == "__main__":
    try:
        add_cve_details_to_csv(INPUT_CSV, OUTPUT_CSV, API_KEY)
    except KeyboardInterrupt:
        print("Interrupted by user – partial results (if any) are saved.")
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
