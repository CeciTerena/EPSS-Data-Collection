import io
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ───────────────────────── Configuration ───────────────────────── #
INPUT_CSV = Path("Data_Files/May/all_merged_not_translated.csv")     # must contain 'cve' and 'date'
OUTPUT_CSV = Path("Data_Files/May/all_features_not_translated.csv")
API_KEY = "39064de0-4531-4313-b27f-f5d93959e850"      # NVD API key
PROGRESS_EVERY = 100                                 # rows between progress saves

# CVSS settings
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PUBLIC_API_DELAY = 7                                # seconds between unauthenticated requests
MAX_BACKOFF = 300                                   # cap exponential back-off at 5 minutes
BASE_HEADERS = {"User-Agent": "metrics-fetcher/1.0"}
cvss_cache: Dict[str, Optional[Dict[str, Any]]] = {}

# EPSS settings
EPSS_URL = "https://api.first.org/data/v1/epss"
epss_cache: Dict[Tuple[str,str], Optional[float]] = {}
MAX_DAYS = 30                                      # days to look back for EPSS score

APPEND_COLUMNS: List[str] = [
    # EPSS
    "epss_score", "epss_status",
    # CVSS
    "description", "cvss_version", "cvss_score", "attack_vector",
    "attack_complexity", "privileges_required", "user_interaction",
    "scope", "confidentiality_impact", "integrity_impact", "availability_impact",
    # Usability 
    "usable"
]

# ─────────────────────── EPSS Helpers ────────────────────────── #

def fetch_epss_score(cve: str, date: str) -> Optional[float]:
    key = (cve, date)
    if key in epss_cache:
        return epss_cache[key]
    try:
        resp = requests.get(EPSS_URL, params={"cve": cve, "date": date}, timeout=10)
        if resp.status_code == 429:
            time.sleep(60)
            return fetch_epss_score(cve, date)
        if resp.status_code != 200:
            epss_cache[key] = None
            return None
        data = resp.json().get("data", [])
        score = float(data[0].get("epss", 0.0)) if data else None
        epss_cache[key] = score
        return score
    except Exception:
        epss_cache[key] = None
        return None


def fetch_epss_fallback(cve: str, date: str, max_days: int = MAX_DAYS) -> Optional[float]:
    base = datetime.fromisoformat(date)
    prev = next_ = None
    for d in range(1, max_days + 1):
        ds = (base - timedelta(days=d)).strftime("%Y-%m-%d")
        prev = fetch_epss_score(cve, ds)
        if prev is not None:
            break
    for d in range(1, max_days + 1):
        ds = (base + timedelta(days=d)).strftime("%Y-%m-%d")
        next_ = fetch_epss_score(cve, ds)
        if next_ is not None:
            break
    if prev is not None and next_ is not None:
        return (prev + next_) / 2
    return prev or next_

# ─────────────────────── CVSS Helpers ────────────────────────── #

def fetch_cve_details(cve: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if cve in cvss_cache:
        return cvss_cache[cve]
    attempt, delay = 0, 0
    headers = BASE_HEADERS.copy()
    if api_key:
        headers["apiKey"] = api_key
    while True:
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(NVD_URL, params={"cveId": cve}, headers=headers, timeout=15)
            if resp.status_code in (403, 429):
                ra = resp.headers.get("Retry-After") or ""
                retry = int(ra) if ra.isdigit() else min((2 ** attempt) * 5, MAX_BACKOFF)
                attempt += 1
                delay = retry
                continue
            resp.raise_for_status()
        except requests.RequestException:
            cvss_cache[cve] = None
            return None
        break
    data = resp.json().get("vulnerabilities", [])
    if not data:
        cvss_cache[cve] = None
        return None
    item = data[0]["cve"]
    desc = next((d.get("value") for d in item.get("descriptions", []) if d.get("lang") == "en"), None)
    details = {col: None for col in APPEND_COLUMNS}
    details["description"] = desc
    metrics = item.get("metrics", {})
    for ns, ver in (("cvssMetricV31", "3.1"), ("cvssMetricV30", "3.0"), ("cvssMetricV2", "2.0")):
        if ns in metrics:
            cv = metrics[ns][0]["cvssData"]
            details.update({
                "cvss_version": ver,
                "cvss_score": cv.get("baseScore"),
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
    cvss_cache[cve] = details
    return details

# ─────────────────────── CSV Processing ────────────────────────── #

def add_metrics_to_csv(input_csv: Path, output_csv: Path, api_key: Optional[str] = None) -> None:
    df = pd.read_csv(input_csv)
    if "cve" not in df.columns or "date" not in df.columns:
        raise ValueError("CSV must contain 'cve' and 'date' columns.")

    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        proc = existing.shape[0]
        if not df["cve"].iloc[:proc].equals(existing["cve"]):
            raise ValueError("Output does not align with input; cannot resume.")
        details_list = existing[APPEND_COLUMNS].to_dict(orient="records")
    else:
        proc = 0
        details_list = []

    total = len(df)
    CVSS_FIELDS = [
        "description", "cvss_version", "cvss_score", "attack_vector",
        "attack_complexity", "privileges_required", "user_interaction",
        "scope", "confidentiality_impact", "integrity_impact", "availability_impact"
    ]
    REQUIRED_FIELDS = [
        "epss_score", "cvss_score", "attack_vector", "attack_complexity",
        "privileges_required", "user_interaction", "scope",
        "confidentiality_impact", "integrity_impact", "availability_impact"
    ]

    for idx in tqdm(range(proc, total), desc="Fetching metrics", unit="row", total=total):
        cve = str(df.at[idx, "cve"])
        date_str = pd.to_datetime(df.at[idx, "date"]).strftime("%Y-%m-%d")

        epss = fetch_epss_score(cve, date_str)
        status = "original" if epss is not None else "enriched"
        if epss is None:
            epss = fetch_epss_fallback(cve, date_str)

        cvss = fetch_cve_details(cve, api_key)

        row_det = {col: None for col in APPEND_COLUMNS}
        row_det.update({"epss_score": epss, "epss_status": status})
        if cvss:
            for key in CVSS_FIELDS:
                row_det[key] = cvss.get(key)

        row_det["usable"] = all(row_det.get(f) is not None for f in REQUIRED_FIELDS)

        details_list.append(row_det)

        if not api_key:
            time.sleep(PUBLIC_API_DELAY + random.uniform(-0.5, 0.5))

        if (idx + 1) % PROGRESS_EVERY == 0:
            save_partial(df, details_list, idx + 1, output_csv)

    save_partial(df, details_list, total, output_csv)
    print("✓ Completed all rows!")


def save_partial(df: pd.DataFrame, details: List[Dict[str, Any]], upto: int, out: Path) -> None:
    part = df.iloc[:upto].reset_index(drop=True)
    det_df = pd.DataFrame(details[:upto])
    merged = pd.concat([part, det_df], axis=1)
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved {upto} rows -> {out}")

# ───────────────────────── Main ─────────────────────────────────── #
if __name__ == "__main__":
    try:
        add_metrics_to_csv(INPUT_CSV, OUTPUT_CSV, API_KEY)
    except KeyboardInterrupt:
        print("Interrupted by user - partial results saved.")
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)
