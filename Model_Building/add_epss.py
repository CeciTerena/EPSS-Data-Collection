import requests
import pandas as pd
from datetime import datetime, timedelta
import time

epss_cache = {}

def fetch_epss_fallback(cve: str, lookup_date: str, max_days: int = 30) -> float | None:
    original = fetch_epss_score(cve, lookup_date)
    if original is not None:
        return original

    base = datetime.fromisoformat(lookup_date)
    prev_score = next_score = None

    for d in range(1, max_days + 1):
        date_str = (base - timedelta(days=d)).strftime("%Y-%m-%d")
        prev_score = fetch_epss_score(cve, date_str)
        if prev_score is not None:
            break

    for d in range(1, max_days + 1):
        date_str = (base + timedelta(days=d)).strftime("%Y-%m-%d")
        next_score = fetch_epss_score(cve, date_str)
        if next_score is not None:
            break

    if prev_score is not None and next_score is not None:
        return (prev_score + next_score) / 2
    return prev_score or next_score

def fetch_epss_score(cve: str, lookup_date: str) -> float | None:  
    key = (cve, lookup_date)
    if key in epss_cache:
        return epss_cache[key]
    try:
        response = requests.get(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve, "date": lookup_date},
            timeout=10  
        )
        if response.status_code == 429:
            print("Rate limited! Sleeping for 60 seconds...")
            time.sleep(60)
            return fetch_epss_score(cve, lookup_date)
        
        elif response.status_code != 200:
            print(f"API error {response.status_code} for {cve} on {lookup_date}")
            epss_cache[key] = None
            return None

        data = response.json().get("data", [])
        if not data:
            epss_cache[key] = None
            return None

        score = float(data[0].get("epss", 0.0))
        epss_cache[key] = score
        return score

    except Exception as e:
        print(f"Exception fetching EPSS for {cve}: {e}")
        epss_cache[key] = None
        return None

def add_epss_scores_to_csv(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    
    df['timestamp'] = df['timestamp'].astype(str).apply(
        lambda ts: datetime.fromisoformat(ts).strftime('%Y-%m-%d')
    )

    epss_scores = []
    status_list = []
    processed_rows = 0

    for idx, row in df.iterrows():
        cve  = row['cve']
        date = row['timestamp']

        raw_score = fetch_epss_score(cve, date)
        if raw_score is not None:
            epss   = raw_score
            status = 'original'
        else:
            epss   = fetch_epss_fallback(cve, date)
            status = 'enriched'

        epss_scores.append(epss)
        status_list.append(status)
        processed_rows += 1

        if processed_rows % 100 == 0:
            df_partial = df.iloc[:processed_rows].copy()
            df_partial['epss_score'] = epss_scores
            df_partial['status']     = status_list
            df_partial.to_csv(output_csv_path, index=False)
            print(f"Saved progress at row {processed_rows} to {output_csv_path}")

    if processed_rows % 100 != 0:
        df_partial = df.iloc[:processed_rows].copy()
        df_partial['epss_score'] = epss_scores
        df_partial['status']     = status_list
        df_partial.to_csv(output_csv_path, index=False)
        print(f"Final save at row {processed_rows} to {output_csv_path}")

    return df_partial


def add_missing_epss_scores(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d')

    rows_to_process = df[df['epss_score'].isna()]
    print(f"Processing {len(rows_to_process)} rows with missing EPSS scores...")

    for idx, row in rows_to_process.iterrows():
        cve  = row['cve']
        date = row['timestamp']

        if pd.isna(date):
            print(f"Row {idx}: invalid date → skipped")
            continue

        try:
            epss = fetch_epss_fallback(cve, date)
        except Exception as e:
            print(f"Row {idx}: error fetching EPSS for {cve} on {date} → {e}")
            continue

        if epss is None:
            print(f"Row {idx}: no EPSS returned for {cve} on {date}")
        else:
            df.at[idx, 'epss_score'] = epss

    df.to_csv(output_csv_path, index=False)
    print("Done. CSV saved to", output_csv_path)
    return df


# =======================================
input_path = "Data_Files/merged_cve_text_with_scores.csv"
output_path = "Data_Files/merged_cve_text_with_scores_enriched.csv"
# add_missing_epss_scores(input_path, output_path)
add_epss_scores_to_csv(input_path, output_path)




