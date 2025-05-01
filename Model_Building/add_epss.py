import requests
import pandas as pd
from datetime import datetime 
import time

epss_cache = {}

def fetch_epss_score(cve: str, lookup_date: str) -> float | None:   #ill change this later to also do things like back off, or retry after some time due to rate limits. 
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
    processed_rows = 0

    for idx, row in df.iterrows():
        # if processed_rows >= max_rows:
        #     break

        cve = row['cve']
        date = row['timestamp']
        epss = fetch_epss_score(cve, date)
        epss_scores.append(epss)

        processed_rows += 1

        if processed_rows % 100 == 0: #or processed_rows == max_rows:  #progress save every 100 rows or at the end
            df_partial = df.iloc[:processed_rows].copy()
            df_partial['epss_score'] = epss_scores
            df_partial.to_csv(output_csv_path, index=False)
            print(f"Saved progress at row {processed_rows} to {output_csv_path}")

    if processed_rows % 100 != 0:
        df_partial = df.iloc[:processed_rows].copy()
        df_partial['epss_score'] = epss_scores
        df_partial.to_csv(output_csv_path, index=False)
        print(f"Final save at row {processed_rows} to {output_csv_path}")

    return df_partial

add_epss_scores_to_csv('merged_cve_text.csv', "merged_cve_text_with_epss.csv")