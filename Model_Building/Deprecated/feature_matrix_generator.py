import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os

"""
Alright this is what i have so far. NOTHING WAS TESTED YET.
Creates a feature matrix (DataFrame) with one row per CVE observation.

Gets files from the folder path (can be a general folder that has all of our cleaned source files) and loads them. 
It also extracts the source from the file name so that can be added to the DataFrame. 

For each JSON entry:
    - Convert the timestamp to a numeric value and to a 'YYYY-MM-DD' string.
    - Fetch the EPSS score for the CVE on the given date.
    - Flatten the text embedding list into individual features (text_emb_0, text_emb_1, …).
    - Add the CVE Count and the 'YYYY-MM-DD' string as features. 
"""

epss_cache = {}

def fetch_epss_score(cve: str, lookup_date: str) -> float | None:   #ill change this later to also do things like back off, or retry after some time due to rate limits. 
    key = (cve, lookup_date)
    if key in epss_cache:
        return epss_cache[key]
    try:
        response = requests.get(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve, "date": lookup_date}
        )
        if response.status_code != 200:
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

def parse_timestamp(ts_str: str) -> str:
    dt = datetime.fromisoformat(ts_str)
    entry_date = datetime.fromtimestamp(dt.timestamp()).strftime('%Y-%m-%d')
    return entry_date

def extract_source_from_path(filepath: str) -> str:
    filename = os.path.basename(filepath).lower()
    if "reddit" in filename:
        return "reddit"
    elif "telegram" in filename:
        return "telegram"
    elif "mastodon" in filename:
        return "mastodon"
    # EXPAND!!!!!!!
    else:
        return "unknown"3

def load_json_files(folder_path: str) -> list:
    entries = []
    for filepath in glob.glob(f"{folder_path}/*.json"):
        source = extract_source_from_path(filepath)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, dict):
                data["source"] = source
                entries.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item["source"] = source
                        entries.append(item)
            else:
                print(f"Unexpected format in {filepath}.")
    return entries

def build_cve_feature_matrix(json_data: list) -> pd.DataFrame:
    rows = []
    for entry in json_data:
        entry_date = parse_timestamp(entry["timestamp"])

        # THESE ARE THE FEATURES AT THE MOMENT. MAYBE MORE!!!!!
        text_embedding = entry.get("text_embedding", [])
        cve_list = entry.get("cves", [])
        cve_counts = entry.get("cve_counts", {})
        
        for cve in cve_list:
            count = cve_counts.get(cve, 0)
            epss_score = fetch_epss_score(cve, entry_date)
            if epss_score is None:
                continue
            
            row = {
                "cve": cve,
                "source": entry.get("source", "unknown"),
                "timestamp": entry_date,
                "cve_count": count,
                "epss_score": epss_score,
                "entry_date": entry_date
            }
            
            # The embedding gets flattened into separate columns. 
            for i, val in enumerate(text_embedding):
                row[f"text_emb_{i}"] = val
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------------

folder_path = "../NLP_Pipeline/cleaned_reddit_posts.json"   #EVENTUALLY WE SHOULD HAVE ALL OUR FILES IN THE SAME PLACE.
json_entries = load_json_files(folder_path) # MAYBE IT WONT EVEN BE JSON!!!

df_features = build_cve_feature_matrix(json_entries)

print("Final feature DataFrame shape:", df_features.shape)
print(df_features.head())
