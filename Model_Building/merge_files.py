import json
import pandas as pd
from datetime import datetime

def load_reddit(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for item in data:
        date = datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d')
        for cve in item['cves']:
            rows.append({
                'cve':            cve,
                'timestamp':      date,
                # 'cve_count':      item['cve_counts'].get(cve, 0), # I guess we can skip this because it's not in the telegram data
                'source':         'reddit',
                'text':           item['clean_text']
            })
    return pd.DataFrame(rows)

def load_telegram(path):
    df = pd.read_csv(path)
    df['cves'] = (
        df['CVE ID']
        .fillna('')
        .astype(str)
        .str.strip()
        .apply(lambda s: [s] if s else [])
    )
    rows = []
    for _, item in df.iterrows():
        entry_date = datetime.fromisoformat(item['Timestamp']).strftime('%Y-%m-%d')
        # one row per CVE code
        for cve in item['cves']:
            rows.append({
                'cve':       cve,
                'timestamp': entry_date,
                'source':    'telegram',
                'text':      item['Message']
            })

    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------

df_reddit   = load_reddit('../Source_Files/cleaned_reddit_posts.json')
df_telegram = load_telegram('../Telegram/telegram_data_cleaned.csv')
# ADD CODE FOR THE OTHER PLATFORMS TO UNIFORMIZE EVERYTHING. 

df_all = pd.concat([df_reddit, df_telegram], ignore_index=True)
df_all.to_csv('merged_cve_text.csv', index=False)
print("Merged CSV shape:", df_all.shape)
