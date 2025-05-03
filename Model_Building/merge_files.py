import json
import pandas as pd
from datetime import datetime
import os

def load_mastodon(path):
    df = pd.read_csv(path)
    df['cve'] = (
        df['cve']
        .fillna('')
        .astype(str)
        .str.strip()
        .apply(lambda s: [s] if s else [])
    )
    rows = []
    for _, item in df.iterrows():
        entry_date = datetime.fromisoformat(item['created_at']).strftime('%Y-%m-%d')
        for cve in item['cve']:
            rows.append({
                'cve':       cve,
                'timestamp': entry_date,
                'source':    'mastodon',
                'text':      item['body']
            })
    return pd.DataFrame(rows)

def load_reddit(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for item in data:
        date = datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d')
        for cve in item['cves']:
            rows.append({
                'cve':       cve,
                'timestamp': date,
                'source':    'reddit',
                'text':      item['clean_text']
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
        for cve in item['cves']:
            rows.append({
                'cve':       cve,
                'timestamp': entry_date,
                'source':    'telegram',
                'text':      item['Message']
            })
    return pd.DataFrame(rows)

merged_path = '../Data/merged_cve_text.csv'

if os.path.exists(merged_path):
    df_all = pd.read_csv(merged_path)
    print(f"Loaded existing merged CSV with shape: {df_all.shape}")
else:
    df_all = pd.DataFrame(columns=['cve', 'timestamp', 'source', 'text'])
    print("No existing merged file found — starting fresh.")

df_reddit   = load_reddit('../Source_Files/cleaned_reddit_posts.json')
df_telegram = load_telegram('../Telegram/telegram_data_cleaned.csv')
df_mastodon = load_mastodon('../Mastodon/mastodon_03_05_cleaned.csv')

df_new = pd.concat([df_reddit, df_telegram, df_mastodon], ignore_index=True)

merged_cols = ['cve', 'timestamp', 'source', 'text']
df_combined = pd.concat([df_all, df_new], ignore_index=True)
df_combined.drop_duplicates(subset=merged_cols, inplace=True)

df_combined.to_csv(merged_path, index=False)
print("Updated merged CSV shape:", df_combined.shape)
