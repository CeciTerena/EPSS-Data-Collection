import os
import pandas as pd
import re
import emoji
import html
import json
from deep_translator import GoogleTranslator

# ---- Patterns ----
md_link_ref = re.compile(r"\[([^\]]+?)\]\((?:https?://|mailto:)[^\s)]+\)")
url_pattern = re.compile(r"(?:https?://\S+|www\.\S+)")
short_link_re = re.compile(r"https?://ift\.tt/\S+")
spaced_url_re = re.compile(r"https?://\s*\S*")
zero_width_re = re.compile(r"[\u200b\u200c\u200d\u2060]")
hashtag_re = re.compile(r"#\s*(?=\w)")
cve_re = re.compile(r"\b(cve-\d{4}-\d{4,7})\b", re.IGNORECASE)
code_block_re = re.compile(r"```[\s\S]*?```")
md_punctuation_re = re.compile(r"[\*_]{1,3}")
heading_hash_re = re.compile(r"^#{1,6}\s+", re.MULTILINE)
excess_punctuation_re = re.compile(r"([^\w\s])\1{2,}")

translator = GoogleTranslator(source='auto', target='en')

# ---- Cleaning Helpers ----
def remove_emoji(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    return emoji.replace_emoji(text, replace='')

def remove_links(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = spaced_url_re.sub('<URL>', text)
    text = url_pattern.sub('<URL>', text)
    text = short_link_re.sub('<URL>', text)
    return zero_width_re.sub('', text)

def remove_hashtags(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    return hashtag_re.sub(',', text)

def remove_usernames_and_metadata(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = re.sub(r'@\w+', '', text)
    return md_link_ref.sub('', text)

def remove_markdown(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    return md_punctuation_re.sub('', text)

def remove_code_blocks(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    return code_block_re.sub(' <CODE> ', text)

def normalize_extra_tokens(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = cve_re.sub(lambda m: m.group(1).upper(), text)
    text = html.unescape(text)
    text = remove_code_blocks(text)
    text = heading_hash_re.sub('', text)
    text = excess_punctuation_re.sub(r'\1', text)
    return text

def collapse_whitespace(text: str) -> str:
    return ' '.join(text.split()) if pd.notna(text) else text

def translate_to_en(text: str) -> str:
    if pd.isna(text) or not text.strip():
        return text
    try:
        return translator.translate(text)
    except:
        return text

# ---- CVE Processing ----

def clean_split_cve(row, df):
    cves = re.findall(r'CVE-\d{4}-\d{4,7}', row['cve'], re.IGNORECASE)
    cves = [cve.upper() for cve in cves]  
    rows = []
    for cve in cves:
        new_row = row.copy()
        new_row['cve'] = cve
        rows.append(new_row)
    df['cve'] = df['cve'].str.replace(r'[^CVE0-9\-]', '', regex=True)
    
    return rows

def fix_cve_formatting(df):
    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(clean_split_cve(row, df))
    df = pd.DataFrame(expanded_rows) 
    return df

# ---- Loader & Normalizer ----
def normalize_cve_column(df: pd.DataFrame, name) -> pd.DataFrame:
    df.rename(columns=lambda col: 'cve' if 'cve' in col.lower() else col, inplace=True)
    df = fix_cve_formatting(df)
    return df

def normalize_text_column(df: pd.DataFrame, path: str) -> pd.DataFrame:
    name = os.path.splitext(os.path.basename(path))[0]
    if path.lower().endswith('.json'):
            if 'clean_text' in df.columns:
                df = df.rename(columns={'clean_text': 'text'})
            else:
                df['text'] = df.apply(
                    lambda r: ' '.join(filter(None, [
                        r.get('title',''),
                        r.get('text',''),
                        r.get('article_text','')
                    ])),
                    axis=1
                )
    else:
        if 'body' in df.columns:
            df = df.rename(columns={'body': 'text'})
        elif 'Message' in df.columns:
            df = df.rename(columns={'Message': 'text'})
        elif 'text' in df.columns:
            pass
        else:
            raise ValueError(f"No text column found in {path}")
    df['source'] = name
    return df

def normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col.lower() in ('timestamp', 'datetime', 'created_at', 'created_utc', 'posted_at'):
            df = df.rename(columns={col: 'timestamp'})
            break
    # if 'timestamp' in df.columns:
        # df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if 'timestamp' in df.columns:
        ts = df['timestamp']
        if pd.api.types.is_numeric_dtype(ts):
            df['timestamp'] = pd.to_datetime(ts, unit='s', origin='unix', errors='coerce')
        else:
            # otherwise it’s probably an ISO‐string like "2025-03-13T20:31:28"
            df['timestamp'] = pd.to_datetime(ts, infer_datetime_format=True, errors='coerce')

        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        df['time'] = df['timestamp'].dt.strftime('%H:%M')
        df = df.drop(columns=['timestamp'])

    return df


def load_and_normalize(path: str) -> pd.DataFrame:
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)

    df = normalize_text_column(df, path)
    df = normalize_timestamp_column(df)
    df = normalize_cve_column(df, path)
    df = df.drop_duplicates()
    
    cols = []
    cols.append('cve')
    cols.extend(['source', 'date', 'time', 'text'])
    return df[cols]


# ---- Unified Clean Function ----
def clean_file(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].apply(translate_to_en)
    df['text'] = df['text'].apply(normalize_extra_tokens)
    df['text'] = df['text'].apply(remove_links).apply(remove_emoji)
    df['text'] = df['text'].apply(remove_hashtags).apply(remove_usernames_and_metadata)
    df['text'] = df['text'].apply(remove_markdown).apply(collapse_whitespace)
    return df

# ---- Merging ----
def merging(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)

# ---- Count and Process ----

def count_raw_entries(path: str) -> int:
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data)
    else:
        return len(pd.read_csv(path))

from typing import Tuple

def process_file(path: str, do_clean: bool) -> Tuple[pd.DataFrame, dict]:
    stats = {}
    stats['initial'] = count_raw_entries(path)
    df_norm = load_and_normalize(path)
    stats['after_normalize'] = len(df_norm)
    if do_clean:
        df_clean = clean_file(df_norm)
        stats['after_clean'] = len(df_clean)
        return df_clean, stats
    else:
        stats['after_clean'] = None
        return df_norm, stats

def summarize_all(clean_paths: list[str], normalize_paths: list[str]) -> pd.DataFrame:
    all_stats = {}
    dfs = []
    
    for p in clean_paths:
        if 'telegram' in p.lower():
            tmp = pd.read_csv(p)
            tmp.drop(columns=['CVE ID.1'], errors='ignore', inplace=True)
            tmp.to_csv(p, index=False)
        elif 'reddit' in p.lower():
            tmp = pd.read_json(p)
            tmp.drop(columns=['cve_counts', 'clean_comments'], errors='ignore', inplace=True)
            if 'cves' in tmp.columns:
                tmp['cves'] = tmp['cves'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            tmp.to_json(p, orient='records', force_ascii=False)
            
        df, stats = process_file(p, do_clean=True)
        all_stats[p] = stats
        dfs.append(df)
        
    for p in normalize_paths:
        if 'telegram' in p.lower():
            tmp = pd.read_csv(p)
            tmp.drop(columns=['CVE ID.1'], errors='ignore', inplace=True)
            tmp.to_csv(p, index=False)
        elif 'reddit' in p.lower():
            tmp = pd.read_json(p)
            tmp.drop(columns=['cve_counts', 'clean_comments'], errors='ignore', inplace=True)
            if 'cves' in tmp.columns:
                tmp['cves'] = tmp['cves'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
            tmp.to_json(p, orient='records', force_ascii=False)
            
        df, stats = process_file(p, do_clean=False)
        all_stats[p] = stats
        dfs.append(df)

    merged = merging(*dfs)
    final_count = len(merged)

    stats_df = (
        pd.DataFrame.from_dict(all_stats, orient='index')
          .rename_axis('path')
          .reset_index()
    )
    stats_df['after_clean'] = stats_df['after_clean'].fillna('-')
    stats_df.loc['TOTAL'] = ['—', '—', '—', final_count]

    return stats_df

# ---- Main ----
def main():
    normalize_and_clean_paths = [
        '../Mastodon/mastodon_22_05.csv',
        '../Telegram/telegram_21_05_cleaned.csv',
        '../Source_Files/cleaned_reddit_posts.json'
    ]

    normalize_only_paths = [
        '../ExploitDB_Scraper/exploitdb_22_05_cve_cleaned.csv',
        '../BleepingComputer_Scraper/bleepingcomputer_22_05_cve_cleaned.csv'
    ]

    stats_df = summarize_all(normalize_and_clean_paths, normalize_only_paths)
    print(stats_df.to_string(index=False))

    merged_df = merging(*(load_and_normalize(p) if p in normalize_only_paths
                         else clean_file(load_and_normalize(p)) for p in normalize_and_clean_paths + normalize_only_paths))
    merged_df.to_csv('all_merged.csv', index=False)
    print("Merged CSV saved to all_merged.csv.")

if __name__ == '__main__':
    main()


#--- Helper to fix reddit timestamp formatting --- 