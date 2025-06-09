import pandas as pd
import re
import emoji
import html
import json
import emoji
from deep_translator import GoogleTranslator

# with open('mastodon_03_05.json', 'r', encoding='utf-8') as f:
#     data = json.load(f) 

# df = pd.DataFrame(data)

df = pd.read_csv('mastodon_22_05.csv')

md_link_ref = re.compile(r"\[([^\]]+?)\]\(((?:https?://|mailto:)[^\s)]+)\)")
url_ref = re.compile(r"(https?://\S+|www\.\S+)")
cve_re = re.compile(r"\b(cve-\d{4}-\d{4,})\b", re.IGNORECASE)
code_block_re = re.compile(r"```[\s\S]*?```")
md_punctuation_re = re.compile(r"(\*\*|\*|__|_|~~|`)(.*?)\1")
heading_hash_re = re.compile(r"^#{1,6}\s+", re.MULTILINE)
excess_punctuation_re = re.compile(r"([^\w\s])\1{2,}")
hashtag_re = re.compile(r'#\s*(?=\w)')
spaced_url_re = re.compile(r'https?://\s*\S*')
zero_width_re = re.compile(r"[\u200b\u200c\u200d\u2060]")
link_re    = re.compile(r'(?:https?://|www\.)\S+')
short_link_re = re.compile(r'https?://ift\.tt/\S+')

def remove_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace='')

def normalize_extra_tokens(text: str) -> str:
    text = cve_re.sub(lambda m: m.group(1).upper(), text)  # Normalize CVE casing
    text = html.unescape(text)
    text = code_block_re.sub(" <CODE> ", text)
    text = md_punctuation_re.sub(r"\2", text)
    text = heading_hash_re.sub("", text)
    text = excess_punctuation_re.sub(r"\1", text)
    text = remove_emoji(text)
    return text

def remove_links(text: str) -> str:
    text = spaced_url_re.sub('<URL>', text)
    text = zero_width_re.sub('<URL>', text)
    text = link_re.sub('<URL>', text)
    text = short_link_re.sub('<URL>', text)
    return text

def remove_hashtags(text: str) -> str:
    return hashtag_re.sub(',', text)

def remove_usernames_and_metadata(text: str) -> str:
    text = re.sub(r'@\w+', '', text)
    text = md_link_ref.sub('', text)  
    return text

def clean_text(text: str) -> str:
    if pd.isna(text):
        return text
    text = normalize_extra_tokens(text)
    text = remove_links(text)
    text = remove_hashtags(text)
    text = remove_usernames_and_metadata(text)
    return re.sub(r'\s+', ' ', text).strip()

translator = GoogleTranslator(source='auto', target='en')

def translate_to_en(text: str) -> str:
    if pd.isna(text) or not text.strip():
        return text
    try:
        return translator.translate(text)
    except Exception:
        return text

print('Cleaning and translating messages...')
df['body'] = (
    df['body']
    .astype(str)
    # .apply(translate_to_en)
    .apply(clean_text)
)

def normalize_and_split_cves(row, df):
    print('Normalizing and splitting rows with multiple CVEs...')
    cves = re.findall(r'CVE-\d{4}-\d{4,7}', row['cve'], re.IGNORECASE)
    cves = [cve.upper() for cve in cves]  # Normalize to uppercase
    rows = []
    for cve in cves:
        new_row = row.copy()
        new_row['cve'] = cve
        rows.append(new_row)
        
    # Remove any non-CVE characters from the CVE column
    df['cve'] = df['cve'].str.replace(r'[^CVE0-9\-]', '', regex=True)
    
    return rows

def fix_cve_formatting(df):
    expanded_rows = []
    for _, row in df.iterrows():
        expanded_rows.extend(normalize_and_split_cves(row), df)
    df = pd.DataFrame(expanded_rows) # Create a new DataFrame with the expanded rows
    df = df.drop_duplicates() # Remove duplicates
    return df

def colapse_whitespace(df):
    df['body'] = df['body'].apply(lambda txt: " ".join(txt.split()))
    return df

df = fix_cve_formatting(df)
df = colapse_whitespace(df)
output_path = 'mastodon_22_05_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned CSV saved to {output_path}.")
