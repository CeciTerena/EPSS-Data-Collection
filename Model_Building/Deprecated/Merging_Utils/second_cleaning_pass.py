import pandas as pd
import re
import emoji

# IN CASE ANYTHING -- MORE SPECIFICALLY THE EMOJIIIIIIIIIS --- 
# DID NOT GET REMOVED IN THE FIRST PASS... BUT WE CAN ALSO JUST DO IT HERE TOO ITS FINE... 
# MAYBE ITS WASTING COMPUTE BUT WHATEVER. 

df = pd.read_csv('Data_Files/merged_cve_text.csv')

def remove_links(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', ' <URL>', text)
    
def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace='')

def remove_usernames_and_metadata(text: str) -> str:
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  
    return text

def remove_markdown(text: str) -> str:
    return re.sub(r'[\*_]{1,3}', '', text) 

def clean_text(text: str) -> str:
    if pd.isna(text):
        return text
    text = remove_links(text)
    text = remove_emojis(text)
    text = remove_usernames_and_metadata(text)
    text = remove_markdown(text)
    return re.sub(r'\s+', ' ', text).strip() 

text_collumn = 'text'

print('Cleaning messages...')
df[text_collumn] = (
    df[text_collumn]
    .astype(str)
    .apply(clean_text)
)

# ===================================================
# Format the CVE column
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

df = fix_cve_formatting(df)

# ===================================================

df.to_csv('Data_Files/merged_cve_text.csv', index=False)
print("Cleaned CSV saved to merged_cve_text.csv.")
