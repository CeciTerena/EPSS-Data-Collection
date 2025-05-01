import pandas as pd
import re
import emoji

# IN CASE ANYTHING -- MORE SPECIFICALLY THE EMOJIIIIIIIIIS --- 
# DID NOT GET REMOVED IN THE FIRST PASS... BUT WE CAN ALSO JUST DO IT HERE TOO ITS FINE... 
# MAYBE ITS WASTING COMPUTE BUT WHATEVER. 


df = pd.read_csv('merged_cve_text.csv')

def remove_links(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)

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

df.to_csv('merged_cve_text.csv', index=False)
print("Cleaned CSV saved to merged_cve_text.csv.")
