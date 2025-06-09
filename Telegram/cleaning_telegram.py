import pandas as pd
import re
import emoji
from deep_translator import GoogleTranslator

df = pd.read_csv('telegram_03_05.csv')

def remove_links(text: str) -> str:
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_emojis(text: str) -> str:
    return emoji.get_emoji_regexp().sub('', text)

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

text_collumn = 'Message'

translator = GoogleTranslator(source='auto', target='en')

def translate_to_en(text: str) -> str:
    if pd.isna(text) or not text.strip():
        return text
    try:
        return translator.translate(text)
    except Exception:
        return text

print('Cleaning and translating messages...')
df[text_collumn] = (
    df[text_collumn]
    .astype(str)
    .apply(clean_text)
    .apply(translate_to_en)
)

save_path = 'telegram_03_05_cleaned.csv'
df.to_csv(save_path, index=False)
print(f"Cleaned CSV saved to {save_path}.")
