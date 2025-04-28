import json
import os
import re
import html
import emoji

md_link_ref = re.compile(r"\[([^\]]+?)\]\(((?:https?://|mailto:)[^\s)]+)\)")
url_ref = re.compile(r"(https?://\S+|www\.\S+)")

cve_re = re.compile(r"\b(cve-\d{4}-\d{4,})\b", re.IGNORECASE)

code_block_re = re.compile(r"```[\s\S]*?```")  
md_punctuation_re = re.compile(r"(\*\*|\*|__|_|~~|`)(.*?)\1")
heading_hash_re = re.compile(r"^#{1,6}\s+", re.MULTILINE) 
excess_punctuation_re = re.compile(r"([^\w\s])\1{2,}") 
# emoji_re = re.compile(
#     "["                       
#     u"\U0001F600-\U0001F64F"  
#     u"\U0001F300-\U0001F5FF"  
#     u"\U0001F680-\U0001F6FF"  
#     u"\U0001F1E6-\U0001F1FF"  
#     u"\U00002700-\U000027BF"  
#     u"\U000024C2-\U0001F251"  
#     u"\u2600-\u26FF"          
#     u"\u2700-\u27BF"          
#     "]+",
#     re.UNICODE,
# )
excess_emoji_re = re.compile("[🧵🔥🚨🕵️]")
    
def remove_emoji(text: str) -> str:
    return emoji.get_emoji_regexp().sub('', text)

def normalize_extra_tokens(text: str) -> str:
    
    text = cve_re.sub(lambda m: m.group(1).upper(), text)  # Normalize CVE casing
    
    text = html.unescape(text)
    text = code_block_re.sub(" <CODE> ", text)
    text = md_punctuation_re.sub(r"\2", text)
    text = heading_hash_re.sub("", text)
    text = excess_punctuation_re.sub(r"\1", text)
    text = remove_emoji(text)
    text = excess_emoji_re.sub("", text)
    
    return re.sub(r"\s+", " ", text).strip()

def strip_links(text: str) -> str:
    text = md_link_ref.sub(r"\1", text)
    return url_ref.sub(" <URL> ", text)

def clean_text_field(text):
    if text in ["", "...", None]:
        return None
    
    text = strip_links(text)
    text = normalize_extra_tokens(text)
    lines = text.splitlines()
    clean_lines = [line.lstrip("> ").strip() for line in lines if line.strip()]
    return " ".join(clean_lines)

def combine_fields(post):
    title = post.get("title", "").strip()
    text = clean_text_field(post.get("text"))
    article_text = clean_text_field(post.get("article_text"))

    parts = []
    if title:
        parts.append(title)
    if text:
        parts.append(text)
    if article_text:
        parts.append(article_text)
    
    return " ".join(parts).strip()

def clean_comments_field(comments):
    if not comments:
        return []
    cleaned_comments = []
    for comment in comments:
        comment_text = comment.get("text")
        level = comment.get("level")
        cleaned = clean_text_field(comment_text)
        if cleaned:
            if level is not None:
                cleaned = f"Level {level}: {cleaned}"
            cleaned_comments.append(cleaned)
    return cleaned_comments


def process_post(post):
    processed = post.copy()
    processed["clean_text"] = combine_fields(post)
    processed["clean_comments"] = clean_comments_field(post.get("comments", []))
    for key in ["text", "article_text", "comments"]:
        if key in processed:
            del processed[key]
    return processed

def load_existing_cleaned(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return {entry.get("permalink"): entry for entry in data if entry.get("permalink")}
        except json.JSONDecodeError:
            return {}

def update_cleaned_file(source_file, target_file):
    with open(source_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    cleaned_lookup = load_existing_cleaned(target_file)
    existing_count = len(cleaned_lookup)
    
    new_posts = []
    for post in raw_data:
        permalink = post.get("permalink")
        if permalink not in cleaned_lookup:
            processed = process_post(post)
            cleaned_lookup[permalink] = processed
            new_posts.append(processed)
    
    # process_posts = []   #THIS IS FOR EVERYTHING
    # for post in raw_data:
    #     processed = process_post(post)
    #     process_posts.append(processed)

    total_cleaned = list(cleaned_lookup.values())
    
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(total_cleaned, f, indent=2, ensure_ascii=False)
    
    print(f"Old count: {existing_count}. Found {len(new_posts)} new posts. Total cleaned posts: {len(total_cleaned)}")

# -----------------------------------------------------------
source_file = "../Reddit_Scraper/reddit_cve_posts.json"  

# cleaned_file1 = "cleaned_reddit_posts.json"
cleaned_file = "../Source_Files/cleaned_reddit_posts.json"

# update_cleaned_file(source_file, cleaned_file1)
update_cleaned_file(source_file, cleaned_file)
