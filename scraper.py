from ast import parse
import time

from bs4 import BeautifulSoup
import re
from numpy import size
import requests
import json
import os

url_timelines = 'https://mastodon.social/api/v1/timelines/public'
url_for_hashtags = 'https://mastodon.social/api/v1/timelines/tag/cve'
rate_limit = {"limit": "40"}
cve_posts = []
post_ids = []
if os.path.exists("post_ids.json"):
    with open("post_ids.json", "r", encoding="utf-8") as f:
        post_ids = json.load(f)
else:
    post_ids = []

def contains_cve(post):
    return "CVE" in post or "Cve" in post or "cve" in post

def extract_cve(post):
    # Extract CVE IDs from the post content using regex
    cve_pattern = r'\b[Cc][Vv][Ee]-\d{4}-\d+\b'
    cve_ids = re.findall(cve_pattern, post)
    return cve_ids.upper() if cve_ids else []

def parse_html_body(content):
    #parse the body of the post
    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')
    # Get all visible text
    return soup.get_text(separator=" ")

max_id = None
for _ in range(100):  
    params = rate_limit.copy()
    if max_id:
        params["max_id"] = max_id
    r = requests.get(url_for_hashtags, params=params)
    posts = r.json()
    if not posts:
        break
    for t in posts:
        if contains_cve(t['content']) and t['id'] not in post_ids:
            post = {
                "body": parse_html_body(t['content']),
                "id": t['id'],
                "created_at": t['created_at'],
                "cve": extract_cve(t['content'])
            }
            if post["cve"]:
                cve_posts.append(post)
                post_ids.append(post["id"])
    max_id = posts[-1]['id']
    time.sleep(1)  # To avoid hitting rate limits
            

for x in range(300):
    r = requests.get(url_timelines, params=rate_limit)
    post = json.loads(r.text)
    for t in post:
        if contains_cve(t['content']) and t['id'] not in post_ids:
            post = {"body": parse_html_body(t['content']), "id": t['id'], "created_at": t['created_at'],  "cve": extract_cve(t['content'])}
            if post["cve"] != []:	
                cve_posts.append(post)
                post_ids.append(post["id"])
    time.sleep(1)  # To avoid hitting rate limits

with open("cve_posts.json", "r", encoding="utf-8") as existing_file:
    existing_data = json.load(existing_file)
    cve_posts.extend(existing_data)
with open("cve_posts.json", "w", encoding="utf-8") as f:
    json.dump(cve_posts, f, ensure_ascii=False, indent=2)


with open("post_ids.json", "w", encoding="utf-8") as f:
    json.dump(post_ids, f, ensure_ascii=False, indent=2)




