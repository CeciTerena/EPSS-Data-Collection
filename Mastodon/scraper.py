from ast import parse
import time

from bs4 import BeautifulSoup
import re
from numpy import size
import requests
import json
import os
# from cleaning_data_mastodon import remove_emojis

url_timelines = 'https://mastodon.social/api/v1/timelines/public'
url_for_hashtags = 'https://mastodon.social/api/v1/timelines/tag/cve'
rate_limit = {"limit": "40"}
cve_posts = []
post_ids = []
base_dir = os.path.dirname(os.path.abspath(__file__))
cve_path = os.path.join(base_dir, 'cve_posts.json')
ids_path = os.path.join(base_dir, 'post_ids.json')

print(cve_path)
print(ids_path)

if os.path.exists(ids_path):
    with open(ids_path, "r", encoding="utf-8") as f:
        post_ids = json.load(f)
else:
    post_ids = []

def contains_cve(post):
    return "CVE" in post or "Cve" in post or "cve" in post

def extract_cve(post):
    # Extract CVE IDs from the post content using regex
    cve_pattern = r'\b[Cc][Vv][Ee]-\d{4}-\d+\b'
    cve_ids = re.findall(cve_pattern, post)
    return cve_ids

def parse_html_body(content):
    #parse the body of the post
    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')
    # Get all visible text
    return soup.get_text(separator=" ")

max_id = None

print("Starting to scrape posts...")
print("Scraping posts by hashtags...")
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
                print("post found")

    max_id = posts[-1]['id']
    time.sleep(1)  # To avoid hitting rate limits
            

print("Scraping posts by timelines...")
max_id_timelines = None

for x in range(100):
    params = rate_limit.copy()
    if max_id_timelines:
        params["max_id_timelines"] = max_id_timelines
    r = requests.get(url_timelines, params=params)
    posts = r.json()
    if not posts:
        break
    for t in posts:
        if contains_cve(t['content']) and t['id'] not in post_ids:
            post = {"body": parse_html_body(t['content']), "id": t['id'], "created_at": t['created_at'],  "cve": extract_cve(t['content'])}
            if post["cve"] != []:	
                cve_posts.append(post)
                post_ids.append(post["id"])
                print("post found")
    max_id_timelines = posts[-1]['id']
    time.sleep(1)  # To avoid hitting rate limits

with open(cve_path, "r", encoding="utf-8") as existing_file:
    existing_data = json.load(existing_file)
    # if cve_posts: 
    #     for post in cve_posts:
    #         remove_emojis(post["body"])
    cve_posts.extend(existing_data)
with open(cve_path, "w", encoding="utf-8") as f:
    print("Saving posts to cve_posts...")
    json.dump(cve_posts, f, ensure_ascii=False, indent=2)


with open(ids_path, "w", encoding="utf-8") as f:
    print("Saving post ids to post_ids...")
    json.dump(post_ids, f, ensure_ascii=False, indent=2)




