import praw
import requests
import re
import json
import os
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime

reddit = praw.Reddit(
    client_id='REDDIT_CLIENT_ID',
    client_secret='REDDIT_CLIENT_SECRET',
    user_agent='REDDIT_USER_AGENT',
)

def extract_redpacketsecurity_article(url):
    try:
        response = requests.get(url, timeout=10, allow_redirects=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for h2 in soup.find_all('h2', class_='wp-block-heading'):
            a = h2.find('a')
            if a and 'CVE-' in a.get_text():
                p = h2.find_next_sibling('p')
                if p:
                    return p.get_text(strip=True)
                else:
                    return "Paragraph after CVE heading not found."
        return "No CVE heading found in <h2> tags."
    except Exception as e:
        return f"Error fetching article: {e}"

def get_top_thread(submission, depth=2):
    submission.comment_sort = 'top'
    submission.comments.replace_more(limit=0)

    top_thread = []
    if not submission.comments:
        return top_thread
    
    top_comment = submission.comments[0]

    def collect(comment, level):
        if level > depth:
            return
        top_thread.append({
            "score": comment.score,
            "author": str(comment.author),
            "text": comment.body.strip(),
            "level": level
        })
        for reply in comment.replies:
            collect(reply, level + 1)

    collect(top_comment, level=0)
    return top_thread


def scrape():
    results = []
    for submission in reddit.subreddit("all").search("CVE", sort="new", time_filter="hour", limit=1000):
        cve_pattern = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
        text = submission.title + " " + submission.selftext
        cve_matches = cve_pattern.findall(text)
        cves = list(set(match.upper() for match in cve_matches))
        cve_counts = dict(Counter(cve_matches))
        if cve_matches:
            timestamp = datetime.fromtimestamp(submission.created_utc).isoformat()
            
            urls = []
            if submission.url and not submission.is_self:
                urls.append(submission.url)
            urls += re.findall(r'(https?://\S+)', submission.selftext)
            
            article_text = None
            if "redpacketsecurity.com" in submission.url:
                article_text = extract_redpacketsecurity_article(submission.url)
            
            thread = get_top_thread(submission)
            
            print(f"Title: {submission.title}")
            print(f"CVEs: {cves}")
            print(f"CVEs Count: {cve_counts}")
            print(f"Time: {timestamp}")
            print(f"URL: https://reddit.com{submission.permalink}")
            print(f"Internal URLS: {urls}")
            print(f"Text: {submission.selftext}...")
            print(f"Article Text: {article_text}")
            print("Top Comments:")
            for c in thread:
                indent = "  " * c['level']
                print(f"{indent}↑{c['score']} by {c['author']}: {c['text']}...")
            print("-" * 40)

            post_data = {
                "cves": cves,
                "cve_counts": cve_counts,	
                "title": submission.title,
                "text": submission.selftext,
                "permalink": submission.permalink,
                "timestamp": timestamp,
                "article_text": article_text, 
                "comments": [{
                    "score": comment["score"],
                    "text": comment["text"],
                    "level": comment["level"]}
                    for comment in thread
                    if comment["text"].strip()]
            }
            results.append(post_data)
    return results


        

def save_results(new_results, filename="reddit_cve_posts.json"):
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
            
        existing_links = set(post['permalink'] for post in existing)
        combined = existing + [r for r in new_results if r['permalink'] not in existing_links]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(combined)} total posts.")
    except Exception as e:
        print(f"Error saving results: {e}")


results = scrape()
if results:
    save_results(results)


