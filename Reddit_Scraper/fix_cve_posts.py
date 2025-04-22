import json
import re
from collections import Counter

cve_pattern = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

def normalize_and_count_cves(texts):
    matches = []
    for text in texts:
        if text:
            matches += cve_pattern.findall(text)
    normalized = [m.upper() for m in matches]
    return list(set(normalized)), dict(Counter(normalized))

def fix_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for post in data:
        all_texts = [post.get("title", ""), post.get("text", ""), post.get("article_text", "")]
        cves, counts = normalize_and_count_cves(all_texts)
        post["cves"] = cves
        post["cve_counts"] = counts

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Fixed CVE casing/counts for {len(data)} posts.")

fix_dataset("reddit_cve_posts.json")
