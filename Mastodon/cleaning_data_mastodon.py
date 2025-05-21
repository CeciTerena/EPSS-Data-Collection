# Module for cleaning data from Mastodon posts

#remove emojis, links, and hashtags from the text

# return cleanes text in cvs

import json
import os
import re
import pandas as pd


base_dir = os.path.dirname(os.path.abspath(__file__))
cve_posts = []
cve_path = os.path.join(base_dir, 'cve_posts.json')
if os.path.exists(cve_path):
    with open(cve_path, "r", encoding="utf-8") as f:
        cve_posts = json.load(f)

# def remove_emojis(text):
#     print("Removing emojis...")
#     emoji_pattern = re.compile(
#         "["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', text)

# # Do not run!!!!
# # def duplicate_entry_if_many_cve(cve_list):
# #     # Check if there are multiple CVE entries in the list
# #     for cve in cve_list:
# #         if len(cve['cve']) > 1:
# #             print("Duplicating entries for multiple CVEs...")
# #             # Create a new entry for each CVE
# #             for multiple_cve in cve['cve']:
# #                 new_entry = cve.copy()
# #                 new_entry['cve'] = [multiple_cve]
# #                 cve_posts.append(new_entry)
            
# #             cve_posts.remove(cve)
                
# # for cve in cve_posts:
# #     cve['body'] = remove_emojis(cve['body'])

# # duplicate_entry_if_many_cve(cve_posts)

# def add_source(cve_list):
#     # Add the source to each CVE entry
#     print("Adding mastodon as a source to CVE entries...")
#     for cve in cve_list:
#         cve['source'] = "mastodon"


# def extract_date(cve_list):
#     # Extract the date from the created_at field
#     for cve in cve_list:
#         cve['created_at'] = cve['created_at'][:10]

# # add_source(cve_posts)
# extract_date(cve_posts)
# with open(cve_path, "w", encoding="utf-8") as f:
#     print("Populating with CVE posts...")
#     json.dump(cve_posts, f, ensure_ascii=False, indent=2)

# Read JSON file into a DataFrame
df = pd.read_json(cve_path)

# Write DataFrame to CSV

column_order = ['cve', 'created_at', 'source', 'body']
csv_path = os.path.join(base_dir, 'cve_posts_in_csv.csv')
df.to_csv(csv_path ,columns=column_order, index=False)  
        