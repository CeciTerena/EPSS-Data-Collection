import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os

model = SentenceTransformer("all-mpnet-base-v2")

with open("cleaned_reddit_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

output = []

def chunk_text_by_token(text, model):
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    max_tokens = model.max_seq_length
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        # Decode back to a string; skip any special tokens if present
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        chunks.append(chunk)
    return chunks

for post in posts:
    entry = {
        "permalink": post.get("permalink"),
        "timestamp": post.get("timestamp"),
        "cves": post.get("cves", []),
        "cve_counts": post.get("cve_counts", []),
        "title": post.get("title"),	
        "text_embedding": None,
        "comment_embeddings": []
    }

    clean_text = post.get("clean_text")
    if clean_text:
        chunks = chunk_text_by_token(clean_text, model)
        if len(chunks) > 1:
            chunk_embeddings = model.encode(chunks)
            final_text_embedding = np.mean(chunk_embeddings, axis=0)
        else:
            final_text_embedding = model.encode(clean_text)
        entry["text_embedding"] = final_text_embedding.tolist()

    comments = post.get("clean_comments", [])
    
    # here each comment gets embedded separately
    # if comments:
    #     comment_embeddings = model.encode(comments)
    #     entry["comment_embeddings"] = [vec.tolist() for vec in comment_embeddings]
    
    #here we take the mean of all comment embeddings and store it in a signle embedding
    if comments: 
        comment_embeddings = model.encode(comments)
        final_comment_embedding = np.mean(comment_embeddings, axis=0)
        entry["comment_embeddings"] = final_comment_embedding.tolist()

    output.append(entry)

with open("reddit_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
