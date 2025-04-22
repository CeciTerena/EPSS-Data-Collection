import json
from sentence_transformers import SentenceTransformer
import os

#THERE ARE PROBABLY WAYS TO OPTIMIZE THIS CODE!!!!!!!!!!!!
#LOOK INTO THIS LATER, INCLUDING REASEARCH ON THE TOPIC IF POSSIBLE!!!!! 
#ALSO DONT ADD THIS TO THE PIPELINE CECI YOU WILL END UP RUNNING FOR WAY TO LONG AND EXCEEDING THE MINUTES.
#RUN ONLY WHEN YOU HAVE CLOUD COMPUTING FOR THIS SHIT. 

model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

with open("cleaned_reddit_posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

#-------------------------------------------

def chunk_text_by_token(text, model):
    tokenizer = model.tokenizer
    tokens = tokenizer.encode(text)
    max_tokens = model.max_seq_length
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        chunks.append(chunk)
    return chunks

#-------------------------------------------

output = []

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
            chunk_embeddings = model.encode(chunks, device="cuda")
            final_text_embedding = np.mean(chunk_embeddings, axis=0)
        else:
            final_text_embedding = model.encode(clean_text, device= "cuda")
        entry["text_embedding"] = final_text_embedding.tolist()  
            #because the og output of encode is numpy array. maybe I can store it in another way? 
            #There's the option to store it as a tensor. Then i can already send this to building a feature matrix? 
            #But at the same time this is good cause Ill need the other information to build the feature matrix and everything will be turned into a pytorch tensor anyway.

    comments = post.get("clean_comments", [])
    
    # here each comment gets embedded separately
    # if comments:
    #     comment_embeddings = model.encode(comments, device="cuda")
    #     entry["comment_embeddings"] = [vec.tolist() for vec in comment_embeddings]
    
    #here we take the mean of all comment embeddings and store it in a signle embedding
    if comments: 
        comment_embeddings = model.encode(comments, device="cuda")
        final_comment_embedding = np.mean(comment_embeddings, axis=0)
        entry["comment_embeddings"] = final_comment_embedding.tolist()

    output.append(entry)

with open("reddit_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
