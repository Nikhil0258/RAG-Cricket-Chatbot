# %%
import pickle
import numpy as np
import faiss
import os

# %%
index_path = 'faiss_index.index'
embeddings_path = 'match_embeddings.pkl'

# %%
with open(embeddings_path,'rb') as f:
    data = pickle.load(f)

# %%
index = faiss.read_index(index_path)

# %%
faiss_count = index.ntotal
embeddings_count = len(data)

# %%
data

# %%
print(f"FAISS Index Count: {faiss_count}")
print(f"Embeddings Count: {embeddings_count}")

# %%
texts = []
metadata = []
for item in data:
    texts.append(item['text'])
    metadata.append({
        "match_id": item["match_id"],
        "chunk_id": item["chunk_id"]
    })

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
sample_query = "Who is the highest run scorer in Australia vs India 2nd test 2021?"
embedding_vector = model.encode(
    sample_query,
    normalize_embeddings=True
)
embedding_vector = np.array([embedding_vector]).astype("float32")

# %%
k = 3
distances, indices = index.search(embedding_vector, k)
retrieved_chunks = []

for idx in indices[0]:
    retrieved_chunks.append({
        "text": texts[idx],
        "metadata": metadata[idx]
    })
for chunk in retrieved_chunks:
    print("MATCH ID:", chunk["metadata"]["match_id"])
    print(chunk["text"][:300])
    print("-" * 50)


# %%
def retrieve_top_k_chunks(query, k=3):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append({
            "text": texts[idx],
            "metadata": metadata[idx]
        })

    return results


# %%
queries = [
    "Who was the top scorer in Australia vs India 2nd Test 2021?",
    "Which match did Ravindra Jadeja take 5 wickets?",
    "India second innings collapse details"
]

for q in queries:
    print("\nQUERY:", q)
    chunks = retrieve_top_k_chunks(q)
    for c in chunks:
        print("MATCH:", c["metadata"]["match_id"])
        print(c["text"][:200])



