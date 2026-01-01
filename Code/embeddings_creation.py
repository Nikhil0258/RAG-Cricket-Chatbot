# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


# %%
file_directory = "C:\\Users\\HI\\OneDrive\\Desktop\\RAG Cricket ChatBot Project\\final_match_summaries"

# %%
import os

# %%
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


# %%
all_embeddings = []

for file in os.listdir(file_directory):
    if file.endswith(".txt"):
        match_id = file.replace("match_", "").replace("_summary.txt", "")

        with open(os.path.join(file_directory, file), "r", encoding="utf-8") as f:
            summary_text = f.read()

        chunks = chunk_text(summary_text)

        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk)

            all_embeddings.append({
                "match_id": match_id,
                "chunk_id": idx,
                "text": chunk,
                "embedding": embedding
            })


# %%
print("Total chunks embedded:", len(all_embeddings))
print("Sample embedding dimension:", len(all_embeddings[0]["embedding"]))

# %%
import pickle

with open("match_embeddings.pkl", "wb") as f:
    pickle.dump(all_embeddings, f)



