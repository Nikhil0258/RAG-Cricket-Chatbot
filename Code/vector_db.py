# %%
import pickle
import numpy as np
import faiss
import os


# %%
EMBEDDINGS_PATH = "C:/Users/HI/OneDrive/Desktop/RAG Cricket ChatBot Project/Code/Temp Files/match_embeddings.pkl"

with open(EMBEDDINGS_PATH, "rb") as f:
    data = pickle.load(f)


# %%
embeddings = []
texts = []
metadata = []

for item in data:
    embeddings.append(item["embedding"])
    texts.append(item["text"])

    # everything except embedding & text becomes metadata
    meta = {k: v for k, v in item.items() if k not in ["embedding", "text"]}
    metadata.append(meta)

embeddings = np.array(embeddings).astype("float32")



# %%
print(embeddings.shape)        # (N, 384)
print(texts[0][:150])
print(metadata[0])

# %%
print("Number of vectors:", embeddings.shape[0])
print("Embedding dimension:", embeddings.shape[1])
print("Sample metadata:", metadata[0])

# %%
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# %%
index.add(embeddings)
print("Total vectors in FAISS index:", index.ntotal)


# %%
FAISS_INDEX_PATH = "C:/Users/HI/OneDrive/Desktop/RAG Cricket ChatBot Project/Code/faiss_index.index"

faiss.write_index(index, FAISS_INDEX_PATH)


# %%
METADATA_PATH = "C:/Users/HI/OneDrive/Desktop/RAG Cricket ChatBot Project/faiss_metadata.pkl"

with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)


# %%
query_vector = embeddings[0].reshape(1, -1)

distances, indices = index.search(query_vector, k=3)

print(indices)
print([metadata[i] for i in indices[0]])



