# %%
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pickle
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

# %%
with open("match_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
print(data[0].keys())
index = faiss.read_index("faiss_index.index")
print(index.ntotal)

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
            "text": data[idx]["text"],
            "match_id": data[idx]["match_id"],
            "chunk_id": data[idx]["chunk_id"]
        })

    return results


# %%
def build_context(retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context += f"[Match {chunk['match_id']}]\n"
        context += chunk["text"] + "\n\n"
    return context


# %%
PROMPT_TEMPLATE = """You are a Cricket Analytics Assistant specialized in India Test Cricket data from 2020-2024.

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in the provided context
2. If information is NOT in the context, state: "This information is not available in the provided data"
3. Do NOT use external cricket knowledge, assumptions, or prior information
4. Do NOT infer, extrapolate, or fill in missing details

RESPONSE GUIDELINES:
- Be concise and factual
- Use clear, natural language
- Cite specific data points when available (e.g., match dates, venues, scores)
- If data is incomplete or ambiguous, acknowledge this
- Stay focused on the question asked - don't add extra analysis unless requested

Context:
{context}

Question:
{question}

Answer:"""

# %%
from dotenv import load_dotenv
import os

load_dotenv()


# %%
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# %%
def rag_answer(user_query, top_k=3):
    retrieved = retrieve_top_k_chunks(user_query, top_k)
    context = build_context(retrieved)
    
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=user_query
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# %%
print(rag_answer("Who was the highest run scorer in India vs Australia 2nd Test? in 2021 test series"))

# %%



