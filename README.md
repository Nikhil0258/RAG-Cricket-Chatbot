# 🏏 RAG Cricket Chatbot (2020–2024)

A **production-grade Retrieval-Augmented Generation (RAG) chatbot** for **Indian Test Cricket (2020–2024)** that combines **deterministic cricket statistics**, **semantic search using FAISS**, and **LLM-based reasoning**, delivered through an interactive **Streamlit UI**.

This project is designed to be **accurate, explainable, and scalable**, avoiding hallucinations by strictly separating **structured stats computation** from **narrative generation**.

---

## 🚀 Key Features

- 🔢 **Deterministic Statistics Engine**
  - Accurate runs, wickets, averages, strike rates, economy
  - Computed from structured JSON scorecards (no LLM math)

- 🔍 **Semantic Search with FAISS**
  - Match summaries embedded using Sentence Transformers
  - High-precision retrieval for narrative questions

- 🧠 **Intent-Aware Query Routing**
  - Numerical → Stats Tool
  - Descriptive → RAG (FAISS + LLM)
  - Hybrid → Stats + RAG (scope-aligned)

- ⚠️ **Ambiguity Detection & Clarification Gate**
  - Prompts user when year/series/match context is missing

- 📊 **Scope Alignment**
  - Statistics and narratives are computed over the same match set
  - Explicit scope disclosure in answers

- 🖥️ **Interactive Streamlit UI**
  - Clean UI for asking cricket questions
  - Ready for cloud deployment

---

## 🧠 System Architecture (High Level)

User Query  
↓  
Query Normalization (Entities: Player, Year, Series, Match)  
↓  
Ambiguity Check  
↓  
Intent Classification  
↓  

┌──────────────────┬────────────────────┬──────────────────────┐  
│ **Numerical**    │ **Descriptive**    │ **Hybrid**           │  
│                  │                    │                      │  
│ Stats Tool       │ FAISS Retriever    │  Stats + FAISS       │  
│ (JSON Data)      │ + LLM              │  + LLM Synthesis     │  
└──────────────────┴────────────────────┴──────────────────────┘  

↓  
Final Answer with Scope Disclosure


## 📂 Project Structure

RAG-Cricket-Chatbot/
│
├── Code/ # Application source code
│ ├── app.py # Streamlit UI entry point
│ ├── final_design.py # Main chatbot orchestration
│ ├── stats_tool.py # Deterministic stats engine
│ ├── rag_chain.py # RAG + LLM chains
│ ├── retriever.py # FAISS retrieval logic
│ ├── vector_db.py # FAISS index creation
│ ├── embeddings_creation.py # Embedding pipeline
│ ├── data_ingestion_pipeline_script.py
│ └── test_suite.py # Comprehensive test suite
│
├── Dataset/ # Cleaned CSV datasets
├── final_json_scorecards/ # Final structured scorecards
├── final_match_summaries/ # Match summaries for RAG
├── Professional version Documents/# Architecture & design docs
│
├── requirements.txt # Python dependencies
├── faiss_metadata.pkl # FAISS metadata
├── .gitignore # Ignored files/folders
└── README.md # Project documentation


## 🧪 Testing Strategy

Testing is **explicit, layered, and comprehensive**.

### ✅ Test Coverage (`test_suite.py`)
- Query normalization tests
- Ambiguity detection tests
- Intent classification tests
- Numerical query validation
- Descriptive (RAG) query validation
- Hybrid query validation
- Error & edge case handling
- Performance checks

### ▶ Run Tests
```bash
python Code/test_suite.py
💡 Example Queries
Numerical
How many runs did Rishabh Pant score in 2021?

What was Bumrah’s bowling economy in 2022?

Descriptive
Describe the India vs Australia 2021 series

What happened in the first test of 2021?

Hybrid
How many runs did Pant score in 2021 and how did he play?

What was Rahane’s performance with match context?

Ambiguous (Clarification Triggered)
How many runs did Pant score?

🖥️ Running the App Locally
1️⃣ Clone Repository
bash
Copy code
git clone https://github.com/Nikhil0258/RAG-Cricket-Chatbot.git
cd RAG-Cricket-Chatbot
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Set Environment Variable
Create a .env file:

env
Copy code
OPENAI_API_KEY=your_api_key_here
5️⃣ Run Streamlit App
bash
Copy code
streamlit run Code/app.py
☁️ Deployment (Streamlit Cloud)
Push code to GitHub (✅ already done)

Go to https://share.streamlit.io

Select repository: RAG-Cricket-Chatbot

Set main file path:

bash
Copy code
Code/app.py
Add secret:

toml
Copy code
OPENAI_API_KEY = "your_api_key"
Deploy 🚀

🔒 Design Principles
❌ No LLM-based calculations

✅ Stats always computed from structured data

✅ Narratives generated only from retrieved context

✅ Explicit scope & data provenance

✅ Production-ready architecture

📈 Future Enhancements
Player comparison queries

Multi-series aggregation

Conversation memory

Advanced filtering (venue, opposition, innings)

Caching optimization for large-scale deployment

👤 Author
Nikhil Sai
Data Engineer | Python | GenAI | RAG Systems

GitHub: https://github.com/Nikhil0258

📜 License
This project is for educational and portfolio purposes.
Data sources are used for analysis and learning only.