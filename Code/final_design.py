"""
============================================================================
CRICKET RAG CHATBOT - PRODUCTION SYSTEM
============================================================================

ARCHITECTURE OVERVIEW:
---------------------
This is a hybrid Retrieval-Augmented Generation (RAG) system for cricket 
statistics and narratives. It combines:

1. **Semantic Search (FAISS)**: Vector embeddings of match summaries
2. **Deterministic Statistics**: Structured data from JSON scorecards
3. **LLM Intelligence**: Query understanding and natural language generation

QUERY FLOW:
-----------
User Query → Entity Extraction → Ambiguity Check → Intent Classification
    ↓
    ├─ Numerical → Stats Tool → Formatted Answer
    ├─ Descriptive → FAISS Retrieval → LLM Generation → Answer
    └─ Hybrid → FAISS + Stats (aligned scope) → LLM Synthesis → Answer

KEY FEATURES:
-------------
- Clarification gate for ambiguous queries
- Scope alignment between statistics and narratives
- Explicit data provenance disclosure
- Match name enrichment (not just IDs)
- Performance optimization via caching
============================================================================
"""

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

import os
import json
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

# Vector embeddings for semantic search
from sentence_transformers import SentenceTransformer

# FAISS for efficient similarity search
import faiss

# LangChain components for LLM orchestration
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# External statistics module
import stats_tool as st

# Environment setup
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# SECTION 2: MODEL INITIALIZATION
# ============================================================================

class ModelManager:
    """Centralized model and configuration management."""
    
    def __init__(self):
        # Embedding model for semantic similarity (384-dimensional vectors)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # LLM for query parsing and response generation
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=0  # Deterministic outputs for consistency
        )
        
        # Project root (…/rag-cricket-chatbot)
        project_root = Path(__file__).resolve().parents[1]

        # Match summaries directory (relative, cloud-safe)
        self.files_dir = project_root / "final_match_summaries"
        
        # FAISS index (populated during initialization)
        self.index = None
        self.texts = []
        self.metadata = []
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_llm(self):
        return self.llm


# Global model manager instance
models = ModelManager()


# ============================================================================
# SECTION 3: TEXT PROCESSING UTILITIES
# ============================================================================

class TextProcessor:
    """Handles text chunking and preprocessing."""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for better semantic retrieval.
        
        Why chunking?
        - Smaller chunks improve FAISS search precision
        - Overlap preserves context at chunk boundaries
        - 400 chars balances specificity vs. context
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks


# ============================================================================
# SECTION 4: VECTOR STORE (FAISS INDEX)
# ============================================================================

class VectorStore:
    """Manages FAISS index creation and semantic retrieval."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.embedding_model = model_manager.get_embedding_model()
    
    def build_index(self) -> Dict[str, Any]:
        """
        Build FAISS index from match summary files.
        
        Process:
        1. Read all .txt files from summaries directory
        2. Chunk each file into 400-char segments
        3. Generate embeddings for each chunk
        4. Create FAISS L2 distance index
        
        Returns:
            Dictionary with index, texts, and metadata
        """
        all_embeddings = []
        
        # Process each match summary file
        for file in os.listdir(self.model_manager.files_dir):
            if file.endswith(".txt"):
                match_id = file.replace("match_", "").replace("_summary.txt", "")
                
                with open(os.path.join(self.model_manager.files_dir, file), "r", encoding="utf-8") as f:
                    summary_text = f.read()
                
                # Split into chunks
                chunks = TextProcessor.chunk_text(summary_text)
                
                # Generate embeddings for each chunk
                for idx, chunk in enumerate(chunks):
                    embedding = self.embedding_model.encode(chunk)
                    all_embeddings.append({
                        "match_id": match_id,
                        "chunk_id": idx,
                        "text": chunk,
                        "embedding": embedding
                    })
        
        # Separate components for FAISS
        embeddings = []
        texts = []
        metadata = []
        
        for item in all_embeddings:
            embeddings.append(item["embedding"])
            texts.append(item["text"])
            meta = {k: v for k, v in item.items() if k not in ["embedding", "text"]}
            metadata.append(meta)
        
        embeddings = np.array(embeddings).astype("float32")
        
        # Create FAISS index (L2 distance)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        print(f"✓ FAISS index built: {index.ntotal} vectors, {embeddings.shape[1]} dimensions")
        
        return {
            "index": index,
            "texts": texts,
            "metadata": metadata
        }
    
    def retrieve_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve most semantically similar chunks using FAISS.
        
        Args:
            query: User's natural language query
            k: Number of chunks to retrieve
        
        Returns:
            List of dictionaries with text and metadata
        """
        query_embedding = self.embedding_model.encode(query).astype("float32").reshape(1, -1)
        distances, indices = self.model_manager.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            results.append({
                "text": self.model_manager.texts[idx],
                "metadata": self.model_manager.metadata[idx]
            })
        
        return results


# ============================================================================
# SECTION 5: QUERY PROCESSING PIPELINE
# ============================================================================

class QueryProcessor:
    """Handles query normalization and entity extraction."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def normalize_query(self, query: str) -> Dict:
        """
        Extract structured entities from natural language query.
        
        Uses LLM to identify:
        - Player names (normalized to full names)
        - Years (2020-2024 range)
        - Team/opponent names
        - Match numbers (1st, 2nd, etc.)
        - Series information
        
        Args:
            query: User's natural language question
        
        Returns:
            Dictionary with extracted entities
        """
        template = """
You are a cricket query parser. Extract structured information from the user's query.

Task:
1. Identify player names (use full names)
2. Identify years (2020-2024 range)
3. Identify team/opponent names
4. Identify match numbers (1st, 2nd, 3rd, 4th, etc.)
5. Identify series information

Return ONLY a valid JSON object with these fields:
{{
    "raw_query": "original query",
    "year": year as integer or null,
    "series": "Team1 vs Team2" or null,
    "match_number": integer or null,
    "player": "Full Player Name" or null,
    "team": "team name" or null
}}

Guidelines:
- Set null for missing information
- Normalize names (e.g., "Pant" becomes "Rishabh Pant")
- Convert ordinals to integers (e.g., "fourth" becomes 4)

Query: {query}

JSON:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"query": query})
            
            # Clean markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            normalized = json.loads(response)
            return normalized
            
        except Exception as e:
            print(f"Normalization error: {e}")
            return {
                "raw_query": query,
                "year": None,
                "series": None,
                "match_number": None,
                "player": None,
                "team": None
            }
    
    def check_ambiguity(self, normalized: Dict) -> Dict:
        """
        Determine if query requires clarification.
        
        Ambiguity rules:
        1. Match number without year/series
        2. Player statistics without timeframe
        3. Generic references without context
        
        Args:
            normalized: Extracted entities from query
        
        Returns:
            Dictionary with clarification status and message
        """
        issues = []
        
        # Check for match number without context
        if normalized.get("match_number") and not normalized.get("year") and not normalized.get("series"):
            issues.append(f"You mentioned the {normalized['match_number']}th test, but which year or series?")
        
        # Check for player stats without timeframe
        if normalized.get("player") and not normalized.get("year") and not normalized.get("series"):
            query_lower = normalized.get("raw_query", "").lower()
            stat_keywords = ["how many", "total", "runs", "wickets", "performance", "statistics", "stats"]
            
            if any(keyword in query_lower for keyword in stat_keywords):
                issues.append(f"You're asking about {normalized['player']}, but for which year or series?")
        
        # Check for vague references
        query_lower = normalized.get("raw_query", "").lower()
        vague_patterns = ["the match", "the test", "the series", "that match"]
        
        if any(pattern in query_lower for pattern in vague_patterns):
            if not normalized.get("year") and not normalized.get("series"):
                issues.append("Which specific match, series, or year are you referring to?")
        
        if issues:
            return {
                "needed": True,
                "message": " ".join(issues) + "\n\nPlease specify: year (2020-2024) or series (e.g., 'India vs Australia 2021')"
            }
        
        return {"needed": False}
    
    def classify_intent(self, query: str) -> str:
        """
        Classify query intent using LLM reasoning.
        
        Categories:
        - numerical: Requests specific statistics
        - descriptive: Requests narrative/summary
        - hybrid: Needs both statistics and narrative
        
        Args:
            query: User's natural language question
        
        Returns:
            Intent classification string
        """
        template = """
You are a query intent classifier for a cricket chatbot.

Classify the user's query into ONE of these categories:

1. NUMERICAL: Query asks for specific statistics, numbers, or metrics
   Examples: "How many runs?", "What was the average?", "Total wickets?"

2. DESCRIPTIVE: Query asks for summaries, narratives, or explanations
   Examples: "Describe the match", "What happened?", "Tell me about the series"

3. HYBRID: Query needs BOTH statistics AND narrative/context
   Examples: "How many runs did Pant score and how did he play?"

Respond with ONLY ONE WORD: numerical, descriptive, or hybrid

Query: {query}

Classification:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"query": query})
            response = response.strip().lower()
            
            valid_intents = ["numerical", "descriptive", "hybrid"]
            
            if response in valid_intents:
                return response
            
            # Extract intent keyword from response
            for intent in valid_intents:
                if intent in response:
                    return intent
            
            return "descriptive"  # Default fallback
            
        except Exception as e:
            print(f"Intent classification error: {e}")
            return "descriptive"


# ============================================================================
# SECTION 6: STATISTICS INTEGRATION
# ============================================================================

class StatsHandler:
    """Handles interaction with stats_tool and formatting."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def parse_numerical_query(self, query: str, normalized: Dict) -> Dict:
        """
        Determine which statistical metric is requested.
        
        Supported metrics:
        - total_runs: Aggregate runs in series
        - individual_runs: Per-match breakdown
        - total_wickets: Aggregate wickets
        - wickets_per_match: Per-match breakdown
        - batting_avg_sr: Average and strike rate
        - bowling_economy_sr: Economy and strike rate
        - boundaries: Fours and sixes
        
        Args:
            query: User's natural language question
            normalized: Pre-extracted entities
        
        Returns:
            Dictionary with player, year, opponent, metric
        """
        template = """
You are a cricket statistics query parser.

Available statistical metrics:
1. total_runs - Total runs scored by player in series/matches
2. individual_runs - Runs scored per individual match (breakdown)
3. total_wickets - Total wickets taken by bowler
4. wickets_per_match - Wickets taken per individual match
5. batting_avg_sr - Batting average and strike rate
6. bowling_economy_sr - Bowling economy rate and strike rate
7. boundaries - Number of fours and sixes hit

Query: {query}

Already extracted information:
{normalized}

Return ONLY a valid JSON object:
{{
    "player": "player name",
    "year": year as integer or null,
    "opponent": "team name" or null,
    "metric": "one of the metrics above"
}}

Guidelines:
- Choose the most appropriate metric
- Use extracted data when available
- For batting queries: total_runs or batting_avg_sr
- For bowling queries: total_wickets or bowling_economy_sr

JSON:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "query": query,
                "normalized": json.dumps(normalized, indent=2)
            })
            
            # Clean markdown formatting
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            parsed = json.loads(response)
            return parsed
            
        except Exception as e:
            print(f"Numerical parsing error: {e}")
            return {
                "player": normalized.get("player"),
                "year": normalized.get("year"),
                "opponent": None,
                "metric": "total_runs"
            }
    
    @lru_cache(maxsize=128)
    def get_match_details(self, match_id: str) -> Dict:
        """
        Retrieve match metadata from scorecards (cached for performance).
        
        Args:
            match_id: Unique match identifier
        
        Returns:
            Dictionary with match name, year, venue, teams
        """
        if not hasattr(st, 'scorecards') or not st.scorecards:
            st.scorecards = st.load_scorecards()
        
        for sc in st.scorecards:
            if str(sc["match_info"]["match_id"]) == str(match_id):
                match_info = sc["match_info"]
                
                team1 = match_info["teams"]["team1"]["name"]
                team2 = match_info["teams"]["team2"]["name"]
                venue = match_info.get("venue", "Unknown Venue")
                start_date = match_info["dates"]["start"]
                
                year = start_date.split("-")[0] if start_date else "Unknown"
                match_name = f"{team1} vs {team2}"
                
                return {
                    "match_name": match_name,
                    "year": year,
                    "venue": venue,
                    "date": start_date,
                    "team1": team1,
                    "team2": team2
                }
        
        return {
            "match_name": f"Match {match_id}",
            "year": "Unknown",
            "venue": "Unknown",
            "date": "Unknown",
            "team1": "Unknown",
            "team2": "Unknown"
        }
    
    def format_stats(self, metric: str, raw_result: Any, player: str, match_ids: Optional[List] = None) -> str:
        """
        Format raw statistics into human-readable text.
        
        Features:
        - Replaces match IDs with match names/venues
        - Adds explicit scope disclosure
        - Calculates totals for list results
        - Formats different metric types appropriately
        
        Args:
            metric: Type of statistic
            raw_result: Raw output from stats tool
            player: Player name
            match_ids: Optional filter for scope alignment
        
        Returns:
            Formatted string with scope disclosure
        """
        if isinstance(raw_result, str):
            return raw_result
        
        # Scope disclosure
        scope_note = ""
        if match_ids:
            scope_note = f"\n**Scope:** Limited to {len(match_ids)} match(es) from retrieved context\n"
        else:
            scope_note = f"\n**Scope:** All available matches in database\n"
        
        # Format based on metric type
        if metric == "individual_runs":
            if isinstance(raw_result, list) and len(raw_result) > 0:
                total_runs = sum(match["runs"] for match in raw_result)
                
                output = f"**{player}'s Run Breakdown:**\n"
                output += scope_note
                output += f"\n**Total Runs: {total_runs}**\n\n"
                output += "**Per Match Details:**\n"
                
                for match in raw_result:
                    match_id = match["match_id"]
                    runs = match["runs"]
                    details = self.get_match_details(match_id)
                    
                    output += f"\n**{details['year']} - {details['match_name']}**\n"
                    output += f"  Venue: {details['venue']}\n"
                    output += f"  Runs Scored: **{runs}**\n"
                
                return output
            else:
                return f"No run data found for {player}"
        
        elif metric == "total_runs":
            output = f"**Total Runs:** {raw_result}\n"
            output += scope_note
            return output
        
        elif metric == "batting_avg_sr":
            if isinstance(raw_result, dict):
                output = f"**{player}'s Batting Statistics:**\n"
                output += scope_note
                output += f"\n- Total Runs: {raw_result.get('runs', 'N/A')}\n"
                avg = raw_result.get('average')
                output += f"- Average: {avg:.2f}\n" if avg else "- Average: N/A\n"
                sr = raw_result.get('strike_rate')
                output += f"- Strike Rate: {sr:.2f}\n" if sr else "- Strike Rate: N/A\n"
                return output
        
        # Add other metric formatting as needed...
        return str(raw_result)
    
    def execute_stats_query(self, parsed_query: Dict, match_ids: Optional[List] = None) -> str:
        """
        Execute statistical computation based on parsed query.
        
        Critical feature: When match_ids are provided, filters results
        to only those matches for scope alignment in hybrid queries.
        
        Args:
            parsed_query: Parsed query with player, year, metric
            match_ids: Optional list to filter results
        
        Returns:
            Formatted statistics string
        """
        metric = parsed_query.get("metric")
        player = parsed_query.get("player")
        year = parsed_query.get("year")
        opponent = parsed_query.get("opponent")
        
        try:
            raw_result = None
            
            # --- UPDATED MAPPING ---
            if metric == "total_runs":
                raw_result = st.total_runs_in_series(player, year, opponent, match_ids)
            elif metric == "individual_runs":
                raw_result = st.individual_runs_per_match(player, year, opponent, match_ids)
            elif metric == "total_wickets":
                raw_result = st.total_wickets(player, year, opponent, match_ids)
            elif metric == "wickets_per_match":
                raw_result = st.wickets_per_match(player, year, opponent, match_ids)
            elif metric == "batting_avg_sr":
                raw_result = st.batting_avg_sr(player, year, opponent, match_ids)
            elif metric == "bowling_economy_sr":
                raw_result = st.bowling_economy_sr(player, year, opponent, match_ids)
            elif metric == "boundaries":
                raw_result = st.boundaries(player, year, opponent, match_ids)
            
            # Filter optimization for list results is now handled inside stats_tool 
            # via match_ids param, so we don't need extra filtering logic here.
            
            return self.format_stats(metric, raw_result, player, match_ids)
            
        except Exception as e:
            return f"Stats tool error: {e}"


# ============================================================================
# SECTION 7: RESPONSE GENERATORS
# ============================================================================

class ResponseGenerator:
    """Handles LLM-based response generation."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def build_context(self, chunks: List[Dict], stats_handler: StatsHandler) -> str:
        """Format retrieved chunks into readable context string."""
        if not chunks:
            return "No relevant context found."
        
        context = ""
        for c in chunks:
            match_id = c['metadata']['match_id']
            details = stats_handler.get_match_details(match_id)
            
            context += f"**{details['year']} - {details['match_name']}** (at {details['venue']}):\n"
            context += c["text"] + "\n\n"
            context += "-" * 70 + "\n\n"
        
        return context
    
    def generate_descriptive(self, query: str, context: str) -> str:
        """Generate narrative response using RAG context."""
        template = """
You are answering a descriptive cricket query about India Test matches.

Context from match summaries:
{context}

Question: {query}

Instructions:
- Provide clear, concise narrative based ONLY on the context
- Do not make up information
- If answer is not in context, state clearly
- Use natural language

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({"context": context, "query": query})
    
    def generate_hybrid(self, query: str, stats: str, context: str) -> str:
        """Generate response combining statistics and narrative."""
        template = """
You are answering a cricket query requiring both statistics and narrative.

Statistical Information (USE EXACT NUMBERS):
{stats}

Match Context (for narrative details):
{context}

Question: {query}

Instructions:
1. Use EXACT statistics provided - do not recalculate
2. Combine verified statistics with narrative context
3. Present information in clear, structured format
4. Start with statistics, then add narrative context

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({"stats": stats, "context": context, "query": query})


# ============================================================================
# SECTION 8: MAIN ORCHESTRATOR
# ============================================================================

class CricketChatbot:
    """Main chatbot orchestrator."""
    
    def __init__(self):
        print("Initializing Cricket RAG Chatbot...")
        
        # Initialize components
        self.model_manager = models
        self.vector_store = VectorStore(self.model_manager)
        self.query_processor = QueryProcessor(self.model_manager.get_llm())
        self.stats_handler = StatsHandler(self.model_manager.get_llm())
        self.response_generator = ResponseGenerator(self.model_manager.get_llm())
        
        # Build FAISS index
        print("Building FAISS index...")
        index_data = self.vector_store.build_index()
        self.model_manager.index = index_data["index"]
        self.model_manager.texts = index_data["texts"]
        self.model_manager.metadata = index_data["metadata"]
        
        print("✓ Chatbot initialized successfully!\n")
    
    def answer(self, query: str) -> Dict:
        """
        Main entry point for query processing.
        
        Workflow:
        1. Normalize query - extract entities
        2. Check if clarification needed
        3. Classify intent
        4. Route to appropriate handler:
           - Numerical: Stats tool only
           - Descriptive: RAG + LLM
           - Hybrid: RAG + Stats (aligned) + LLM
        
        Args:
            query: User's natural language question
        
        Returns:
            Dictionary with intent, answer, and metadata
        """
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}\n")
        
        # Step 1: Extract entities
        print("Step 1: Normalizing query...")
        normalized = self.query_processor.normalize_query(query)
        print(f"  Extracted: {normalized}\n")
        
        # Step 2: Check ambiguity
        print("Step 2: Checking for ambiguity...")
        clarification = self.query_processor.check_ambiguity(normalized)
        
        if clarification["needed"]:
            print("  → Clarification required\n")
            return {
                "intent": "clarification",
                "answer": clarification["message"],
                "normalized": normalized,
                "source": "clarification_gate"
            }
        
        print("  → No clarification needed\n")
        
        # Step 3: Classify intent
        print("Step 3: Classifying intent...")
        intent = self.query_processor.classify_intent(query)
        print(f"  Intent: {intent}\n")
        
        # Route based on intent
        if intent == "numerical":
            return self._handle_numerical(query, normalized)
        elif intent == "descriptive":
            return self._handle_descriptive(query, normalized)
        else:
            return self._handle_hybrid(query, normalized)
    
    def _handle_numerical(self, query: str, normalized: Dict) -> Dict:
        """Handle pure numerical queries."""
        print("Step 4: Processing numerical query...\n")
        
        parsed_query = self.stats_handler.parse_numerical_query(query, normalized)
        # 2️⃣ ✅ VALIDATION GUARD (ADD THIS)
        if parsed_query.get("year") and not (2020 <= parsed_query["year"] <= 2024):
            return {
            "intent": "clarification",
            "answer": (
                "Data is available only for India Test Cricket from 2020 to 2024. "
                "Please specify a valid year within this range."
            ),
            "source": "validation_guard"
        }

        if not parsed_query.get("player") and parsed_query.get("metric") != "team_level":
            return {
            "intent": "clarification",
            "answer": (
                "Please specify a player name for this statistical query "
                "(e.g., Rishabh Pant, Jasprit Bumrah)."
            ),
            "source": "validation_guard"
        }

        stats_result = self.stats_handler.execute_stats_query(parsed_query)
        
        print("✓ Answer generated from Stats Tool\n")
        
        return {
            "intent": "numerical",
            "answer": stats_result,
            "normalized": normalized,
            "parsed": parsed_query,
            "source": "stats_tool"
        }
    
    def _handle_descriptive(self, query: str, normalized: Dict) -> Dict:
        """Handle narrative/descriptive queries."""
        print("Step 4: Processing descriptive query...\n")
        
        chunks = self.vector_store.retrieve_chunks(query, k=5)
        print(f"  Retrieved {len(chunks)} chunks\n")
        
        context = self.response_generator.build_context(chunks, self.stats_handler)
        answer = self.response_generator.generate_descriptive(query, context)
        
        print("✓ Answer generated from RAG + LLM\n")
        
        return {
            "intent": "descriptive",
            "answer": answer,
            "normalized": normalized,
            "chunks_used": len(chunks),
            "source": "rag"
        }
    
    def _handle_hybrid(self, query: str, normalized: Dict) -> Dict:
        """Handle queries needing both stats and narrative."""
        print("Step 4: Processing hybrid query...\n")
        
        # Get RAG context first
        chunks = self.vector_store.retrieve_chunks(query, k=5)
        print(f"  Retrieved {len(chunks)} chunks")
        
        # Extract match IDs for scope alignment
        match_ids = list(set(c['metadata']['match_id'] for c in chunks))
        print(f"  Extracted {len(match_ids)} match IDs\n")
        
        # Get statistics (optionally filtered to match_ids)
        parsed_query = self.stats_handler.parse_numerical_query(query, normalized)
        stats_result = self.stats_handler.execute_stats_query(parsed_query, match_ids=match_ids)
        
        # Generate combined response
        context = self.response_generator.build_context(chunks, self.stats_handler)
        answer = self.response_generator.generate_hybrid(query, stats_result, context)
        
        print("✓ Answer generated from Stats + RAG + LLM (aligned)\n")
        
        return {
            "intent": "hybrid",
            "answer": answer,
            "normalized": normalized,
            "parsed": parsed_query,
            "chunks_used": len(chunks),
            "match_ids": match_ids,
            "source": "hybrid"
        }
    
    def ask(self, query: str):
        """Convenience method for interactive use."""
        result = self.answer(query)
        print(f"{'='*70}")
        print("ANSWER:")
        print(f"{'='*70}")
        print(result['answer'])
        print(f"\n{'='*70}")
        print(f"Source: {result['source']} | Intent: {result['intent']}")
        print(f"{'='*70}\n")
        return result


# ============================================================================
# SECTION 9: USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize chatbot
    chatbot = CricketChatbot()
    
    # Example usage
    print("\n" + "="*70)
    print("CRICKET RAG CHATBOT - READY")
    print("="*70)
    
    # Test query
    chatbot.ask("How many runs did Rishabh Pant score in 2021?")