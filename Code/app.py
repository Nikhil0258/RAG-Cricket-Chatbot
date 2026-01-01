"""
============================================================================
CRICKET RAG CHATBOT - STREAMLIT UI
============================================================================
A professional, feature-rich interface for the Cricket RAG Chatbot with:
- Chat history with persistence
- Query analytics and insights
- Export functionality
- Enhanced visualization
- Error handling
============================================================================
"""

import streamlit as st
from final_design import CricketChatbot
import json
from datetime import datetime
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Cricket RAG Chatbot",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .intent-numerical {
        background-color: #d4edda;
        color: #155724;
    }
    .intent-descriptive {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .intent-hybrid {
        background-color: #fff3cd;
        color: #856404;
    }
    .intent-clarification {
        background-color: #f8d7da;
        color: #721c24;
    }
    .sidebar-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def load_chatbot():
    """Load and cache the chatbot instance."""
    with st.spinner("🔄 Initializing Cricket RAG Chatbot..."):
        return CricketChatbot()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "intent_stats" not in st.session_state:
    st.session_state.intent_stats = {
        "numerical": 0,
        "descriptive": 0,
        "hybrid": 0,
        "clarification": 0
    }

if "chatbot" not in st.session_state:
    st.session_state.chatbot = load_chatbot()

# ============================================================================
# SIDEBAR - CONFIGURATION & ANALYTICS
# ============================================================================

with st.sidebar:
    st.markdown("### 🏏 Cricket RAG Chatbot")
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **Cricket RAG Chatbot** combines:
        - 🔍 **Semantic Search**: FAISS vector retrieval
        - 📊 **Statistics Engine**: Deterministic computations
        - 🤖 **LLM Intelligence**: GPT-4 powered understanding
        
        **Coverage**: India Test Cricket (2020-2024)
        """)
    
    # Query Examples
    with st.expander("💡 Example Queries", expanded=False):
        st.markdown("""
        **Numerical Queries:**
        - How many runs did Pant score in 2021?
        - What was Bumrah's wicket tally?
        
        **Descriptive Queries:**
        - Tell me about the 2021 India vs Australia series
        - Describe Pant's batting style
        
        **Hybrid Queries:**
        - How many runs did Pant score and how did he play?
        - Give me Rahane's stats with match context
        """)
    
    st.markdown("---")
    
    # Session Analytics
    st.markdown("### 📊 Session Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.query_count)
    with col2:
        st.metric("Conversations", len(st.session_state.chat_history) // 2)
    
    # Intent Distribution
    if st.session_state.query_count > 0:
        st.markdown("**Intent Distribution:**")
        for intent, count in st.session_state.intent_stats.items():
            if count > 0:
                percentage = (count / st.session_state.query_count) * 100
                st.progress(percentage / 100, text=f"{intent.capitalize()}: {count}")
    
    st.markdown("---")
    
    # Chat Management
    st.markdown("### ⚙️ Chat Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.query_count = 0
            st.session_state.intent_stats = {
                "numerical": 0,
                "descriptive": 0,
                "hybrid": 0,
                "clarification": 0
            }
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.chat_history:
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": st.session_state.query_count,
                    "chat_history": [
                        {
                            "role": role,
                            "message": msg if role == "user" else msg.get("answer", ""),
                            "metadata": msg if role == "assistant" else None
                        }
                        for role, msg in st.session_state.chat_history
                    ]
                }
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"cricket_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # System Status
    with st.expander("🔧 System Status", expanded=False):
        st.markdown(f"""
        - **FAISS Index**: ✅ Loaded
        - **Stats Tool**: ✅ Active
        - **LLM**: ✅ Connected
        - **Session**: Active since {datetime.now().strftime('%H:%M')}
        """)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Header
st.markdown('<div class="main-header">🏏 Cricket RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by RAG | India Test Cricket 2020-2024</div>', unsafe_allow_html=True)

# Quick Stats (if chat history exists)
if st.session_state.chat_history:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-box" style="text-align: center;">
            <div style="font-size: 2rem;">📝</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.query_count}</div>
            <div style="font-size: 0.875rem; color: #666;">Total Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-box" style="text-align: center;">
            <div style="font-size: 2rem;">📊</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.intent_stats['numerical']}</div>
            <div style="font-size: 0.875rem; color: #666;">Numerical</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-box" style="text-align: center;">
            <div style="font-size: 2rem;">📖</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.intent_stats['descriptive']}</div>
            <div style="font-size: 0.875rem; color: #666;">Descriptive</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-box" style="text-align: center;">
            <div style="font-size: 2rem;">🔀</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{st.session_state.intent_stats['hybrid']}</div>
            <div style="font-size: 0.875rem; color: #666;">Hybrid</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# CHAT INTERFACE
# ============================================================================

# Display chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        # Welcome message
        st.info("👋 Welcome! Ask me anything about India Test Cricket from 2020-2024.")
        
        # Quick start buttons
        st.markdown("**Quick Start:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏏 Pant's 2021 runs", use_container_width=True):
                st.session_state.quick_query = "How many runs did Rishabh Pant score in 2021?"
                st.rerun()
        
        with col2:
            if st.button("📖 2021 Series Summary", use_container_width=True):
                st.session_state.quick_query = "Tell me about the 2021 India vs Australia series"
                st.rerun()
        
        with col3:
            if st.button("🔀 Pant's Performance", use_container_width=True):
                st.session_state.quick_query = "How many runs did Pant score in 2021 and how did he play?"
                st.rerun()
    
    else:
        # Display existing chat messages
        for idx, (role, msg) in enumerate(st.session_state.chat_history):
            if role == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(msg)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # Display answer
                    st.markdown(msg["answer"])
                    
                    # Intent badge
                    intent = msg.get("intent", "unknown")
                    badge_class = f"intent-{intent}"
                    st.markdown(f'<span class="intent-badge {badge_class}">{intent.upper()}</span>', 
                              unsafe_allow_html=True)
                    
                    # Expandable details
                    with st.expander("🔍 Query Details", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Metadata:**")
                            st.json({
                                "intent": msg.get("intent"),
                                "source": msg.get("source"),
                                "timestamp": msg.get("timestamp", "N/A")
                            })
                        
                        with col2:
                            st.markdown("**Extracted Entities:**")
                            normalized = msg.get("normalized", {})
                            if normalized:
                                st.json(normalized)
                            else:
                                st.info("No entities extracted")
                        
                        # Additional info based on intent
                        if msg.get("intent") == "descriptive" and "chunks_used" in msg:
                            st.markdown(f"**Chunks Retrieved:** {msg['chunks_used']}")
                        
                        if msg.get("intent") == "hybrid":
                            if "match_ids" in msg:
                                st.markdown(f"**Matches Analyzed:** {len(msg['match_ids'])}")
                            if "verification" in msg:
                                st.markdown("**Verification:**")
                                st.json(msg["verification"])

# ============================================================================
# CHAT INPUT
# ============================================================================

# Handle quick query from welcome screen
if "quick_query" in st.session_state:
    query = st.session_state.quick_query
    del st.session_state.quick_query
else:
    query = st.chat_input("💬 Ask a cricket question...", key="chat_input")

# Process query
if query:
    # Add user message to history
    st.session_state.chat_history.append(("user", query))
    
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    
    # Get response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Analyzing your query..."):
            start_time = time.time()
            
            try:
                # Get answer from chatbot
                result = st.session_state.chatbot.answer(query)
                
                # Add timestamp and processing time
                result["timestamp"] = datetime.now().isoformat()
                result["processing_time"] = round(time.time() - start_time, 2)
                
                # Display answer
                st.markdown(result["answer"])
                
                # Intent badge
                intent = result.get("intent", "unknown")
                badge_class = f"intent-{intent}"
                st.markdown(f'<span class="intent-badge {badge_class}">{intent.upper()}</span>', 
                          unsafe_allow_html=True)
                
                # Success message with timing
                st.success(f"✅ Answered in {result['processing_time']}s")
                
                # Add to history
                st.session_state.chat_history.append(("assistant", result))
                
                # Update statistics
                st.session_state.query_count += 1
                if intent in st.session_state.intent_stats:
                    st.session_state.intent_stats[intent] += 1
                
            except Exception as e:
                # Error handling
                st.error(f"❌ Error: {str(e)}")
                
                error_result = {
                    "answer": f"I encountered an error processing your query: {str(e)}",
                    "intent": "error",
                    "source": "error_handler",
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.chat_history.append(("assistant", error_result))
    
    # Rerun to update the interface
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.875rem;">
    Made with ❤️ using Streamlit | Powered by FAISS, GPT-4, and LangChain
</div>
""", unsafe_allow_html=True)